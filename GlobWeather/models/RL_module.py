from typing import Any, List, Tuple, Dict, Optional, Union
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from lightning import LightningModule, Trainer
from torchvision.transforms import transforms
from lightning.pytorch.utilities import rank_zero_only
import numpy as np
from torch.utils.data import DataLoader

from GlobWeather.models.hub.Arrow import arrow
from GlobWeather.utils.metrics import (
    lat_weighted_mse,
    lat_weighted_acc,
    lat_weighted_rmse,
)
from GlobWeather.utils.data_utils import CONSTANTS, WEIGHT_DICT
from GlobWeather.models.RL.Env import WeatherEnv, WeatherForcast
from GlobWeather.models.RL.Agent import ReplayBuffer, Agent, RLDataset
from GlobWeather.models.RL.models import QNet
from GlobWeather.models.RL.RL_dataset import ERA5MultiLeadtimeDataset, collate_fn_val, Fine_tune_RLDataset
from GlobWeather.models.RL.RL_utils import variables
import os
import xarray as xr
CLIM_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_cloud_cover",

    # pressure level variables
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
]


class RLModule(LightningModule):
    def __init__(
        self,
        weather_net: arrow,
        qnet: QNet,
        target_net: QNet,
        weather_path: str = None,
        dqn_path: str = None,
        root_dir: str = "./dataset/",
        agent_lr: float = 5e-4,
        fine_tune_lr: float = 5e-7,
        batch_size: int = 64,
        sync_rate: int = 10,
        fine_tune_rate: int = 10,
        buffer_size: int = 1000,
        targets: Union[int, list] = 72,
        step_punish: float = 0.01,
        reward_scale_factor: float = 3,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        gamma: float = 0.99,
        refresh_size: int = 300,
        dataset_sample_size: int = 200,
        data_freq: int = 6,
        max_optimize_step: int = 8,
        start_year: int = 2010,
        end_year: int = 2016,
        env_device_num: int = 7
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['weather_net', "qnet", "target_net"])
        self.automatic_optimization = False

        device = torch.device("cuda:{}".format(env_device_num))

        self.env = WeatherEnv(root_dir, "train", device, weather_net, weather_path, 
                              start_year=start_year, end_year=end_year, targets=targets,
                              step_punish=step_punish, reward_scale_factor=reward_scale_factor)
        self.inp_transformer = self.env.weather_model.inp_transform
        self.reverse_inp_transform = self.env.weather_model.reverse_inp_transform
        self.diff_transform = self.env.weather_model.diff_transform
        self.reverse_diff_transform = self.env.weather_model.reverse_diff_transform
        self.finetune_dataset = Fine_tune_RLDataset(os.path.join(root_dir, 'train'), variables,
                                                    self.inp_transformer, self.diff_transform,
                                                    [6, 12, 24], 6)
        self.buffer = ReplayBuffer(buffer_size)
        self.agent = Agent(self.env, self.buffer)
        self.qnet = qnet.to(device)
        self.target_net = target_net.to(device)
        self.total_reward = -3
        self.episode_reward = 0
        self.lat, self.lon = self.get_lat_lon()
        self.refresh_buffer(buffer_size, device)
    
    def get_lat_lon(self):
        lat = np.load(os.path.join(self.hparams.root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.hparams.root_dir, "lon.npy"))
        return lat, lon
    
    def get_epsilon(self, start: int, end: int, frames: int) -> float:
        if self.global_step > frames:
            return end
        return start - (self.global_step / frames) * (start - end)

    def on_train_epoch_start(self):
        self.refresh_buffer(self.hparams.refresh_size, self.device)

    def training_step(self, batch: Any, batch_idx: int):
        dqn_loss = self.dqn_mse_loss(batch)

        optimizer1, optimizer2 = self.optimizers()
        optimizer1.zero_grad()
        self.manual_backward(dqn_loss)
        optimizer1.step()

        if self.global_step % self.hparams.fine_tune_rate == 0:
            B = 1
            batch_year = np.random.randint(self.hparams.start_year, self.hparams.end_year, size=B)
            batch_time_idx = np.random.randint(0, 1200, size=B)
            target_time = self.hparams.targets if isinstance(self.hparams.targets, int) else np.random.choice(self.hparams.targets)
            batch_traj = []
            batch_total_reward = []
            batch_rollout_pd = []
            for i in range(B):
                traj, rollout_pd, total_reward = self.agent.get_decision(self.qnet, 0.01, self.device,
                                                          year=batch_year[i], time_idx=batch_time_idx[i],
                                                          target_time=target_time, get_return=True, max_gradient_step=self.hparams.max_optimize_step)
                batch_traj.append(traj)
                batch_total_reward.append(total_reward / len(traj))
                batch_rollout_pd.append(rollout_pd)
            
            mean_return = np.mean(batch_total_reward)
            batch_traj = np.array(batch_traj) // 6
            # print(batch_traj)

            batch_rollout_pd = torch.concat(batch_rollout_pd, dim=0)
            inp_data, oup_data, _, _, interval_tensors = self.finetune_dataset.get_from_trajectory(batch_year, batch_time_idx, batch_traj)
            fine_tune_loss = self.multi_steps_masked_loss(rollout_pd, oup_data, batch_traj)
            optimizer2.zero_grad()
            self.manual_backward(fine_tune_loss)
            optimizer2.step()

            self.agent.reset()

            if self.logger is not None:
                self.logger.log_metrics(
                    {
                        "return": mean_return,
                        "train/agg_loss": fine_tune_loss.item(),
                    },
                    step=self.global_step
                )
        
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.qnet.state_dict())
        
        self.log_dict(
            {
                "train/dqn_loss": dqn_loss,
                # "total_reward": self.total_reward,
            },
            on_step=True,
            on_epoch=False
        )

    def dqn_mse_loss(self, batch):
        states, actions, rewards, dones, next_states = batch
        if isinstance(self.qnet, QNet):
            state_action_values = self.qnet(*states).gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                next_state_values = self.target_net(*next_states).max(1)[0]
                next_state_values[dones] = 0.0
                next_state_values = next_state_values.detach()
        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)
    
    def multi_steps_masked_loss(self, rollout_pd, rollout_gt, traj):
        rollout_gt = rollout_gt.to(rollout_pd.device)
        # get_mask
        rollout_pd = rollout_pd[:, :self.hparams.max_optimize_step]
        rollout_gt = rollout_gt[:, :self.hparams.max_optimize_step]
        rollout_pd = rollout_pd.flatten(0, 1)
        rollout_gt = rollout_gt.flatten(0, 1)
        loss_dict = lat_weighted_mse(
            rollout_pd,
            rollout_gt,
            variables,
            self.lat,
            weighted=True,
            weight_dict=WEIGHT_DICT
        )
        
        return loss_dict[f"w_mse_aggregate"]

    def refresh_buffer(self, steps: int = 1000, device='cpu'):
        # refresh
        if self.global_step >= 1000:
            for _ in range(steps):
                self.agent.play_step(self.qnet, self.hparams.eps_end, device)  # on-policy
        else:
            for _ in range(steps):
                self.agent.play_step(self.qnet, self.hparams.eps_start, device)  # off-policy
    
    def configure_optimizers(self):
        optimizer_1 = torch.optim.Adam(self.qnet.parameters(), lr=self.hparams.agent_lr)
        optimizer_2 = torch.optim.Adam(self.env.weather_model.net.head.parameters(), lr=self.hparams.fine_tune_lr)
        return [optimizer_1, optimizer_2]
    
    def train_dataloader(self) -> DataLoader:
        dataset = RLDataset(self.buffer, self.hparams.dataset_sample_size)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader
    
    def on_save_checkpoint(self, checkpoint):
        # Remove the target_net's parameters
        if "state_dict" in checkpoint:
            keys_to_remove = [k for k in checkpoint["state_dict"] if "target_net" in k]
            for k in keys_to_remove:
                del checkpoint["state_dict"][k]
            # Also save the weather model head parameters
            weather_head_state_dict = {}
            for k, v in self.env.weather_model.net.head.state_dict().items():
                weather_head_state_dict[f"env.weather_model.net.head.{k}"] = v
            checkpoint["state_dict"].update(weather_head_state_dict)


class RL_InferenceModule(LightningModule):
    def __init__(
        self,
        weather_net: arrow,
        qnet: QNet,
        weather_path: str = None,
        dqn_path: str = None,
        root_dir: str = "./dataset/",
        val_lead_time: list = [48, 72],
        step_punish: float = 0.02,
        data_freq: int = 6,
        start_year: int = 2018,
        end_year: int = 2019,
        env_device_num: int = 7
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['weather_net', "qnet"])
        targets = val_lead_time
        device = torch.device("cuda:{}".format(env_device_num)) 

        self.env = WeatherEnv(root_dir, "test", device, weather_net, weather_path, 
                              start_year=start_year, end_year=end_year, targets=targets,
                              step_punish=step_punish)
        self.env.weather_model.eval()   # for validation
        self.buffer = ReplayBuffer(1)
        self.agent = Agent(self.env, self.buffer)
        self.qnet = qnet

        self.lat = np.load(os.path.join(root_dir, "lat.npy"))
        self.lon = np.load(os.path.join(root_dir, "lon.npy"))

        if dqn_path is not None:
            self.load_pretrained_weights(dqn_path)
    
    def load_pretrained_weights(self, pretrained_path):
        if pretrained_path.startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_path)
        else:
            checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))
        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        state_dict = checkpoint["state_dict"]
        weather_head_state_dict = {}
        keys_to_remove = []
        for k, v in state_dict.items():
            if k.startswith('env.weather_model.net.head.'):
                # Remove 'env.weather_model.net.head.' prefix (26 characters) to get the actual parameter name
                weather_head_state_dict[k[len('env.weather_model.net.head.'):]] = v
                keys_to_remove.append(k)
        # Remove weather model head parameters from state_dict before loading
        for k in keys_to_remove:
            del state_dict[k]
        if weather_head_state_dict:
            self.env.weather_model.net.head.load_state_dict(weather_head_state_dict)
        msg = self.load_state_dict(state_dict)
        print(msg)

    def set_clim(self, clim_dir="/root/dataset/2018_6h_256x128"):
        clim_list = []
        for variable in CLIM_VARIABLES:
            ds = xr.open_dataset(os.path.join(clim_dir, f"{variable}/2018.nc"))
            if 'level' in ds.dims:
                for i in range(len(ds['level'].values)):
                    clim_list.append(torch.from_numpy(ds[variable][:, :, i].values))
            else:
                clim_list.append(torch.from_numpy(ds[variable].values))

        self.clim = torch.stack(clim_list, dim=2)

    def test_step(self, batch: Any, batch_idx: int):
        inp, out, dict_dayofyear, dict_hour, variables, year, time_idx = batch
        dict_clim = {}

        for lead_time in self.hparams.val_lead_time:
            dict_clim[lead_time] = torch.stack([
                self.clim[dict_hour[lead_time][i], dict_dayofyear[lead_time][i]] for i in range(len(inp))
            ], dim=0)

        def get_loss_dict(y, yhat, list_metrics, postfix, clim):
            all_loss_dicts = []
            for metric in list_metrics:
                if metric is lat_weighted_acc:
                    loss_dict = metric(
                        yhat,
                        y,
                        self.env.weather_model.reverse_inp_transform,
                        variables,
                        lat=self.lat,
                        clim=clim,
                        log_postfix=postfix
                    )
                else:
                    loss_dict = metric(
                        yhat,
                        y,
                        self.env.weather_model.reverse_inp_transform,
                        variables,
                        lat=self.lat,
                        log_postfix=postfix,
                        weighted=True,
                        weight_dict=WEIGHT_DICT
                    )
                all_loss_dicts.append(loss_dict)
            
            final_loss_dict = {}
            for d in all_loss_dicts:
                final_loss_dict.update(d)
                
            final_loss_dict = {f"test/{k}": v for k, v in final_loss_dict.items()}
            return final_loss_dict

        for lead_time in self.hparams.val_lead_time:
            all_norm_preds = []

            action_list = self.agent.run_episode(
                self.qnet, 0.01, self.device,
                year, time_idx, lead_time
                )
            pred = self.env.cur_weather.to(self.device)
            print(year, time_idx, lead_time, action_list)
            all_norm_preds.append(pred)

            rl_loss_dict = get_loss_dict(
                pred,
                out[lead_time],
                list_metrics=[lat_weighted_rmse, lat_weighted_acc],
                postfix=f"{lead_time}_hrs_RL",
                clim=dict_clim[lead_time]
            )

            self.log_dict(
                rl_loss_dict,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=inp.shape[0],
            )
            
            action_space = [6, 12, 24]
            for base_interval in action_space:
                if lead_time % base_interval == 0:
                    steps = lead_time // base_interval
                    norm_pred = self.env.weather_model.forward_validation(inp, variables, base_interval, steps)
                    base_loss_dict = get_loss_dict(
                        out[lead_time],
                        norm_pred,
                        list_metrics=[lat_weighted_rmse],
                        postfix=f"{lead_time}_hrs_base_{base_interval}",
                        clim=dict_clim[lead_time]
                    )
                    all_norm_preds.append(norm_pred)
            
                    self.log_dict(
                        base_loss_dict,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                        batch_size=inp.shape[0],
                    )

            mean_norm_pred = torch.stack(all_norm_preds, dim=0).mean(0)

            ens_loss_dict = get_loss_dict(
                mean_norm_pred,
                out[lead_time],
                list_metrics=[lat_weighted_rmse, lat_weighted_acc],
                postfix=f"{lead_time}_hrs_ensemble_mean",
                clim=dict_clim[lead_time]
            )

            self.log_dict(
                ens_loss_dict,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=inp.shape[0]
            )

    def test_dataloader(self) -> DataLoader:
        dataset = ERA5MultiLeadtimeDataset(
            root_dir=os.path.join(self.hparams.root_dir, "test"),
            variables=variables,
            transform=self.env.weather_model.inp_transform,
            list_lead_times=self.hparams.val_lead_time,
            data_freq=self.hparams.data_freq
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=2,
            pin_memory=False,
            collate_fn=collate_fn_val
        )
        return dataloader