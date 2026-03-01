import numpy as np
import os
from metrics.metrics import *
from GlobWeather.models.hub.Arrow import arrow
from GlobWeather.models.RL.RL_utils import variables
from GlobWeather.data.iterative_dataset import get_data_given_path
from torchvision.transforms import transforms
from typing import Union
from GlobWeather.utils.data_utils import CONSTANTS
from torch import nn
from dataclasses import dataclass


def get_next_time(root_dir, year, inp_file_idx, steps):
    # year: current year
    # inp_file_idx: file index of the input in the current year
    # steps: number of steps forward
    out_file_idx = inp_file_idx + steps
    out_path = os.path.join(root_dir, f'{year}_{out_file_idx:04}.h5')
    if not os.path.exists(out_path):
        # cross-year
        for i in range(steps):
            out_file_idx = inp_file_idx + i
            out_path = os.path.join(root_dir, f'{year}_{out_file_idx:04}.h5')
            if os.path.exists(out_path):
                max_step_forward = i
        remaining_steps = steps - max_step_forward
        next_year = year + 1
        return next_year, remaining_steps-1
    else:
        return year, out_file_idx

@dataclass
class Env_State:
    year: int = None
    time_idx: int = None
    travel_time: int = None
    rest_time: int = None
    target_time: int = None
    cur_weather_z: torch.Tensor = None

    def __getitem__(self, idx):
        return (self.year, self.time_idx, self.travel_time, self.rest_time, self.target_time)[idx]


class WeatherForcast(nn.Module):
    # for fine-tune and weather environment
    def __init__(
        self, 
        net: arrow,
        pretrained_path: str = None):
        super().__init__()
        self.net = net
        self.load_pretrained_weights(pretrained_path)
    
    def load_pretrained_weights(self, pretrained_path):
        if pretrained_path.startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_path)
        else:
            checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))
        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        state_dict = checkpoint["state_dict"]
        msg = self.load_state_dict(state_dict)
        print(msg)

        for param in self.net.parameters():
            param.requires_grad = False
        
        self.net.head.requires_grad_(True)
    
    def set_transforms(self, inp_transform, diff_transform):
        self.inp_transform = inp_transform
        self.reverse_inp_transform = self.get_reverse_transform(inp_transform)
        
        self.diff_transform = diff_transform
        self.reverse_diff_transform = {
            k: self.get_reverse_transform(v) for k, v in diff_transform.items()
        }
    
    def get_reverse_transform(self, transform):
        mean, std = transform.mean, transform.std
        std_reverse = 1 / std
        mean_reverse = -mean * std_reverse
        return transforms.Normalize(mean_reverse, std_reverse)
    
    def replace_constant(self, yhat, out_variables):
        for i in range(yhat.shape[1]):
            if out_variables[i] in CONSTANTS:
                yhat[:, i] = 0.0
        return yhat

    def pad(self, x: torch.Tensor):
        h = x.shape[-2]
        # Calculate the pad size for the height if it's not divisible by the patch size
        if h % self.net.patch_size != 0:
            pad_size = self.net.patch_size - h % self.net.patch_size
            # Only pad the top
            padded_x = torch.nn.functional.pad(x, (0, 0, pad_size, 0), 'constant', 0)
        else:
            padded_x = x
            pad_size = 0
        return padded_x, pad_size

    def forward(self, x: torch.Tensor, variables, interval) -> torch.Tensor:
        padded_x, pad_size = self.pad(x)
        output = self.net(padded_x, variables, interval)[:, :, pad_size:]
        return output

    def forward_onestep(self, x: torch.Tensor, variables, interval):
        interval_tensor = torch.Tensor([interval]).to(device=x.device, dtype=x.dtype) / 10.0
        interval_tensor = interval_tensor.repeat(x.shape[0])
        pred_diff_norm = self(x, variables, interval_tensor) # diff in the normalized space
        pred_diff_norm = self.replace_constant(pred_diff_norm, variables)
        pred_diff = self.reverse_diff_transform[interval](pred_diff_norm) # diff in the original space
        pred = self.reverse_inp_transform(x) + pred_diff # prediction in the original space
        x = self.inp_transform(pred) # prediction in the normalized space

        return x, pred_diff_norm

    @torch.no_grad
    def forward_validation(self, x: torch.Tensor, variables, interval, steps):
        # x: initial condition, B, V, H, W
        # variables: list of variable names
        # interval: scalar value, e.g., 6, use the same interval across the batch
        # steps: scalar value, e.g., 24, number of autoregressive steps

        # x is always in the normalized input space
        interval_tensor = torch.Tensor([interval]).to(device=x.device, dtype=x.dtype) / 10.0
        interval_tensor = interval_tensor.repeat(x.shape[0])
        for _ in range(steps):
            pred_diff = self(x, variables, interval_tensor) # diff in the normalized space
            pred_diff = self.replace_constant(pred_diff, variables)
            pred_diff = self.reverse_diff_transform[interval](pred_diff) # diff in the original space
            pred = self.reverse_inp_transform(x) + pred_diff # prediction in the original space
            x = self.inp_transform(pred) # prediction in the normalized space
        return x
    
    @torch.no_grad
    def get_weather_representations(self, x: torch.Tensor, variables) -> torch.Tensor:
        x, _ = self.pad(x)     # padding
        all_variables = variables + self.net.const_variables
        if len(self.net.const_variables) != 0:
            x = self.net.create_input(x)
        z = self.net.embedding(x, all_variables)
        # z = self.net.embed_norm_layer(z)
        return z


class WeatherEnv:
    def __init__(self, root_dir, split, device, net, pretrained_path,
                 data_freq=6, start_year=2010, end_year=2017,
                 targets: Union[int, list]=[24, 48, 72, 96, 120, 144, 168],
                 step_punish=0.01, reward_scale_factor=3):
        self.targets = targets
        self.start_year = start_year
        self.end_year = end_year
        self.data_freq = data_freq

        self.root_dir = os.path.join(root_dir, split)
        self.device = device
        self.set_weather_model(net, pretrained_path, root_dir)
        self.set_metrics(root_dir)

        self.cur_weather = None
        self.state = Env_State()
        self.action_space = [6, 12, 24]
        self.step_punish = step_punish
        self.reward_scale_factor = reward_scale_factor

    def reset(self, year=None, time_idx=None, target_time=None):
        # if specific, it means RL inference
        if target_time is None: 
            self.target_time = self.set_target()
        else:
            if isinstance(self.targets, list): assert target_time in self.targets, "invalid target time"
            self.target_time = target_time
        
        if year is None: year = np.random.randint(self.start_year, self.end_year)
        if time_idx is None: time_idx = np.random.randint(0, 1200)
        travel_time = 0
        rest_time = self.target_time
        
        self.cur_weather = get_data_given_path(os.path.join(self.root_dir, f'{year}_{time_idx:04}.h5'), variables)
        self.cur_weather = torch.from_numpy(self.cur_weather).to(self.device).unsqueeze(0)
        self.cur_weather = self.weather_model.inp_transform(self.cur_weather)
        cur_weather_z = self.weather_model.get_weather_representations(self.cur_weather, variables)

        self.state = Env_State(year, time_idx, travel_time, rest_time, self.target_time, cur_weather_z)

        return self.state

    def step(self, action_idx):
        action = self.action_space[action_idx]

        with torch.no_grad():
            pred, _ = self.weather_model.forward_onestep(self.cur_weather, variables, action)
        self.cur_weather = pred

        year, time_idx = self.state[0], self.state[1]
        travel_time = self.state[2]
        rest_time = self.state[3]

        year, time_idx = get_next_time(self.root_dir, year, time_idx, steps=action // self.data_freq)
        travel_time += action
        rest_time -= action

        assert travel_time + rest_time == self.target_time, "travel_time + rest_time != target_time"
        
        cur_weather_z = self.weather_model.get_weather_representations(self.cur_weather, variables)  # B(1) x L x D
        new_state = Env_State(year, time_idx, travel_time, rest_time, self.target_time, cur_weather_z)
        self.state = new_state
        
        if rest_time <= 0: 
            terminated = True
            reward = self.get_reward(year, time_idx, pred)*self.reward_scale_factor
        else:
            terminated = False
            reward = self.get_reward(year, time_idx, pred)/self.reward_scale_factor - self.step_punish
        
        truncated = False
        info = {}
        return new_state, reward, terminated, truncated, info
    
    def fine_tune_step(self, action_idx, sign_no_grad=False):
        action = self.action_space[action_idx]

        # 1. get next weather state
        if sign_no_grad:
            with torch.no_grad():
                pred, pred_diff_norm = self.weather_model.forward_onestep(self.cur_weather, variables, action)
        else:
            pred, pred_diff_norm = self.weather_model.forward_onestep(self.cur_weather, variables, action)
        self.cur_weather = pred

        # 2. update state
        year, time_idx = self.state.year, self.state.time_idx
        travel_time = self.state.travel_time
        rest_time = self.state.rest_time

        year, time_idx = get_next_time(self.root_dir, year, time_idx, steps=action // self.data_freq)
        travel_time += action
        rest_time -= action

        assert travel_time + rest_time == self.target_time, "travel_time + rest_time != target_time"
        
        cur_weather_z = self.weather_model.get_weather_representations(self.cur_weather, variables)  # B(1) x L x D
        new_state = Env_State(year, time_idx, travel_time, rest_time, self.target_time, cur_weather_z)
        self.state = new_state

        # 3. get reward and check done
        if rest_time <= 0: 
            terminated = True
            reward = self.get_reward(year, time_idx, pred)*self.reward_scale_factor
        else:
            terminated = False
            reward = self.get_reward(year, time_idx, pred)/self.reward_scale_factor - self.step_punish
        
        truncated = False
        info = {}
        return new_state, reward, terminated, truncated, pred_diff_norm
    
    def get_reward(self, year, time_idx, pred):
        truth = get_data_given_path(os.path.join(self.root_dir, f'{year}_{time_idx:04}.h5'), variables)
        truth = torch.from_numpy(truth).to(self.device).unsqueeze(0)
        # pred = self.weather_model.reverse_inp_transform(pred)
        truth = self.weather_model.inp_transform(truth)
        reward = self.metric(pred, truth)[0]
        return -reward.item()
    
    def set_weather_model(self, net, pretrained_path, normalize_path):
        model = WeatherForcast(net, pretrained_path=pretrained_path).to(self.device)
        # model.eval()

        normalize_mean = dict(np.load(os.path.join(normalize_path, "normalize_mean.npz")))
        normalize_mean = np.concatenate([normalize_mean[v] for v in variables], axis=0)
        normalize_std = dict(np.load(os.path.join(normalize_path, "normalize_std.npz")))
        normalize_std = np.concatenate([normalize_std[v] for v in variables], axis=0)
        inp_transform = transforms.Normalize(normalize_mean, normalize_std)

        out_transforms = {}
        for l in [6, 12, 24]:
            normalize_diff_std = dict(np.load(os.path.join(normalize_path, f"normalize_diff_std_{l}.npz")))
            normalize_diff_std = np.concatenate([normalize_diff_std[v] for v in variables], axis=0)
            out_transforms[l] = transforms.Normalize(np.zeros_like(normalize_diff_std), normalize_diff_std)
        model.set_transforms(inp_transform, out_transforms)
        self.weather_model = model

        print("set weather model success")
    
    def set_metrics(self, root_dir):
        clim = torch.randn(69, 128)
        lat = np.load(os.path.join(root_dir, "lat.npy"))
        lon = np.load(os.path.join(root_dir, "lon.npy"))
        metainfo = MetricsMetaInfo(lat, lon, clim)
        self.metric = LatWeightedRMSE(aggregate_only=False, metainfo=metainfo)

        print("set metrics success")
    
    def set_target(self):
        if isinstance(self.targets, int):
            return self.targets
        elif isinstance(self.targets, list):
            return np.random.choice(self.targets)
        else:
            raise ValueError("target must be an int or a list")