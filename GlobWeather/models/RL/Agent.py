import torch
import numpy as np
from GlobWeather.models.RL.Env import WeatherEnv
import time
from collections import deque, namedtuple
from typing import Iterator, Tuple
from torch import nn
from torch.utils.data.dataset import IterableDataset
from GlobWeather.models.RL.models import QNet

Experience = namedtuple(
    "Experience",
    field_names=["state", "action_idx", "reward", "done", "new_state"],
)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience: Experience) -> None:
        if experience.state.cur_weather_z is not None:
            experience.state.cur_weather_z = experience.state.cur_weather_z.cpu()
            experience.new_state.cur_weather_z = experience.new_state.cur_weather_z.cpu()
        self.buffer.append(experience)

    def sample(self, size: int):
        indices = np.random.choice(len(self.buffer), size, replace=False)
        states_1 = [self.buffer[idx][0][:] for idx in indices]
        states_2 = torch.concat([self.buffer[idx][0].cur_weather_z for idx in indices], dim=0)
        actions, rewards, dones = zip(*(self.buffer[idx][1:-1] for idx in indices))
        next_states_1 = [self.buffer[idx][-1][:] for idx in indices]
        next_states_2 = torch.concat([self.buffer[idx][-1].cur_weather_z for idx in indices], dim=0)

        return (
            (np.array(states_1), states_2),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            (np.array(next_states_1), next_states_2),
        )
    
    def clear(self):
        self.buffer.clear()
    

class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time

    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Tuple]:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield (states[0][i], states[1][i]), actions[i], rewards[i], dones[i], (new_states[0][i], new_states[1][i])


class Agent:
    def __init__(self, env, replay_buffer: ReplayBuffer, list_intevals=[6, 12, 24]):
        self.env = env
        self.replay_buffer = replay_buffer
        self.action_space = list_intevals
        self.reset()
    
    def reset(self, year=None, time_idx=None, target_time=None):
        self.state = self.env.reset(year, time_idx, target_time)
    
    def get_action_dqn(self, net: nn.Module, epsilon: float, device='cpu'):
        # ! valid action
        # note: state is a list
        rest_time = self.state[3]
        valid_action = [i for i, action in enumerate(self.action_space) if rest_time - action >= 0]
        if np.random.random() < epsilon:
            action = np.random.choice(valid_action)
        else:
            state_time = torch.tensor(self.state[0:]).unsqueeze(0).to(device)
            state_weather = self.state.cur_weather_z.to(device)
            if isinstance(net, QNet):
                q_values = net(state_time, state_weather)
            else:
                raise NotImplementedError
            _, action = torch.max(q_values[:, valid_action], dim=1)
            action = valid_action[action.item()]
        
        return action
    
    @torch.no_grad()
    def play_step(self, net: nn.Module, epsilon: float, device='cpu'):
        action = self.get_action_dqn(net, epsilon, device)
        new_state, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated: done = True
        else: done = False

        exp = Experience(self.state, action, reward, done, new_state)
        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done

    @torch.no_grad()
    def run_episode(self, net: nn.Module, epsilon: float, device='cpu',
                    year=None, time_idx=None, target_time=None, get_return=False):
        self.reset(year, time_idx, target_time)
        done = False
        action_list = []
        total_reward = 0
        while not done:
            action = self.get_action_dqn(net, epsilon, device)
            self.state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated: done = True
            action_list.append(self.action_space[action])
        
        if get_return:
            return action_list, total_reward
        else:
            return action_list
    
    def get_decision(self, net: nn.Module, epsilon: float, device='cpu',
                    year=None, time_idx=None, target_time=None, 
                    get_return=False, max_gradient_step=6):
        self.reset(year, time_idx, target_time)
        done = False
        Max_T = self.env.target_time // self.env.data_freq
        action_list = []
        rollout_pd = []
        total_reward = 0
        cur_step = 1
        sign_no_grad = False
        while not done:
            if cur_step > max_gradient_step:
                sign_no_grad = True
            action = self.get_action_dqn(net, epsilon, device)
            self.state, reward, terminated, truncated, pred_diff_norm = self.env.fine_tune_step(action, sign_no_grad)
            total_reward += reward
            if terminated or truncated: done = True
            action_list.append(self.action_space[action])
            rollout_pd.append(pred_diff_norm)
            cur_step += 1
        
        action_list.extend([0] * (Max_T - len(action_list)))
        rollout_pd.extend([torch.zeros_like(rollout_pd[-1])] * (Max_T - len(rollout_pd)))
        rollout_pd = torch.stack(rollout_pd)
        rollout_pd = rollout_pd.transpose(0, 1)

        if get_return:
            return action_list, rollout_pd, total_reward
        else:
            return action_list, rollout_pd