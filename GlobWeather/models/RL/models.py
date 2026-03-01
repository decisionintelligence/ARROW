from scipy.special import h1vp
from torch import nn
import torch
from timm.models.vision_transformer import Mlp
import numpy as np
from xformers.ops import memory_efficient_attention, unbind


class MemEffAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class QNet(nn.Module):
    def __init__(self, target_time, data_freq=6, 
                 month_dim=32, week_dim=16, time_dim=32, target_dim=32,
                 hidden_dim=1024, state_dim=1024, weather_dim=1024,
                 n_actions=3):
        super().__init__()
        if isinstance(target_time, list):
            # One QNet for all descision
            self.target_list = torch.tensor(target_time)
            self.target_time = max(target_time)
            self.target_time_embed = nn.Embedding(len(self.target_list), target_dim)
        else:
            # One QNet for one descision
            self.target_time = target_time
            self.target_time_embed = None
        
        self.month_embed = nn.Embedding(13, month_dim)
        self.week_embed = nn.Embedding(5, week_dim)
        time_point = self.target_time // data_freq + 1
        self.travel_time_embed = nn.Embedding(time_point, time_dim)
        self.rest_time_embed = nn.Embedding(time_point, time_dim)
        self.data_freq = data_freq

        if self.target_time_embed is not None:
            in_features = month_dim + week_dim + time_dim * 2 + target_dim
        else:
            in_features = month_dim + week_dim + time_dim * 2

        self.hidden_fc = nn.Linear(in_features, state_dim)
        self.state_mlp = Mlp(in_features=state_dim, hidden_features=hidden_dim, out_features=state_dim)
        self.action_head = nn.Linear(state_dim, n_actions)

    def forward(self, state, weather_rep):
        time_idx = state[:, 1]
        travel_time = state[:, 2]
        rest_time = state[:, 3]
        target_time = state[:, 4]

        month = time_idx // 120
        week = time_idx % 120 // 28
        travel_time = travel_time // self.data_freq
        rest_time = rest_time // self.data_freq

        month_embed = self.month_embed(month)
        week_embed = self.week_embed(week)
        travel_time_embed = self.travel_time_embed(travel_time)
        rest_time_embed = self.rest_time_embed(rest_time)
        state_embed = torch.cat([month_embed, week_embed, travel_time_embed, rest_time_embed], dim=1)

        if self.target_time_embed is not None:
            self.target_list = self.target_list.to(target_time.device)
            target_time = (target_time.unsqueeze(1) == self.target_list.unsqueeze(0)).nonzero()[:, 1]
            target_time_embed = self.target_time_embed(target_time)
            state_embed = torch.cat([state_embed, target_time_embed], dim=1)

        h = self.hidden_fc(state_embed)
        h = self.state_mlp(h)
        weather_rep = weather_rep.mean(dim=1)
        fused = h + weather_rep
        actions = self.action_head(fused)

        return actions