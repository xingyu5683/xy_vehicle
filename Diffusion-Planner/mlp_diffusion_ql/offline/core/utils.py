#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : utils.py
@Description: 辅助函数和工具类
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def flatten_obs(obs_dict):
    """
    将carla-v0_test1环境的观测字典展平为向量
    
    观测格式:
    - ego_state: (9,) - 自车状态
    - nearby_vehicles: (5, 4) - 周围车辆矩阵 (5个车辆 × 4维)，需要展平为 (20,)
    - waypoints: (12, 3) - 路径点矩阵 (12个waypoints × 3维)，需要展平为 (36,)
    - lane_info: (2,) - 车道信息
    
    总计: 9 + 20 + 36 + 2 = 67维
    """
    return np.concatenate([
        obs_dict['ego_state'],                    # 9 dimensions
        obs_dict['nearby_vehicles'].flatten(),    # (5, 4) -> 20 dimensions
        obs_dict['waypoints'].flatten(),          # (12, 3) -> 36 dimensions
        obs_dict['lane_info']                     # 2 dimensions
    ]).astype(np.float32)


class SinusoidalPosEmb(nn.Module):
    """正弦位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def extract(a, t, x_shape):
    """从张量中提取特定时间步的值"""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """余弦beta调度"""
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2, dtype=torch.float32):
    """线性beta调度"""
    betas = np.linspace(beta_start, beta_end, timesteps)
    return torch.tensor(betas, dtype=dtype)


def vp_beta_schedule(timesteps, dtype=torch.float32):
    """VP beta调度"""
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype=dtype)


class WeightedL2(nn.Module):
    """加权L2损失"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, targ, weights=1.0):
        loss = F.mse_loss(pred, targ, reduction='none')
        weighted_loss = (loss * weights).mean()
        return weighted_loss


class EMA:
    """指数移动平均"""
    def __init__(self, beta):
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

