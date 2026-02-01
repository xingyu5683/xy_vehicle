#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : data_sampler.py
@Description: 数据采样器
"""
import h5py
import torch
import numpy as np
from .reward_functions import carla_env_reward_function


class DataSampler:
    """数据采样器，适配QL_Diffusion的需求（返回state, action, reward, next_state, done）"""
    def __init__(self, data_file, device='cuda', reward_tune='no', reward_function=None, desired_speed=8.1):
        """
        参数:
            data_file: HDF5数据文件路径
            device: 设备 ('cuda' 或 'cpu')
            reward_tune: 奖励调整方式 ('no', 'normalize', 'iql_antmaze')
            reward_function: 自定义奖励函数，如果提供则使用此函数重新计算reward
                           函数签名: reward_function(state, action, next_state, done, info, desired_speed) -> reward
                           state, action, next_state, done, info都是torch.Tensor
                           reward应该是torch.Tensor，shape为(N, 1)或(N,)
                           如果为None，则使用数据集中的原始reward
            desired_speed: 期望速度 (m/s)，用于自定义奖励函数，默认8.1
        """
        print(f">> 加载数据集: {data_file}")
        
        with h5py.File(data_file, 'r') as f:
            observations = torch.from_numpy(f['observations'][:]).float()
            actions = torch.from_numpy(f['actions'][:]).float()
            rewards = torch.from_numpy(f['rewards'][:]).float()
            
            # 读取next_observations和done（如果存在）
            if 'next_observations' in f:
                next_observations = torch.from_numpy(f['next_observations'][:]).float()
            else:
                # 如果没有next_observations，使用observations[1:]作为next_state
                next_observations = torch.cat([observations[1:], observations[-1:]], dim=0)
            
            if 'done' in f:
                dones = torch.from_numpy(f['done'][:]).float()
            else:
                # 如果没有done，假设所有transition都不是终止状态
                dones = torch.zeros(len(observations))
            
            # 读取info信息（如果存在）
            if 'info' in f:
                infos = torch.from_numpy(f['info'][:]).float()
            else:
                # 如果没有info，创建默认值（全0）
                infos = torch.zeros(len(observations), 3, dtype=torch.float32)
                print(">> 警告: 数据集中没有info信息，将使用默认值（无碰撞、无偏离道路）")
            
            # 读取元数据
            if 'obs_dim' in f.attrs:
                self.obs_dim = f.attrs['obs_dim']
            else:
                self.obs_dim = observations.shape[1]
            
            if 'action_dim' in f.attrs:
                self.action_dim = f.attrs['action_dim']
            else:
                self.action_dim = actions.shape[1]
        
        self.state = observations
        self.action = actions
        self.next_state = next_observations
        self.done = dones.reshape(-1, 1)  # 重塑为 (N, 1)
        self.info = infos  # shape: (N, 3)
        self.size = self.state.shape[0]
        self.state_dim = self.state.shape[1]
        self.action_dim = self.action.shape[1]
        self.device = device
        self.reward_function = reward_function
        self.desired_speed = desired_speed
        
        # 决定使用数据集中的reward还是重新计算
        if reward_function is not None:
            print(f">> 使用自定义奖励函数重新计算reward (desired_speed={desired_speed} m/s)...")
            # 使用自定义函数重新计算reward
            self.reward = self._compute_rewards_with_function()
            print(f">> 使用自定义奖励函数计算完成")
        else:
            # 使用数据集中的reward
            self.reward = rewards.reshape(-1, 1)  # 重塑为 (N, 1)
            print(f">> 使用数据集中的原始reward")
        
        # 奖励调整
        if reward_tune == 'normalize':
            reward_mean = self.reward.mean()
            reward_std = self.reward.std()
            self.reward = (self.reward - reward_mean) / (reward_std + 1e-8)
            print(f">> 奖励已归一化: 均值={reward_mean:.4f}, 标准差={reward_std:.4f}")
        elif reward_tune == 'iql_antmaze':
            self.reward = self.reward - 1.0
            print(f">> 奖励已调整 (iql_antmaze): 减去1.0")
        else:
            print(f">> 奖励未调整 (reward_tune='{reward_tune}')")
        
        print(f">> 数据集大小: {self.size}")
        print(f">> 观测维度: {self.state_dim}")
        print(f">> 动作维度: {self.action_dim}")
        print(f">> 奖励范围: [{self.reward.min():.2f}, {self.reward.max():.2f}]")
        print(f">> 奖励均值: {self.reward.mean():.4f}, 标准差: {self.reward.std():.4f}")
    
    def _compute_rewards_with_function(self, batch_size=10000):
        """
        使用自定义奖励函数批量计算reward
        
        参数:
            batch_size: 批量处理大小（避免内存溢出）
        
        返回:
            rewards: torch.Tensor, shape (N, 1)
        """
        all_rewards = []
        num_samples = self.size
        
        # 批量处理
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_state = self.state[i:end_idx]
            batch_action = self.action[i:end_idx]
            batch_next_state = self.next_state[i:end_idx]
            batch_done = self.done[i:end_idx]
            batch_info = self.info[i:end_idx]
            
            # 调用自定义奖励函数
            batch_reward = self.reward_function(
                batch_state, batch_action, batch_next_state, batch_done, batch_info, self.desired_speed
            )
            
            # 确保reward是torch.Tensor
            if not isinstance(batch_reward, torch.Tensor):
                batch_reward = torch.tensor(batch_reward, dtype=torch.float32)
            
            # 确保shape正确
            if batch_reward.dim() == 1:
                batch_reward = batch_reward.unsqueeze(1)  # (N,) -> (N, 1)
            
            all_rewards.append(batch_reward)
        
        # 合并所有batch的reward
        rewards = torch.cat(all_rewards, dim=0)
        return rewards
    
    def sample(self, batch_size):
        """随机采样一个批次，返回 (state, action, reward, next_state, done)"""
        indices = torch.randint(0, self.size, (batch_size,))
        
        return (
            self.state[indices].to(self.device),
            self.action[indices].to(self.device),
            self.reward[indices].to(self.device),
            self.next_state[indices].to(self.device),
            self.done[indices].to(self.device)
        )

