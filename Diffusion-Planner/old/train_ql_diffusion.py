"""
Diffusion Q-learning 训练脚本
使用 Diffusion_Planner 作为策略网络，在离线数据集上训练
"""

import os
import sys
import glob
import argparse
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple, Optional, Callable
from tqdm import tqdm

# 设置无缓冲输出，确保print信息实时显示
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.loss import diffusion_loss_func
from diffusion_planner.utils.normalizer import StateNormalizer, ObservationNormalizer
from diffusion_planner.utils.train_utils import opendata
from torch.utils.tensorboard import SummaryWriter


class NPZDataset(Dataset):
    """从npz文件加载数据的Dataset"""
    
    def __init__(self, data_dir: str, device: str = "cuda"):
        self.data_dir = data_dir
        self.device = device
        self.npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        
        if len(self.npz_files) == 0:
            raise ValueError(f"No npz files found in {data_dir}")
        
        print(f"Found {len(self.npz_files)} npz files", flush=True)
    
    def __len__(self):
        return len(self.npz_files)
    
    def __getitem__(self, idx):
        """返回 (state, action, reward, next_state, not_done)"""
        data = opendata(self.npz_files[idx])
        
        # 当前状态（不带_next）
        state = {
            'ego_current_state': torch.from_numpy(data['ego_current_state']).float(),
            'neighbor_agents_past': torch.from_numpy(data['neighbor_agents_past']).float(),
            'lanes': torch.from_numpy(data['lanes']).float(),
            'lanes_speed_limit': torch.from_numpy(data['lanes_speed_limit']).float(),
            'lanes_has_speed_limit': torch.from_numpy(data['lanes_has_speed_limit']).bool(),
            'route_lanes': torch.from_numpy(data['route_lanes']).float(),
            'route_lanes_speed_limit': torch.from_numpy(data['route_lanes_speed_limit']).float(),
            'route_lanes_has_speed_limit': torch.from_numpy(data['route_lanes_has_speed_limit']).bool(),
            'static_objects': torch.from_numpy(data['static_objects']).float(),
        }
        
        # 下一状态（带_next）
        next_state = {
            'ego_current_state': torch.from_numpy(data['ego_current_state_next']).float(),
            'neighbor_agents_past': torch.from_numpy(data['neighbor_agents_past_next']).float(),
            'lanes': torch.from_numpy(data['lanes_next']).float(),
            'lanes_speed_limit': torch.from_numpy(data['lanes_speed_limit_next']).float(),
            'lanes_has_speed_limit': torch.from_numpy(data['lanes_has_speed_limit_next']).bool(),
            'route_lanes': torch.from_numpy(data['route_lanes_next']).float(),
            'route_lanes_speed_limit': torch.from_numpy(data['route_lanes_speed_limit_next']).float(),
            'route_lanes_has_speed_limit': torch.from_numpy(data['route_lanes_has_speed_limit_next']).bool(),
            'static_objects': torch.from_numpy(data['static_objects_next']).float(),
        }
        
        # 动作：自车未来轨迹 [T, 3] -> [T, 4] (x, y, cos, sin)
        ego_future = data['ego_agent_future']  # [T, 3] (x, y, heading)
        heading = ego_future[:, 2]
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        action = np.concatenate([ego_future[:, :2], cos_h[:, None], sin_h[:, None]], axis=1)  # [T, 4]
        action = torch.from_numpy(action).float()
        
        # 计算奖励：使用与Q函数相同的输入格式
        # 提取state_features
        state_features = extract_state_features({k: v.unsqueeze(0) for k, v in state.items()}, use_all_features=False)  # [1, state_feature_dim]
        state_features = state_features.squeeze(0)  # [state_feature_dim]
        
        # Flatten action
        action_flat = action.reshape(-1)  # [T*4]
        
        # 计算reward
        reward = compute_reward(state_features, action_flat)  # scalar
        
        # not_done（暂时设为True）
        not_done = torch.tensor(1.0)
        
        return state, action, reward, next_state, not_done


def collate_fn(batch):
    """自定义collate函数，处理字典类型的state"""
    states, actions, rewards, next_states, not_dones = zip(*batch)
    
    # 堆叠字典
    batch_state = {}
    batch_next_state = {}
    
    for key in states[0].keys():
        batch_state[key] = torch.stack([s[key] for s in states], dim=0)
        batch_next_state[key] = torch.stack([s[key] for s in next_states], dim=0)
    
    batch_action = torch.stack(actions, dim=0)  # [B, T, 4]
    batch_reward = torch.stack(rewards, dim=0)  # [B]
    batch_not_done = torch.stack(not_dones, dim=0)  # [B]
    
    return batch_state, batch_action, batch_reward, batch_next_state, batch_not_done


class Critic(nn.Module):
    """
    Critic网络：Q(s, a)
    
    改进版本：
    1. 支持状态-动作分离编码（可选）
    2. 可配置的网络深度和宽度
    3. 可选的残差连接
    """
    
    def __init__(
        self, 
        state_feature_dim: int, 
        action_dim: int, 
        hidden_dim: int = 512,
        num_layers: int = 3,
        use_separate_encoding: bool = False, # 是否使用状态-动作分离encoder，如果为True，则使用状态和动作的encoder分别编码，然后融合
    ):
        super(Critic, self).__init__()
        self.use_separate_encoding = use_separate_encoding
        
        if use_separate_encoding:
            # 方案：先分别编码状态和动作，再融合
            self.state_encoder = nn.Sequential(
                nn.Linear(state_feature_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Mish(),
            )
            self.action_encoder = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Mish(),
            )
            fusion_dim = hidden_dim * 2
        else:
            # 原始方案：直接concatenate
            fusion_dim = state_feature_dim + action_dim
        
        # Q1网络
        q1_layers = []
        input_dim = fusion_dim
        for i in range(num_layers):
            q1_layers.append(nn.Linear(input_dim, hidden_dim))
            q1_layers.append(nn.Mish())
            input_dim = hidden_dim
        q1_layers.append(nn.Linear(hidden_dim, 1))
        self.q1_model = nn.Sequential(*q1_layers)
        
        # Q2网络（结构相同）
        q2_layers = []
        input_dim = fusion_dim
        for i in range(num_layers):
            q2_layers.append(nn.Linear(input_dim, hidden_dim))
            q2_layers.append(nn.Mish())
            input_dim = hidden_dim
        q2_layers.append(nn.Linear(hidden_dim, 1))
        self.q2_model = nn.Sequential(*q2_layers)
    
    def forward(self, state_features: torch.Tensor, action: torch.Tensor):
        """
        state_features: [B, state_feature_dim]
        action: [B, action_dim]
        """
        if self.use_separate_encoding:
            state_encoded = self.state_encoder(state_features)  # [B, hidden_dim]
            action_encoded = self.action_encoder(action)  # [B, hidden_dim]
            x = torch.cat([state_encoded, action_encoded], dim=-1)  # [B, hidden_dim * 2]
        else:
            x = torch.cat([state_features, action], dim=-1)  # [B, state_feature_dim + action_dim]
        
        return self.q1_model(x), self.q2_model(x)
    
    def q_min(self, state_features: torch.Tensor, action: torch.Tensor):
        q1, q2 = self.forward(state_features, action)
        return torch.min(q1, q2)


# 全局变量：存储reward的均值和标准差（用于Z-score归一化）
REWARD_MEAN = None
REWARD_STD = None
REWARD_NORMALIZE = False  # 是否启用reward归一化


def compute_reward(state_features: torch.Tensor, action_flat: torch.Tensor, normalize: bool = None) -> torch.Tensor:
    """
    计算奖励函数
    
    Args:
        state_features: [B, state_feature_dim] 或 [state_feature_dim] - 状态特征
        action_flat: [B, action_dim] 或 [action_dim] - 动作（flatten后的未来轨迹）
        normalize: 是否应用Z-score归一化，如果为None则使用全局REWARD_NORMALIZE设置
    
    Returns:
        reward: [B] 或 scalar - 奖励值
    """
    global REWARD_MEAN, REWARD_STD, REWARD_NORMALIZE
    
    if normalize is None:
        normalize = REWARD_NORMALIZE
    # 确保输入是2D tensor
    if state_features.dim() == 1:
        state_features = state_features.unsqueeze(0)  # [1, state_feature_dim]
    if action_flat.dim() == 1:
        action_flat = action_flat.unsqueeze(0)  # [1, action_dim]
    
    B = state_features.shape[0]
    device = state_features.device
    
    # 从action_flat中恢复轨迹形状 [B, T, 4]
    # action_dim = T * 4，假设T=80
    T = action_flat.shape[1] // 4
    action = action_flat.reshape(B, T, 4)  # [B, T, 4] (x, y, cos_h, sin_h)
    
    # # 1. 速度奖励：鼓励合理的速度（从ego_current_state中提取速度信息）
    # # state_features的前10维是ego_current_state: [x, y, cos(h), sin(h), vx, vy, ax, ay, steering, yaw_rate]
    # if state_features.shape[1] >= 10:
    #     ego_vx = state_features[:, 4]  # [B] vx
    #     ego_vy = state_features[:, 5]  # [B] vy
    #     ego_speed = torch.sqrt(ego_vx ** 2 + ego_vy ** 2)  # [B] 速度大小
        
    #     # 目标速度：假设理想速度为10-15 m/s (36-54 km/h)
    #     target_speed = 12.5  # m/s
    #     speed_reward = -torch.abs(ego_speed - target_speed) / target_speed  # 归一化速度奖励
    # else:
    #     speed_reward = torch.zeros(B, device=device)
    
    # 2. 轨迹平滑度奖励：惩罚轨迹的加速度变化
    # 计算轨迹的加速度（二阶差分）
    if T >= 3:
        # 位置差分得到速度
        positions = action[:, :, :2]  # [B, T, 2] (x, y)
        velocities = positions[:, 1:, :] - positions[:, :-1, :]  # [B, T-1, 2]
        
        # 速度差分得到加速度
        if velocities.shape[1] >= 2:
            accelerations = velocities[:, 1:, :] - velocities[:, :-1, :]  # [B, T-2, 2]
            accel_magnitude = torch.norm(accelerations, dim=-1)  # [B, T-2]
            smoothness_reward = -accel_magnitude.mean(dim=1)  # [B] 平均加速度越小越好
        else:
            smoothness_reward = torch.zeros(B, device=device)
    else:
        smoothness_reward = torch.zeros(B, device=device)
    
    # # 3. 进度奖励：鼓励向前移动（在ego坐标系中，y方向是前进方向）
    # if T > 0:
    #     # 计算轨迹的最终位置相对于起始位置的y方向位移
    #     start_y = action[:, 0, 1]  # [B] 起始y位置
    #     end_y = action[:, -1, 1]  # [B] 结束y位置
    #     progress = end_y - start_y  # [B] y方向位移（前进为正）
    #     progress_reward = progress / 10.0  # 归一化（假设10m为参考距离）
    # else:
    #     progress_reward = torch.zeros(B, device=device)
    
    # # 4. 轨迹稳定性奖励：惩罚轨迹的横向偏移
    # if T > 0:
    #     # 计算轨迹的横向偏移（x方向的变化）
    #     x_positions = action[:, :, 0]  # [B, T]
    #     x_std = torch.std(x_positions, dim=1)  # [B] x方向的标准差
    #     stability_reward = -x_std / 2.0  # 归一化（假设2m为参考）
    # else:
    #     stability_reward = torch.zeros(B, device=device)
    
    # # 5. 碰撞风险惩罚（简化版）：基于邻居车辆的距离
    # # 从state_features中提取邻居车辆信息
    # # 假设state_features的结构：[ego_current_state(10), neighbor_agents_past(32*11=352), ...]
    # collision_penalty = torch.zeros(B, device=device)
    # if state_features.shape[1] > 10:
    #     # 提取邻居车辆的最后位置（假设neighbor_agents_past的最后一个时间步）
    #     neighbor_features = state_features[:, 10:10+32*11]  # [B, 352]
    #     neighbor_features = neighbor_features.reshape(B, 32, 11)  # [B, 32, 11]
        
    #     # 提取邻居车辆的位置（前4维是x, y, cos_h, sin_h）
    #     neighbor_positions = neighbor_features[:, :, :2]  # [B, 32, 2]
    #     ego_position = state_features[:, :2]  # [B, 2] 自车位置（在ego坐标系中为0,0）
        
    #     # 计算距离
    #     distances = torch.norm(neighbor_positions - ego_position.unsqueeze(1), dim=-1)  # [B, 32]
        
    #     # 找到最近距离
    #     min_distances, _ = torch.min(distances, dim=1)  # [B]
        
    #     # 如果距离太近，给予惩罚（假设安全距离为3m）
    #     safe_distance = 3.0
    #     collision_penalty = -torch.exp(-min_distances / safe_distance)  # 距离越近，惩罚越大
    
    # # 组合所有奖励（加权求和）
    # reward = (
    #     0.3 * speed_reward +
    #     0.2 * smoothness_reward +
    #     0.3 * progress_reward +
    #     0.1 * stability_reward +
    #     0.1 * collision_penalty
    # )
    # 当前只使用平滑度奖励
    reward = smoothness_reward
    
    # 应用Z-score归一化（如果启用）
    if normalize and REWARD_MEAN is not None and REWARD_STD is not None and REWARD_STD > 1e-8:
        reward = (reward - REWARD_MEAN) / REWARD_STD
    
    # 如果输入是单个样本（1D输入），返回标量
    if state_features.dim() == 1 or (B == 1 and reward.dim() == 1):
        return reward.squeeze(0) if reward.dim() > 0 else reward
    
    return reward


def extract_state_features(inputs: Dict[str, torch.Tensor], use_all_features: bool = False) -> torch.Tensor:
    """
    从diffusion_planner的输入中提取状态特征
    
    Args:
        inputs: 状态字典
        use_all_features: 是否使用所有可用特征（包括lanes, route_lanes, static_objects等）
                        如果True，会使用更多信息但维度会大幅增加
    """
    features = []
    
    # 提取自车当前状态 [B, 10]
    if 'ego_current_state' in inputs:
        features.append(inputs['ego_current_state'])
    
    # 提取邻居车辆历史状态
    if 'neighbor_agents_past' in inputs:
        neighbor_past = inputs['neighbor_agents_past']  # [B, Pn, T_past, 11]
        B = neighbor_past.shape[0]
        
        if use_all_features:
            # 方案1: 使用所有时序信息（维度很大）
            # neighbor_flat = neighbor_past.reshape(B, -1)  # [B, Pn * T_past * 11]
            
            # 方案2: 聚合时序信息（推荐）
            # 取最后几个时间步的均值
            neighbor_last_n = neighbor_past[:, :, -3:, :]  # [B, Pn, 3, 11]
            neighbor_mean = neighbor_last_n.mean(dim=2)  # [B, Pn, 11]
            neighbor_max = neighbor_last_n.max(dim=2)[0]  # [B, Pn, 11]
            neighbor_last = neighbor_past[:, :, -1, :]  # [B, Pn, 11]
            # 拼接均值、最大值和最后值
            neighbor_combined = torch.cat([neighbor_mean, neighbor_max, neighbor_last], dim=-1)  # [B, Pn, 33]
            neighbor_flat = neighbor_combined.reshape(B, -1)  # [B, Pn * 33]
        else:
            # 只取最后一个时间步（原始方案）
            neighbor_last = neighbor_past[:, :, -1, :]  # [B, Pn, 11]
            neighbor_flat = neighbor_last.reshape(B, -1)  # [B, Pn * 11]
        
        features.append(neighbor_flat)
    
    # 如果启用，添加其他特征
    if use_all_features:
        # 车道信息 [B, lane_num, lane_len, feature_dim]
        if 'lanes' in inputs:
            lanes = inputs['lanes']  # [B, 70, 20, ?]
            B = lanes.shape[0]
            # 聚合：取每个车道的中心点或均值
            lanes_flat = lanes.mean(dim=2)  # [B, 70, feature_dim]
            lanes_flat = lanes_flat.reshape(B, -1)  # [B, 70 * feature_dim]
            features.append(lanes_flat)
        
        # 路径车道信息 [B, route_num, route_len, feature_dim]
        if 'route_lanes' in inputs:
            route_lanes = inputs['route_lanes']  # [B, 25, 20, ?]
            B = route_lanes.shape[0]
            route_flat = route_lanes.mean(dim=2)  # [B, 25, feature_dim]
            route_flat = route_flat.reshape(B, -1)  # [B, 25 * feature_dim]
            features.append(route_flat)
        
        # 静态物体 [B, 5, 10]
        if 'static_objects' in inputs:
            static_objects = inputs['static_objects']  # [B, 5, 10]
            B = static_objects.shape[0]
            static_flat = static_objects.reshape(B, -1)  # [B, 50]
            features.append(static_flat)
    
    if len(features) > 0:
        return torch.cat(features, dim=-1)
    else:
        raise ValueError("No valid state features found in inputs")


class QL_Diffusion:
    """Diffusion Q-learning算法，使用Diffusion_Planner作为策略网络"""
    
    def __init__(
        self,
        policy_config: Any,
        state_feature_dim: int,
        action_dim: int,
        device: str = "cuda",
        discount: float = 0.99,
        tau: float = 0.005,
        lr_policy: float = 3e-4,
        lr_critic: float = 3e-4,
        grad_norm: float = 1.0,
        state_normalizer: StateNormalizer = None,
        observation_normalizer: ObservationNormalizer = None,
        resume_checkpoint: str = None,
        eta: float = 1.0,  # Q值损失的权重，对应ql_diffusion.py中的eta
        r_fun: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,  # 可选的reward函数，如果提供则直接使用reward而不是Q网络
        max_q_backup: bool = False,  # 是否使用Max Q Backup来减少Q值过估计
        max_q_backup_samples: int = 10,  # Max Q Backup时采样的动作数量
    ):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.grad_norm = grad_norm
        self.eta = eta  # Q值损失权重
        self.r_fun = r_fun  # Reward函数，如果提供则直接使用reward而不是Q网络
        self.max_q_backup = max_q_backup  # Max Q Backup标志
        self.max_q_backup_samples = max_q_backup_samples  # Max Q Backup采样数量
        
        # 初始化策略网络（Diffusion_Planner）
        self.policy = Diffusion_Planner(policy_config).to(device)
        
        # 加载base_weight下的预训练模型参数
        # base_weight目录与train_ql_diffusion.py在同一级
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_weight_path = os.path.join(script_dir, 'base_weight', 'model_epoch_500_trainloss_0.0486.pth')
        
        if os.path.exists(base_weight_path):
            print(f"\n{'='*60}", flush=True)
            print(f"Loading base weight from: {base_weight_path}", flush=True)
            print(f"{'='*60}", flush=True)
            self._load_policy_checkpoint(base_weight_path)
        else:
            print(f"\n{'='*60}", flush=True)
            print(f"Warning: Base weight file not found at {base_weight_path}", flush=True)
            print("Policy will be initialized randomly.", flush=True)
            print(f"{'='*60}\n", flush=True)
        
        # 如果还指定了resume_checkpoint，可以用于加载其他组件（如optimizer状态）
        # 但policy权重已经从上方的base_weight加载了
        if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
            print(f"Note: resume_checkpoint specified ({resume_checkpoint}) but policy weights already loaded from base_weight.", flush=True)
        
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)
        
        # 初始化target策略网络
        self.policy_target = copy.deepcopy(self.policy)
        
        # 初始化Critic网络
        self.critic = Critic(state_feature_dim, action_dim).to(device)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=lr_critic)
        
        # 初始化target Critic网络
        self.critic_target = copy.deepcopy(self.critic)
        
        # 归一化器
        self.state_normalizer = state_normalizer
        self.observation_normalizer = observation_normalizer
        
        # 保存policy_config以便后续使用
        self.policy_config = policy_config
        
        self.step = 0
    
    def _load_policy_checkpoint(self, checkpoint_path: str):
        """加载策略网络的checkpoint"""
        print(f"\n{'='*60}", flush=True)
        print(f"Loading policy checkpoint from: {checkpoint_path}", flush=True)
        print(f"{'='*60}", flush=True)
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 尝试多种格式
            state_dict = None
            
            # 格式1: 新格式（我们保存的QL训练checkpoint）
            if 'policy_state_dict' in checkpoint:
                state_dict = checkpoint['policy_state_dict']
                print("Found 'policy_state_dict' in checkpoint", flush=True)
            
            # 格式2: 原始diffusion_planner格式（包含'model'键）
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
                print("Found 'model' in checkpoint", flush=True)
            
            # 格式3: 包含'ema_state_dict'（如果启用EMA）
            elif 'ema_state_dict' in checkpoint:
                state_dict = checkpoint['ema_state_dict']
                print("Found 'ema_state_dict' in checkpoint", flush=True)
            
            # 格式4: 直接是state_dict
            else:
                # 检查是否是直接的state_dict（所有键都是模型参数名）
                if isinstance(checkpoint, dict) and any('encoder' in k or 'decoder' in k for k in checkpoint.keys()):
                    state_dict = checkpoint
                    print("Checkpoint appears to be a direct state_dict", flush=True)
            
            if state_dict is None:
                raise ValueError("Could not find valid state_dict in checkpoint")
            
            # 处理DDP格式（移除'module.'前缀）
            if any(k.startswith('module.') for k in state_dict.keys()):
                print("Removing 'module.' prefix from state_dict keys (DDP format)", flush=True)
                state_dict = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
            
            # 加载state_dict
            missing_keys, unexpected_keys = self.policy.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys in checkpoint: {len(missing_keys)} keys", flush=True)
                if len(missing_keys) <= 10:
                    for key in missing_keys:
                        print(f"  - {key}", flush=True)
                else:
                    for key in missing_keys[:10]:
                        print(f"  - {key}", flush=True)
                    print(f"  ... and {len(missing_keys) - 10} more", flush=True)
            
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys", flush=True)
                if len(unexpected_keys) <= 10:
                    for key in unexpected_keys:
                        print(f"  - {key}", flush=True)
                else:
                    for key in unexpected_keys[:10]:
                        print(f"  - {key}", flush=True)
                    print(f"  ... and {len(unexpected_keys) - 10} more", flush=True)
            
            print(f"✓ Policy checkpoint loaded successfully!", flush=True)
            print(f"{'='*60}\n", flush=True)
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}", flush=True)
            print("Continuing with randomly initialized policy...", flush=True)
            print(f"{'='*60}\n", flush=True)
    
    def train(self, data_loader: DataLoader, iterations: int = None, epoch: int = None):
        """训练一步"""
        diffusion_losses = []
        bc_losses = []  # 行为克隆损失（ego_planning_loss）
        critic_losses = []
        q_losses = []  # Q值损失
        q1_new_action_values = []  # 记录新动作的Q1值
        new_reward_values = []  # 记录新动作的reward值
        
        if iterations is None:
            iterations = len(data_loader)
        
        # 创建进度条
        desc = f"Training"
        if epoch is not None:
            desc = f"Epoch {epoch}"
        
        # 使用stderr输出进度条，这样即使stdout被重定向，进度条也能显示
        # tqdm会自动检测终端，如果检测不到会禁用进度条
        # 通过将进度条输出到stderr，可以避免被tee重定向
        pbar = tqdm(
            enumerate(data_loader), 
            total=iterations, 
            desc=desc, 
            unit="batch",
            file=sys.stderr,  # 输出到stderr，避免被tee重定向
        )
        
        for i, (state, action, reward, next_state, not_done) in pbar:
            if i >= iterations:
                break
            
            # 移动到设备
            state = {k: v.to(self.device) for k, v in state.items()}
            next_state = {k: v.to(self.device) for k, v in next_state.items()}
            action = action.to(self.device)  # [B, T, 4]
            reward = reward.to(self.device)  # [B]
            not_done = not_done.to(self.device)  # [B]
            
            B = action.shape[0]
            action_flat = action.reshape(B, -1)  # [B, T*4]
            
            # ========== Q函数学习 ==========
            # 提取状态特征
            # 注意：这里需要与初始化时使用相同的use_all_features设置
            state_features = extract_state_features(state, use_all_features=False)  # [B, state_feature_dim]
            next_state_features = extract_state_features(next_state, use_all_features=False)  # [B, state_feature_dim]
            
            # 当前Q值
            current_q1, current_q2 = self.critic(state_features, action_flat)
            
            # 从target策略采样下一动作
            with torch.no_grad():
                # 归一化next_state
                next_state_norm = next_state
                if self.observation_normalizer is not None:
                    next_state_norm = self.observation_normalizer(next_state)
                
                # 使用target策略采样动作（设置为eval模式进行推理）
                self.policy_target.eval()
                
                if self.max_q_backup:
                    # Max Q Backup: 对每个next_state采样多个动作，取max Q值（与Diffusion-Policies-for-Offline-RL-master一致）
                    # 将next_state重复max_q_backup_samples次
                    next_state_rpt = {}
                    for k, v in next_state_norm.items():
                        # v: [B, ...]
                        next_state_rpt[k] = v.repeat_interleave(self.max_q_backup_samples, dim=0)  # [B*N, ...]
                    
                    # 对重复的next_state采样动作
                    _, decoder_output_rpt = self.policy_target(next_state_rpt)
                    next_action_rpt = decoder_output_rpt['prediction'][:, 0, :, :]  # [B*N, T, 4]
                    next_action_flat_rpt = next_action_rpt.reshape(B * self.max_q_backup_samples, -1)  # [B*N, T*4]
                    
                    # 重复next_state_features
                    next_state_features_rpt = next_state_features.repeat_interleave(self.max_q_backup_samples, dim=0)  # [B*N, state_feature_dim]
                    
                    # 计算所有采样动作的Q值
                    target_q1_rpt, target_q2_rpt = self.critic_target(next_state_features_rpt, next_action_flat_rpt)  # [B*N, 1]
                    
                    # 确保Q值是[B*N, 1]形状
                    if target_q1_rpt.dim() > 2:
                        target_q1_rpt = target_q1_rpt.squeeze(-1)  # 移除多余的维度
                    if target_q2_rpt.dim() > 2:
                        target_q2_rpt = target_q2_rpt.squeeze(-1)
                    
                    # 将Q值reshape为[B, N]，然后取max（与参考实现一致）
                    target_q1_rpt = target_q1_rpt.view(B, self.max_q_backup_samples)  # [B, N]
                    target_q2_rpt = target_q2_rpt.view(B, self.max_q_backup_samples)  # [B, N]
                    
                    # 对每个样本的多个动作取max Q值（与Diffusion-Policies-for-Offline-RL-master一致）
                    target_q1 = target_q1_rpt.max(dim=1, keepdim=True)[0]  # [B, 1]
                    target_q2 = target_q2_rpt.max(dim=1, keepdim=True)[0]  # [B, 1]
                    
                    # 对Q1和Q2取min（Double Q-learning，减少过估计）
                    target_q = torch.min(target_q1, target_q2)  # [B, 1]
                else:
                    # 标准方法：只采样一个动作
                    _, decoder_output = self.policy_target(next_state_norm)
                    next_action = decoder_output['prediction'][:, 0, :, :]  # [B, T, 4] 只取自车
                    next_action_flat = next_action.reshape(B, -1)  # [B, T*4]
                    
                    # Target Q值
                    target_q1, target_q2 = self.critic_target(next_state_features, next_action_flat)
                    target_q = torch.min(target_q1, target_q2)  # [B, 1]
                
                self.policy_target.train()
            
            # Bellman目标
            # 确保所有tensor的形状正确：reward和not_done是[B]，target_q是[B, 1]
            # 使用unsqueeze确保broadcasting正确
            reward_expanded = reward.unsqueeze(-1)  # [B, 1]
            not_done_expanded = not_done.unsqueeze(-1)  # [B, 1]
            target_q = (reward_expanded + not_done_expanded * self.discount * target_q).detach()  # [B, 1]
            
            # Critic损失
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            # 更新Critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic_optimizer.step()
            
            # ========== 策略学习 ==========
            # 归一化state
            state_norm = state
            if self.observation_normalizer is not None:
                state_norm = self.observation_normalizer(state)
            
            # 准备futures数据（用于计算diffusion loss）
            ego_future = action  # [B, T, 4]
            
            # 创建空的neighbors_future（因为neighbor_agents_future已从state中移除）
            Pn = self.policy.decoder.decoder._predicted_neighbor_num
            neighbors_future = torch.zeros(B, Pn, action.shape[1], 4).to(self.device)  # [B, Pn, T, 4]
            neighbor_future_mask = torch.zeros(B, Pn, action.shape[1], dtype=torch.bool).to(self.device)  # [B, Pn, T]
            
            futures = (ego_future, neighbors_future, neighbor_future_mask)
            
            # 计算扩散损失（行为克隆损失，对应ql_diffusion.py中的bc_loss）
            loss = {}
            marginal_prob = self.policy.sde.marginal_prob
            
            loss, _ = diffusion_loss_func(
                model=self.policy,
                inputs=state_norm,
                marginal_prob=marginal_prob,
                futures=futures,
                norm=self.state_normalizer,
                loss=loss,
                model_type=self.policy_config.diffusion_model_type,
            )
            
            # 从策略中采样新动作（对应ql_diffusion.py中的new_action = self.actor(state)）
            # 使用与BC loss相同的方式：在train模式下通过提供sampled_trajectories和diffusion_time来获取prediction
            # 这样可以保持梯度流，让Q loss能够反传到policy（与BC loss使用相同的梯度传播路径）
            
            # 保持policy在train模式（与BC loss相同）
            # 生成用于采样的trajectories（类似BC loss中的方式，但不需要真实的future）
            marginal_prob_q = self.policy.sde.marginal_prob
            eps = 1e-3
            
            # 准备ego和neighbor的current states
            ego_current_q = state_norm['ego_current_state'][:, :4]  # [B, 4]
            neighbors_current_q = state_norm["neighbor_agents_past"][:, :self.policy.decoder.decoder._predicted_neighbor_num, -1, :4]  # [B, Pn, 4]
            Pn = self.policy.decoder.decoder._predicted_neighbor_num
            P = 1 + Pn  # ego + neighbors
            T = action.shape[1]  # 时间步数
            
            # 创建dummy future trajectories（用于生成noise，实际值不重要）
            # 注意：需要与BC loss中的格式一致：[B, P, T, 4]，其中P=1+neighbor_num
            ego_future_dummy = torch.zeros(B, T, 4, device=self.device)  # [B, T, 4]
            neighbors_future_dummy = torch.zeros(B, Pn, T, 4, device=self.device)  # [B, Pn, T, 4]
            
            # 先合并ego和neighbor的future（与BC loss中的格式一致）
            gt_future_q = torch.cat([ego_future_dummy[:, None, :, :], neighbors_future_dummy], dim=1)  # [B, P, T, 4]
            
            # 归一化dummy future（用于计算mean和std，与BC loss中的方式一致）
            if self.state_normalizer is not None:
                gt_future_q = self.state_normalizer(gt_future_q)  # [B, P, T, 4]
            
            # 计算neighbor_current_mask（与BC loss中的方式一致）
            neighbor_current_mask_q = (torch.sum(torch.ne(neighbors_current_q[..., :4], 0), dim=-1) == 0).to(torch.bool)  # [B, Pn]
            # 创建neighbor_future_mask（全为False，因为dummy future都是0）
            neighbor_future_mask_q = torch.zeros(B, Pn, T, dtype=torch.bool, device=self.device)  # [B, Pn, T]
            # 合并neighbor mask（与BC loss中的格式一致：[B, Pn, 1+T]）
            neighbor_mask_q = torch.cat([neighbor_current_mask_q.unsqueeze(-1), neighbor_future_mask_q], dim=-1)  # [B, Pn, 1+T]
            # 添加ego的mask（ego总是有效的，全为False）
            neighbor_mask_q_full = torch.cat([torch.zeros(B, 1, 1+T, dtype=torch.bool, device=self.device), 
                                             neighbor_mask_q], dim=1)  # [B, P, 1+T]
            
            # 生成diffusion time和noise（类似BC loss）
            t_q = torch.rand(B, device=self.device) * (1 - eps) + eps  # [B,]
            z_q = torch.randn(B, P, T, 4, device=self.device)  # [B, P, T, 4]
            
            # 计算mean和std（与BC loss中的方式一致）
            current_states_q = torch.cat([ego_current_q[:, None, :], neighbors_current_q], dim=1)  # [B, P, 4]
            all_gt_q = torch.cat([current_states_q[:, :, None, :], gt_future_q], dim=2)  # [B, P, 1+T, 4]
            # mask掉无效的neighbor（与BC loss中的方式一致）
            # 注意：all_gt_q[:, 1:]是[B, Pn, 1+T, 4]（去掉了ego），所以使用neighbor_mask_q（不包含ego）
            all_gt_q[:, 1:][neighbor_mask_q] = 0.0
            
            mean_q, std_q = marginal_prob_q(all_gt_q[:, :, 1:, :], t_q)  # [B, P, T, 4]
            # std_q的形状是[B]，需要扩展为[B, P, T, 4]的形状（与diffusion_loss_func中的方式一致）
            std_q = std_q.view(-1, *([1] * (len(all_gt_q[:, :, 1:, :].shape)-1)))  # [B, 1, 1, 1]
            
            # 生成xT（加噪后的trajectories）
            xT_q = mean_q + std_q * z_q  # [B, P, T, 4]
            xT_q = torch.cat([current_states_q[:, :, None, :], xT_q], dim=2)  # [B, P, 1+T, 4]
            
            # 准备inputs（包含sampled_trajectories和diffusion_time，让模型在train模式下运行）
            state_norm_with_sampling = {**state_norm}
            state_norm_with_sampling["sampled_trajectories"] = xT_q.reshape(B, P, -1)  # [B, P, (1+T)*4]
            state_norm_with_sampling["diffusion_time"] = t_q
            
            # 在train模式下调用policy，获取score（有梯度，与BC loss相同的方式）
            _, decoder_output_new = self.policy(state_norm_with_sampling)  # 返回score
            
            # 从score解码得到prediction（x0）
            # 对于x_start类型：score就是x0的预测
            # 对于score类型：需要从score和xT反推x0
            if self.policy_config.diffusion_model_type == "x_start":
                # score就是x0的预测
                x0_pred = decoder_output_new['score'][:, 0, 1:, :]  # [B, T, 4] 只取自车
            else:  # score类型
                # 从score和xT反推x0: x0 = (xT - sigma_t * score) / alpha_t
                xT_ego = xT_q[:, 0, 1:, :]  # [B, T, 4]
                score_ego = decoder_output_new['score'][:, 0, 1:, :]  # [B, T, 4]
                # 计算alpha_t和sigma_t（使用VPSDE_linear的公式）
                t_q_expanded = t_q.view(B, 1, 1)  # [B, 1, 1]
                mean_log_coeff = -0.25 * t_q_expanded ** 2 * (20.0 - 0.1) - 0.5 * 0.1 * t_q_expanded
                alpha_t = torch.exp(mean_log_coeff)  # [B, 1, 1]
                sigma_t = torch.sqrt(1 - alpha_t ** 2)  # [B, 1, 1]
                x0_pred = (xT_ego - sigma_t * score_ego) / (alpha_t + 1e-8)  # [B, T, 4]
            
            # 反归一化
            # StateNormalizer.inverse期望输入形状为[B, P, 1+T, 4]，其中包含current state
            # 但x0_pred只有future，形状为[B, T, 4]，需要先添加current state
            if self.state_normalizer is not None:
                # 获取ego的current state（已归一化）
                ego_current_q_norm = state_norm['ego_current_state'][:, :4]  # [B, 4]
                # 将current state和future拼接：[B, 1+T, 4]
                x0_pred_with_current = torch.cat([ego_current_q_norm[:, None, :], x0_pred], dim=1)  # [B, 1+T, 4]
                # 扩展为[B, 1, 1+T, 4]（只有ego，P=1）
                x0_pred_expanded = x0_pred_with_current.unsqueeze(1)  # [B, 1, 1+T, 4]
                # 调用inverse进行反归一化
                new_action_expanded = self.state_normalizer.inverse(x0_pred_expanded)  # [B, 1, 1+T, 4]
                # 去掉current state，只保留future：[B, T, 4]
                new_action = new_action_expanded[:, 0, 1:, :]  # [B, T, 4]
            else:
                new_action = x0_pred  # [B, T, 4]
            
            new_action_flat = new_action.reshape(B, -1)  # [B, T*4]
            
            # 验证new_action是否有梯度（应该为True，因为使用了train模式）
            if not hasattr(self, '_grad_check_printed'):
                print(f"\n{'='*60}", flush=True)
                print("梯度传播检查", flush=True)
                print(f"{'='*60}", flush=True)
                print(f"x0_pred.shape: {x0_pred.shape}", flush=True)
                print(f"new_action.shape: {new_action.shape}", flush=True)
                print(f"new_action_flat.shape: {new_action_flat.shape}", flush=True)
                print(f"期望的action_dim: {action.shape[1] * 4}", flush=True)
                print(f"new_action.requires_grad: {new_action.requires_grad}", flush=True)
                print(f"new_action.grad_fn: {new_action.grad_fn}", flush=True)
                if new_action.grad_fn is None:
                    print("⚠ 警告: new_action没有梯度！", flush=True)
                    print("   这会导致Q loss无法反传到policy，policy权重无法更新", flush=True)
                else:
                    print("✓ new_action有梯度，Q loss可以反传到policy", flush=True)
                print(f"{'='*60}\n", flush=True)
                self._grad_check_printed = True
            
            # 验证new_action_flat的形状是否与action一致
            expected_action_dim = action.shape[1] * 4
            if new_action_flat.shape[1] != expected_action_dim:
                raise RuntimeError(
                    f"new_action_flat的形状不匹配: 期望 [B, {expected_action_dim}], "
                    f"实际 [B, {new_action_flat.shape[1]}]. "
                    f"x0_pred.shape={x0_pred.shape}, new_action.shape={new_action.shape}"
                )
            
            # 计算Q值损失（对应ql_diffusion.py中的q_loss计算方式）
            # 如果提供了r_fun，直接使用reward函数；否则使用Q网络
            if self.r_fun is None:
                # 使用Q网络计算Q值
                # 注意：critic在策略学习时作为固定函数使用，梯度会通过new_action传播到policy
                # 虽然梯度也会传播到critic的参数，但我们只更新policy，不更新critic
                self.critic.eval()  # 设置为eval模式，但不影响梯度计算
                q1_new_action, q2_new_action = self.critic(state_features, new_action_flat)
                self.critic.train()  # 恢复train模式
                
                # 计算新动作的reward（用于TensorBoard记录）
                with torch.no_grad():
                    new_reward = compute_reward(state_features, new_action_flat)  # [B] 或 scalar
                    if new_reward.dim() == 0:
                        new_reward_mean = new_reward.item()
                    else:
                        new_reward_mean = new_reward.mean().item()
                    new_reward_values.append(new_reward_mean)
                    
                    # 记录q1_new_action的平均值（detach以避免影响梯度）
                    q1_new_action_mean = q1_new_action.mean().detach().item()
                    q1_new_action_values.append(q1_new_action_mean)
                
                # 使用归一化的Q值，避免Q值过大或过小
                # 注意：分母使用.detach()，所以只有分子的梯度会传播
                if np.random.uniform() > 0.5:
                    q_loss = -q1_new_action.mean() / q2_new_action.abs().mean().detach()
                else:
                    q_loss = -q2_new_action.mean() / q1_new_action.abs().mean().detach()
            else:
                # 使用r_fun直接计算reward
                q_new_action = self.r_fun(state_features, new_action_flat)  # [B] 或 [B, 1]
                
                # 确保q_new_action是[B]形状
                if q_new_action.dim() > 1:
                    q_new_action = q_new_action.squeeze(-1)
                
                # 记录q_new_action的值（用于TensorBoard记录）
                with torch.no_grad():
                    if q_new_action.dim() == 0:
                        q_new_action_mean = q_new_action.item()
                    else:
                        q_new_action_mean = q_new_action.mean().item()
                    # 记录q_new_action的值（当使用r_fun时，q_new_action就是reward）
                    q1_new_action_values.append(q_new_action_mean)
                    new_reward_values.append(q_new_action_mean)  # 使用相同的值记录
                
                # 使用归一化的reward，避免reward过大或过小
                # 注意：分母使用.detach()，所以只有分子的梯度会传播
                lmbda = self.eta / q_new_action.abs().mean().detach()
                q_loss = -lmbda * q_new_action.mean()
            
            # 获取ego_planning_loss
            ego_planning_loss = loss.get('ego_planning_loss', torch.tensor(0.0).to(self.device))
            
            # 将Q值损失乘以权重后加到ego_planning_loss上（对应ql_diffusion.py中的actor_loss = bc_loss + self.eta * q_loss）
            # 梯度流：q_loss -> new_action -> policy，q_loss会参与反向传播并更新policy参数
            # 虽然梯度也会传播到critic参数，但我们只调用policy_optimizer.step()，所以critic不会被更新
            ego_planning_loss_with_q = ego_planning_loss + self.eta * q_loss ######################

            bc_loss = loss['loss'] = loss['neighbor_prediction_loss'] + self.policy_config.alpha_planning_loss * loss['ego_planning_loss']
            
            # 合并损失
            diffusion_loss = loss.get('neighbor_prediction_loss', torch.tensor(0.0).to(self.device)) + \
                           self.policy_config.alpha_planning_loss * ego_planning_loss_with_q#######################
            
            # 更新策略（前100个epoch只更新critic，不更新policy）
            if epoch is not None and epoch < 0:
                # 前100个epoch：只更新critic，不更新policy
                # 不计算policy的梯度，也不更新policy参数
                # 注意：虽然会计算一些不必要的梯度，但为了代码简洁，暂时保留
                pass  # 跳过policy更新
            else:
                # 100个epoch后：正常更新policy
                self.policy_optimizer.zero_grad()
                diffusion_loss.backward()
                # if self.grad_norm > 0:
                #     nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.grad_norm, norm_type=2)
                self.policy_optimizer.step()
            
            # 更新target网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            # 更新policy target网络（前100个epoch不更新）
            if epoch is None or epoch >= 0:
                for param, target_param in zip(self.policy.parameters(), self.policy_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            diffusion_losses.append(diffusion_loss.item())
            bc_losses.append(bc_loss.item())  # 记录纯行为克隆损失
            critic_losses.append(critic_loss.item())
            q_losses.append(q_loss.item())  # 记录Q值损失
            
            # 更新进度条显示：显示行为克隆损失而不是包含Q值的diffusion_loss
            avg_bc_loss = np.mean(bc_losses)
            avg_critic_loss = np.mean(critic_losses)
            avg_diffusion_loss = np.mean(diffusion_losses)
            avg_q_loss = np.mean(q_losses)
            pbar.set_postfix({
                'BC_l': f'{avg_bc_loss:.4f}',  # 显示行为克隆损失
                'Cri_l': f'{avg_critic_loss:.4f}',
                'Q_l': f'{avg_q_loss:.4f}',  # Q值损失
                'Dif_l': f'{avg_diffusion_loss:.4f}'
            })
            
            self.step += 1
        
        pbar.close()
        
        # 返回行为克隆损失、critic损失、diffusion损失、Q损失、q1_new_action平均值和new_reward平均值（用于日志和TensorBoard）
        return (
            np.mean(bc_losses), 
            np.mean(critic_losses), 
            np.mean(diffusion_losses), 
            np.mean(q_losses),
            np.mean(q1_new_action_values),
            np.mean(new_reward_values)
        )
    
    def save_checkpoint(self, save_path: str, epoch: int, bc_loss: float, critic_loss: float):
        """保存模型checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'step': self.step,
            'policy_state_dict': self.policy.state_dict(),
            'policy_target_state_dict': self.policy_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'bc_loss': bc_loss,  # 保存行为克隆损失
            'critic_loss': critic_loss,
        }
        
        # 保存最新checkpoint
        latest_path = os.path.join(save_path, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # 保存带epoch编号的checkpoint
        epoch_path = os.path.join(save_path, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, epoch_path)
        
        return latest_path, epoch_path


def set_seed(seed: int, device: str = "cuda"):
    """
    设置随机种子以确保实验可重复
    
    Args:
        seed: 随机种子值
        device: 设备类型（'cuda' 或 'cpu'）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    
    # 设置PyTorch的确定性行为（可能会降低性能）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    print(f"随机种子已设置为: {seed}", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Diffusion Q-learning Training')
    parser.add_argument('--data_dir', type=str, default='/mnt/data/test/test_data', help='path to npz data')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr_policy', type=float, default=3e-4, help='policy learning rate')
    parser.add_argument('--lr_critic', type=float, default=3e-4, help='critic learning rate')
    parser.add_argument('--discount', type=float, default=0.99, help='discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='target network update coefficient')
    parser.add_argument('--grad_norm', type=float, default=1.0, help='gradient norm clipping')
    parser.add_argument('--normalization_file', type=str, default='normalization.json', help='normalization file path')
    parser.add_argument('--save_dir', type=str, default='./training_log/ql_diffusion', help='directory to save logs and checkpoints')
    parser.add_argument('--log_interval', type=int, default=10, help='log metrics every N epochs')
    parser.add_argument('--save_interval', type=int, default=50, help='save checkpoint every N epochs')
    parser.add_argument('--save_best', action='store_true', help='save best model based on total loss')
    parser.add_argument('--resume_checkpoint', type=str, default=None, 
                       help='Path to checkpoint file to resume training or load pretrained policy. '
                            'Note: Policy weights are automatically loaded from base_weight/model_epoch_500_trainloss_0.0486.pth. '
                            'This parameter is kept for compatibility but policy weights will be loaded from base_weight first.')
    parser.add_argument('--eta', type=float, default=1.0, 
                       help='Weight for Q-value loss added to ego_planning_loss (corresponds to eta in ql_diffusion.py)')
    parser.add_argument('--use_r_fun', action='store_true',
                       help='Use r_fun (reward function) directly instead of Q-network for gradient descent')
    # Reward归一化控制：默认启用，使用 --no_reward_normalize 禁用
    parser.add_argument('--no_reward_normalize', dest='reward_normalize', action='store_false', default=True,
                       help='Disable Z-score normalization for reward (default: enabled)')
    parser.add_argument('--no_tensorboard', action='store_true',
                       help='Disable TensorBoard logging to avoid background thread errors')
    parser.add_argument('--max_q_backup', action='store_true',
                       help='Use Max Q Backup to reduce Q-value overestimation (sample multiple next actions and take min Q)')
    parser.add_argument('--max_q_backup_samples', type=int, default=10,
                       help='Number of action samples for Max Q Backup (default: 10)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: None, use random seed)')
    
    args = parser.parse_args()
    
    # 设置随机种子（如果指定）
    if args.seed is not None:
        set_seed(args.seed, args.device)
    else:
        print("未指定随机种子，使用系统默认随机行为", flush=True)
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    
    # 创建带时间戳的保存目录
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    save_dir_with_timestamp = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir_with_timestamp, exist_ok=True)
    
    tb_log_dir = os.path.join(save_dir_with_timestamp, 'tb')
    checkpoint_dir = os.path.join(save_dir_with_timestamp, 'checkpoints')
    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 初始化TensorBoard writer，添加错误处理
    # 如果TensorBoard写入失败，禁用TensorBoard以避免训练中断
    writer = None
    tb_enabled = False
    tb_error_count = 0  # 记录TensorBoard错误次数
    max_tb_errors = 10  # 最大错误次数，超过后禁用TensorBoard
    
    # 如果用户明确禁用TensorBoard，跳过初始化
    if args.no_tensorboard:
        print(f"\n{'='*60}", flush=True)
        print(f"TensorBoard logging is disabled (--no_tensorboard)", flush=True)
        print(f"Training will continue without TensorBoard logging.", flush=True)
        print(f"Checkpoints will be saved to: {checkpoint_dir}", flush=True)
        print(f"{'='*60}\n", flush=True)
        writer = None
        tb_enabled = False
    else:
        # 设置线程异常处理器，捕获TensorBoard后台线程的错误（Python 3.8+）
        import threading
        if hasattr(threading, 'excepthook'):
            original_excepthook = threading.excepthook
            
            def tensorboard_exception_handler(exc):
                """处理TensorBoard后台线程的异常"""
                if isinstance(exc.exception, OSError):
                    # TensorBoard相关错误，只打印一次警告
                    if not hasattr(tensorboard_exception_handler, '_warned'):
                        print(f"\n{'='*60}", flush=True)
                        print(f"Warning: TensorBoard background thread error detected:", flush=True)
                        print(f"  Error: {exc.exception}", flush=True)
                        print(f"  This is a known issue with TensorBoard on some file systems.", flush=True)
                        print(f"  Training will continue, but TensorBoard logging may be unreliable.", flush=True)
                        print(f"  Consider using --no_tensorboard to disable TensorBoard completely.", flush=True)
                        print(f"{'='*60}\n", flush=True)
                        tensorboard_exception_handler._warned = True
                else:
                    # 其他异常，使用默认处理
                    original_excepthook(exc)
            
            threading.excepthook = tensorboard_exception_handler
        
        try:
            # 尝试创建TensorBoard writer，增加flush间隔以减少写入频率
            # 使用更大的 flush_secs 和 max_queue 来减少写入频率，降低出错概率
            # 增加 flush_secs 到 300 秒（5分钟），减少后台线程写入频率
            writer = SummaryWriter(log_dir=tb_log_dir, flush_secs=300, max_queue=200)
            tb_enabled = True
            print(f"\n{'='*60}", flush=True)
            print(f"Training session: {timestamp}", flush=True)
            print(f"{'='*60}", flush=True)
            print(f"TensorBoard logs will be saved to: {tb_log_dir}", flush=True)
            print(f"Checkpoints will be saved to: {checkpoint_dir}", flush=True)
            print(f"To view TensorBoard, run: tensorboard --logdir {tb_log_dir}", flush=True)
            print(f"{'='*60}\n", flush=True)
        except (OSError, IOError, Exception) as e:
            print(f"\n{'='*60}", flush=True)
            print(f"Warning: Failed to initialize TensorBoard writer: {e}", flush=True)
            print(f"Error type: {type(e).__name__}", flush=True)
            print(f"Training will continue without TensorBoard logging.", flush=True)
            print(f"Checkpoints will be saved to: {checkpoint_dir}", flush=True)
            print(f"{'='*60}\n", flush=True)
            writer = None
            tb_enabled = False
    
    # 创建数据集和数据加载器
    dataset = NPZDataset(args.data_dir, device)
    
    # 创建训练数据加载器
    data_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # 计算reward的统计量（用于Z-score归一化）
    print(f"\n{'='*60}", flush=True)
    print("计算Reward统计量（用于Z-score归一化）...", flush=True)
    print(f"{'='*60}", flush=True)
    
    rewards_list = []
    sample_count = min(3000, len(dataset))  # 采样计算，避免太慢
    print(f"采样 {sample_count} 个样本计算reward统计量...", flush=True)
    
    for i in range(sample_count):
        state, action, reward, next_state, not_done = dataset[i]
        # 提取state_features
        state_features = extract_state_features({k: v.unsqueeze(0) for k, v in state.items()}, use_all_features=False)
        state_features = state_features.squeeze(0)
        action_flat = action.reshape(-1)
        # 计算reward（不归一化）
        computed_reward = compute_reward(state_features, action_flat, normalize=False)
        if computed_reward.numel() == 1:
            rewards_list.append(computed_reward.item())
        else:
            rewards_list.append(computed_reward.mean().item())
    
    rewards_array = np.array(rewards_list)
    reward_mean = float(rewards_array.mean())
    reward_std = float(rewards_array.std())
    
    print(f"Reward统计量（基于 {sample_count} 个样本）:", flush=True)
    print(f"  均值: {reward_mean:.6f}", flush=True)
    print(f"  标准差: {reward_std:.6f}", flush=True)
    print(f"  最小值: {rewards_array.min():.6f}", flush=True)
    print(f"  最大值: {rewards_array.max():.6f}", flush=True)
    
    # 设置全局变量
    global REWARD_MEAN, REWARD_STD, REWARD_NORMALIZE
    REWARD_MEAN = reward_mean
    REWARD_STD = reward_std
    # 使用命令行参数控制归一化，如果未指定则默认为True（启用）
    REWARD_NORMALIZE = args.reward_normalize if args.reward_normalize is not None else True
    
    if REWARD_NORMALIZE:
        print(f"✓ Reward Z-score归一化已启用", flush=True)
    else:
        print(f"✗ Reward Z-score归一化已禁用", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # 计算状态特征维度
    # use_all_features=False: 只使用ego和neighbor（原始方案，维度362）
    # use_all_features=True: 使用所有特征（维度会大幅增加，需要更多计算资源）
    USE_ALL_FEATURES = False  # 可以改为True以使用更多状态信息
    
    sample_state, _, _, _, _ = dataset[0]
    sample_state = {k: v.unsqueeze(0).to(device) for k, v in sample_state.items()}
    state_feature_dim = extract_state_features(sample_state, use_all_features=USE_ALL_FEATURES).shape[1]
    print(f"State feature dimension: {state_feature_dim} (use_all_features={USE_ALL_FEATURES})", flush=True)
    
    # 动作维度
    _, sample_action, _, _, _ = dataset[0]
    action_dim = sample_action.shape[0] * sample_action.shape[1]  # T * 4
    print(f"Action dimension: {action_dim}", flush=True)
    
    # 创建配置对象
    class PolicyConfig:
        def __init__(self, device):
            self.hidden_dim = 192
            self.num_heads = 6
            self.encoder_depth = 3
            self.decoder_depth = 3
            self.encoder_drop_path_rate = 0.1
            self.decoder_drop_path_rate = 0.1
            self.agent_num = 32
            self.static_objects_num = 5
            self.lane_num = 70
            self.route_num = 25
            self.time_len = 21
            self.future_len = 80
            self.lane_len = 20
            self.static_objects_state_dim = 10
            self.predicted_neighbor_num = 10
            self.diffusion_model_type = "x_start"
            self.device = device
            self.alpha_planning_loss = 1.0
            self.state_normalizer = None
            self.observation_normalizer = None
            self.guidance_fn = None
    
    policy_config = PolicyConfig(device)
    
    # 创建归一化器
    import json
    import copy
    
    normalization_file = args.normalization_file
    if not os.path.exists(normalization_file):
        normalization_file = os.path.join(os.path.dirname(__file__), normalization_file)
    
    if os.path.exists(normalization_file):
        class NormalizerArgs:
            def __init__(self, normalization_file_path, predicted_neighbor_num):
                self.normalization_file_path = normalization_file_path
                self.predicted_neighbor_num = predicted_neighbor_num
        
        norm_args = NormalizerArgs(normalization_file, policy_config.predicted_neighbor_num)
        state_normalizer = StateNormalizer.from_json(norm_args)
        observation_normalizer = ObservationNormalizer.from_json(norm_args)
        
        policy_config.state_normalizer = state_normalizer
        policy_config.observation_normalizer = observation_normalizer
        
        print(f"Normalizers loaded from {normalization_file}", flush=True)
    else:
        print(f"Warning: Normalization file not found: {normalization_file}", flush=True)
        state_normalizer = None
        observation_normalizer = None
    
    # 创建r_fun（可选）：如果使用r_fun，则直接使用reward函数而不是Q网络
    # r_fun应该接受(state_features, action_flat)并返回reward tensor
    r_fun = None
    
    if args.use_r_fun:
        # 定义r_fun：直接使用compute_reward函数
        def r_fun_wrapper(state_features: torch.Tensor, action_flat: torch.Tensor) -> torch.Tensor:
            """
            Reward函数包装器，直接计算reward用于梯度下降
            
            Args:
                state_features: [B, state_feature_dim] - 状态特征
                action_flat: [B, action_dim] - 动作（flatten后的未来轨迹）
            
            Returns:
                reward: [B] - 奖励值（已归一化）
            """
            return compute_reward(state_features, action_flat, normalize=True)
        
        r_fun = r_fun_wrapper
        print("使用 r_fun 直接计算reward（不使用Q网络）", flush=True)
    else:
        print("使用 Q网络 计算Q值", flush=True)
    
    # 打印Max Q Backup设置
    if args.max_q_backup:
        print(f"启用 Max Q Backup (采样数量: {args.max_q_backup_samples})", flush=True)
    else:
        print("未启用 Max Q Backup (使用标准方法)", flush=True)
    
    # 初始化agent
    agent = QL_Diffusion(
        policy_config=policy_config,
        state_feature_dim=state_feature_dim,
        action_dim=action_dim,
        device=device,
        discount=args.discount,
        tau=args.tau,
        lr_policy=args.lr_policy,
        lr_critic=args.lr_critic,
        grad_norm=args.grad_norm,
        state_normalizer=state_normalizer,
        observation_normalizer=observation_normalizer,
        resume_checkpoint=args.resume_checkpoint,
        eta=args.eta,
        r_fun=r_fun,
        max_q_backup=args.max_q_backup,
        max_q_backup_samples=args.max_q_backup_samples,
    )
    
    print("Agent initialized!", flush=True)
    
    # 训练循环
    iterations = len(data_loader)
    print(f"\n开始训练，共 {args.num_epochs} 个epoch，每个epoch {iterations} 个batch", flush=True)
    print(f"前100个epoch：只更新critic网络，不更新policy", flush=True)
    print(f"100个epoch后：正常更新critic和policy\n", flush=True)
    
    # 用于保存最佳模型
    best_diffusion_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        # 显示当前训练阶段
        if epoch == 0:
            print(f"Epoch {epoch}: 只更新critic网络（不更新policy）", flush=True)
        elif epoch == 1:
            print(f"Epoch {epoch}: 开始正常更新critic和policy", flush=True)
        
        bc_loss, critic_loss, diffusion_loss, q_loss, q1_new_action_mean, new_reward_mean = agent.train(data_loader, iterations=iterations, epoch=epoch)
        
        # 记录到TensorBoard（添加错误计数和自动禁用机制）
        if writer is not None and tb_enabled:
            try:
                writer.add_scalar('Loss/BC_Loss', bc_loss, epoch)  # 行为克隆损失
                writer.add_scalar('Loss/Critic_Loss', critic_loss, epoch)
                writer.add_scalar('Loss/Diffusion_Loss', diffusion_loss, epoch)
                writer.add_scalar('Loss/Q_Loss', q_loss, epoch)  # Q值损失
                writer.add_scalar('Q_Value/Q1_New_Action', q1_new_action_mean, epoch)  # 新动作的Q1值
                writer.add_scalar('Reward/New_Action_Reward', new_reward_mean, epoch)  # 新动作的reward
                tb_error_count = 0  # 成功写入，重置错误计数
            except (OSError, IOError, Exception) as e:
                tb_error_count += 1
                if tb_error_count <= 3:  # 只打印前3次错误，避免刷屏
                    print(f"Warning: Failed to write to TensorBoard (error {tb_error_count}/{max_tb_errors}): {e}", flush=True)
                if tb_error_count >= max_tb_errors:
                    print(f"\n{'='*60}", flush=True)
                    print(f"TensorBoard has failed {max_tb_errors} times. Disabling TensorBoard logging.", flush=True)
                    print(f"Training will continue without TensorBoard.", flush=True)
                    print(f"{'='*60}\n", flush=True)
                    tb_enabled = False
                    try:
                        writer.close()
                    except:
                        pass
                    writer = None
        
        # 记录学习率
        if writer is not None and tb_enabled:
            try:
                writer.add_scalar('Learning_Rate/Policy', args.lr_policy, epoch)
                writer.add_scalar('Learning_Rate/Critic', args.lr_critic, epoch)
            except (OSError, IOError, Exception) as e:
                tb_error_count += 1
                if tb_error_count <= 3:
                    print(f"Warning: Failed to write learning rate to TensorBoard: {e}", flush=True)
                if tb_error_count >= max_tb_errors:
                    tb_enabled = False
                    try:
                        writer.close()
                    except:
                        pass
                    writer = None
        
        # 保存checkpoint
        if epoch % args.save_interval == 0:
            latest_path, epoch_path = agent.save_checkpoint(checkpoint_dir, epoch, bc_loss, critic_loss)
            print(f'\nCheckpoint saved: {latest_path}', flush=True)
            print(f'Checkpoint saved: {epoch_path}', flush=True)
        
        # 保存最佳模型
        if args.save_best and diffusion_loss < best_diffusion_loss:
            best_diffusion_loss = diffusion_loss
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            agent.save_checkpoint(checkpoint_dir, epoch, bc_loss, critic_loss)
            # 重命名为best_model.pth
            latest_path = os.path.join(checkpoint_dir, 'latest.pth')
            if os.path.exists(latest_path):
                import shutil
                shutil.copy(latest_path, best_path)
            print(f'\nBest model saved (diffusion_loss: {best_diffusion_loss:.4f}): {best_path}', flush=True)
            if writer is not None and tb_enabled:
                try:
                    writer.add_scalar('Loss/Best_diffusion_loss', best_diffusion_loss, epoch)
                except (OSError, IOError, Exception) as e:
                    tb_error_count += 1
                    if tb_error_count <= 3:
                        print(f"Warning: Failed to write to TensorBoard: {e}", flush=True)
                    if tb_error_count >= max_tb_errors:
                        tb_enabled = False
                        try:
                            writer.close()
                        except:
                            pass
                        writer = None
        
        if epoch % args.log_interval == 0:
            print(f'\nEpoch: {epoch} BC_loss: {bc_loss:.4f} Critic_loss: {critic_loss:.4f} diffusion_loss: {diffusion_loss:.4f}', flush=True)
    
    # 保存最终模型
    print("\n保存最终模型...", flush=True)
    final_path, _ = agent.save_checkpoint(checkpoint_dir, args.num_epochs, bc_loss, critic_loss)
    print(f"Final checkpoint saved: {final_path}", flush=True)
    
    # 关闭TensorBoard writer（安全关闭）
    if writer is not None:
        try:
            writer.close()
        except (OSError, IOError, Exception) as e:
            print(f"Warning: Error closing TensorBoard writer: {e}", flush=True)
    
    print("\n" + "="*60, flush=True)
    print("Training completed!", flush=True)
    print("="*60, flush=True)
    print(f"Training session: {timestamp}", flush=True)
    if tb_enabled:
        print(f"TensorBoard logs saved to: {tb_log_dir}", flush=True)
    else:
        print(f"TensorBoard logging was disabled due to errors.", flush=True)
    print(f"Checkpoints saved to: {checkpoint_dir}", flush=True)
    print(f"Full save directory: {save_dir_with_timestamp}", flush=True)
    print("="*60, flush=True)


if __name__ == "__main__":
    main()

