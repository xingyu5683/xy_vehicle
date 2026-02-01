#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : online_train_carla.py
@Description: 在线强化学习训练主函数 - 与CARLA环境实时交互并训练QL_Diffusion模型
"""
import os
import sys
import os.path as osp
# 添加项目根目录到 Python 路径
current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 添加CARLA agents模块路径（根据您的实际路径修改）
carla_pythonapi_path = "/workspace/volumes/hpc-al-sh01/Carla/CARLA_0.9.13/PythonAPI/carla"
if carla_pythonapi_path not in sys.path:
    sys.path.insert(0, carla_pythonapi_path)

import yaml
import csv
import random
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import gymnasium as gym
import gym_carla
import carla


try:
    from .core import QL_Diffusion, carla_env_reward_function, flatten_obs
except ImportError:
    try:
        from core import QL_Diffusion, carla_env_reward_function, flatten_obs
    except ImportError:
        from models import QL_Diffusion
        from reward_functions import carla_env_reward_function
        from utils import flatten_obs

from util.run_util import load_config, set_seed


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity, state_dim, action_dim, device='cuda'):
        """
        参数:
            capacity: 缓冲区最大容量
            state_dim: 状态维度
            action_dim: 动作维度
            device: 设备
        """
        self.capacity = capacity
        self.device = device
        
        # 预分配内存
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32)
        
        self.ptr = 0  # 当前写入位置
        self.size = 0  # 当前缓冲区大小
    
    def add(self, state, action, reward, next_state, done):
        """添加一条经验"""
        self.states[self.ptr] = torch.FloatTensor(state)
        self.actions[self.ptr] = torch.FloatTensor(action)
        self.rewards[self.ptr] = torch.FloatTensor([reward])
        self.next_states[self.ptr] = torch.FloatTensor(next_state)
        self.dones[self.ptr] = torch.FloatTensor([done])
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """采样一个batch的经验"""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return (
            self.states[indices].to(self.device),
            self.actions[indices].to(self.device),
            self.rewards[indices].to(self.device),
            self.next_states[indices].to(self.device),
            self.dones[indices].to(self.device)
        )
    
    def __len__(self):
        return self.size


def evaluate_policy(agent, env, n_episodes=5, max_steps=1000):
    """
    评估策略性能
    
    参数:
        agent: QL_Diffusion agent
        env: CARLA环境
        n_episodes: 评估episode数
        max_steps: 每个episode的最大步数
    
    返回:
        avg_reward: 平均累积奖励
        avg_length: 平均episode长度
        success_rate: 成功率（完成任务的比例）
    """
    agent.actor.eval()
    
    total_rewards = []
    total_lengths = []
    success_count = 0
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        state = flatten_obs(obs)
        
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        while not (done or truncated) and episode_length < max_steps:
            # 使用当前策略选择动作（无噪声）
            #with torch.no_grad():
                #state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                #action = agent.sample_action(state_tensor)
                #action = action.cpu().numpy().flatten()

            # sample_action 方法期望接收 numpy 数组，会在内部转换为 tensor
            action = agent.sample_action(state)
            
            # 执行动作
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = flatten_obs(next_obs)
            
            episode_reward += reward
            episode_length += 1
            
            state = next_state
        
        total_rewards.append(episode_reward)
        total_lengths.append(episode_length)
        
        # 判断是否成功（根据info或done标志）
        if done and episode_reward > 0:  # 可以根据具体任务调整成功标准
            success_count += 1
    
    agent.actor.train()
    
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(total_lengths)
    success_rate = success_count / n_episodes
    
    return avg_reward, avg_length, success_rate


def train_online_rl(
        config_file,
        output_dir='./log/online_train_output',
        total_timesteps=1000000,
        batch_size=256,
        buffer_size=1000000,
        learning_starts=1000,
        train_freq=1,
        gradient_steps=1,
        lr=3e-4,
        device='cuda',
        save_freq=10000,
        eval_freq=5000,
        n_eval_episodes=5,
        use_tensorboard=True,
        update_critic=True,
        # QL_Diffusion 特定参数
        discount=0.99,
        tau=0.005,
        eta=1.0,
        beta_schedule='linear',
        n_timesteps=100,
        hidden_dim=256,
        mode='whole_grad',
        critic_num_layers=3,
        add_timestamp=True,
        use_custom_reward=False,
        desired_speed=8.1,
        critic_lr=None,
        max_grad_norm=1.0,
        seed=None,
        # 探索参数
        exploration_noise=0.1,
        max_episode_steps=1000,
        # 预训练模型
        pretrained_model_path=None):
    """
    在线强化学习训练主函数
    
    参数:
        config_file: CARLA环境配置文件路径
        output_dir: 输出目录
        total_timesteps: 总训练步数
        batch_size: 批次大小
        buffer_size: 回放缓冲区大小
        learning_starts: 开始学习的步数
        train_freq: 训练频率（每N步训练一次）
        gradient_steps: 每次训练的梯度步数
        lr: 学习率
        device: 设备
        save_freq: 保存频率
        eval_freq: 评估频率
        n_eval_episodes: 评估episode数
        use_tensorboard: 是否使用tensorboard
        update_critic: 是否更新Critic
        discount: 折扣因子
        tau: 软更新系数
        eta: Q-learning权重
        beta_schedule: beta调度方式
        n_timesteps: 扩散时间步数
        hidden_dim: 隐藏层维度
        mode: 采样模式
        critic_num_layers: Critic网络层数
        add_timestamp: 是否添加时间戳
        use_custom_reward: 是否使用自定义奖励函数
        desired_speed: 期望速度
        critic_lr: Critic学习率
        max_grad_norm: 梯度裁剪
        seed: 随机种子
        exploration_noise: 探索噪声标准差
        max_episode_steps: 每个episode最大步数
        pretrained_model_path: 预训练模型路径
    """
    # 设置随机种子
    if seed is not None:
        set_seed(seed)
        print(f">> 随机种子已设置: {seed}")
    else:
        print(f">> 未设置随机种子，每次运行结果可能不同")
    
    # 添加时间戳到输出目录
    if add_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = output_dir.rstrip('/')
        output_dir = f"{output_dir}_{timestamp}"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f">> 输出目录: {output_dir}")
    
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f">> 使用设备: {device}")
    
    # 加载环境配置
    params = load_config(config_file)
    
    # 创建环境
    print(f">> 创建CARLA环境...")
    env = gym.make('carla-v0_test1', env_params=params)
    
    # 获取状态和动作维度
    obs, info = env.reset()
    state = flatten_obs(obs)
    state_dim = len(state)
    action_dim = env.action_space.shape[0]
    
    print(f">> 状态维度: {state_dim}, 动作维度: {action_dim}")
    
    # 创建回放缓冲区
    replay_buffer = ReplayBuffer(
        capacity=buffer_size,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )
    print(f">> 回放缓冲区大小: {buffer_size}")
    
    # 创建QL_Diffusion agent
    agent = QL_Diffusion(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=1.0,
        device=device,
        discount=discount,
        tau=tau,
        eta=eta,
        beta_schedule=beta_schedule,
        n_timesteps=n_timesteps,
        hidden_dim=hidden_dim,
        lr=lr,
        r_fun=None,
        mode=mode,
        critic_num_layers=critic_num_layers,
        critic_lr=critic_lr,
        max_grad_norm=max_grad_norm,
        update_critic=bool(update_critic),
    )
    
    # 加载预训练模型（如果提供）
    if pretrained_model_path is not None:
        agent.load_model(pretrained_model_path)
        print(f">> 已加载预训练模型: {pretrained_model_path}")
    
    # 保存训练配置
    config_dict = {
        'training_info': {
            'config_file': osp.abspath(config_file),
            'output_dir': osp.abspath(output_dir),
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(device),
            'pretrained_model': pretrained_model_path,
        },
        'environment_info': {
            'state_dim': state_dim,
            'action_dim': action_dim,
            'max_episode_steps': max_episode_steps,
        },
        'training_hyperparameters': {
            'total_timesteps': total_timesteps,
            'batch_size': batch_size,
            'buffer_size': buffer_size,
            'learning_starts': learning_starts,
            'train_freq': train_freq,
            'gradient_steps': gradient_steps,
            'lr': float(lr),
            'critic_lr': float(critic_lr) if critic_lr is not None else float(lr),
            'save_freq': save_freq,
            'eval_freq': eval_freq,
            'n_eval_episodes': n_eval_episodes,
            'use_tensorboard': use_tensorboard,
            'update_critic': bool(update_critic),
            'use_custom_reward': use_custom_reward,
            'desired_speed': float(desired_speed),
            'max_grad_norm': float(max_grad_norm),
            'seed': seed,
            'exploration_noise': float(exploration_noise),
        },
        'ql_diffusion_hyperparameters': {
            'discount': float(discount),
            'tau': float(tau),
            'eta': float(eta),
            'beta_schedule': beta_schedule,
            'n_timesteps': n_timesteps,
            'hidden_dim': hidden_dim,
            'mode': mode,
            'critic_num_layers': critic_num_layers,
            'max_action': 1.0,
        },
    }
    
    config_yaml_file = osp.join(output_dir, 'training_config.yaml')
    with open(config_yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f">> 训练配置已保存到: {config_yaml_file}")
    
    # TensorBoard
    writer = None
    if use_tensorboard:
        log_dir = osp.join(output_dir, 'tensorboard')
        writer = SummaryWriter(log_dir=log_dir)
        print(f">> TensorBoard日志: {log_dir}")
    
    # CSV日志
    train_csv_file = osp.join(output_dir, 'training_log.csv')
    eval_csv_file = osp.join(output_dir, 'evaluation_log.csv')
    
    with open(train_csv_file, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['timestep', 'episode', 'episode_reward', 'episode_length', 
                           'bc_loss', 'q_loss', 'critic_loss', 'buffer_size'])
    
    with open(eval_csv_file, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['timestep', 'avg_reward', 'avg_length', 'success_rate'])
    
    # 训练循环
    print(f"\n>> 开始在线训练: 总步数 {total_timesteps}")
    
    obs, info = env.reset()
    state = flatten_obs(obs)
    episode_reward = 0
    episode_length = 0
    episode_num = 0
    
    # 用于记录最近的训练loss
    recent_bc_loss = 0
    recent_q_loss = 0
    recent_critic_loss = 0
    
    pbar = tqdm(total=total_timesteps, desc="Online Training")
    
    for timestep in range(1, total_timesteps + 1):
        # 选择动作
        if timestep < learning_starts:
            # 初始阶段使用随机动作
            action = env.action_space.sample()
        else:
            # 使用策略选择动作，加入探索噪声
            #with torch.no_grad():
                #state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                #action = agent.sample_action(state_tensor)
                #action = action.cpu().numpy().flatten()
                
                # 添加探索噪声
                #noise = np.random.normal(0, exploration_noise, size=action.shape)
                #action = np.clip(action + noise, -1.0, 1.0)

            # 注意：sample_action 接受 numpy array，内部会转换为 tensor
            action = agent.sample_action(state)  # 直接传入 numpy array
            
            # 添加探索噪声
            noise = np.random.normal(0, exploration_noise, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        
        # 执行动作
        next_obs, reward, done, truncated, info = env.step(action)
        next_state = flatten_obs(next_obs)
        
        # 自定义奖励（如果需要）
        if use_custom_reward:
            #reward = carla_env_reward_function(state, action, next_state, desired_speed)
            # 将输入转换为 tensor 格式以调用奖励函数
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            done_tensor = torch.FloatTensor([float(done)]).unsqueeze(0).to(device)
           
            # 从 info 字典中提取所需信息（ego_collision, ego_off_road, ego_min_dis）
            # 注意：某些值可能为 None，需要使用默认值
            ego_collision = info.get('ego_collision', False)
            ego_off_road = info.get('ego_off_road', False)
            ego_min_dis = info.get('ego_min_dis', None)

            # 转换为浮点数，处理 None 和布尔值
            ego_collision = float(ego_collision) if ego_collision is not None else 0.0
            ego_off_road = float(ego_off_road) if ego_off_road is not None else 0.0
            ego_min_dis = float(ego_min_dis) if ego_min_dis is not None else 100.0
            
            info_tensor = torch.FloatTensor([[ego_collision, ego_off_road, ego_min_dis]]).to(device)
            
            # 计算自定义奖励
            reward_tensor = carla_env_reward_function(
                state_tensor, action_tensor, next_state_tensor, done_tensor, info_tensor, desired_speed
            )
            reward = reward_tensor.item()

        # 添加到回放缓冲区
        replay_buffer.add(state, action, reward, next_state, float(done))
        
        episode_reward += reward
        episode_length += 1
        
        state = next_state
        
        # Episode结束
        if done or truncated or episode_length >= max_episode_steps:
            # 记录episode信息
            episode_num += 1
            
            # 写入CSV
            with open(train_csv_file, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([timestep, episode_num, episode_reward, episode_length,
                                   recent_bc_loss, recent_q_loss, recent_critic_loss, len(replay_buffer)])
            
            if writer is not None:
                writer.add_scalar('train/episode_reward', episode_reward, timestep)
                writer.add_scalar('train/episode_length', episode_length, timestep)
            
            pbar.set_postfix({
                'Ep': episode_num,
                'R': f'{episode_reward:.2f}',
                'L': episode_length,
                'Buf': len(replay_buffer)
            })
            
            # 重置环境
            obs, info = env.reset()
            state = flatten_obs(obs)
            episode_reward = 0
            episode_length = 0
        
        # 训练模型
        if timestep >= learning_starts and timestep % train_freq == 0:
            for _ in range(gradient_steps):
                # 从回放缓冲区采样
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = \
                    replay_buffer.sample(batch_size)
                
                # 训练一步
                bc_loss, q_loss, critic_loss = agent.train_step(
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
                )
                
                recent_bc_loss = bc_loss
                recent_q_loss = q_loss
                recent_critic_loss = critic_loss
            
            if writer is not None:
                writer.add_scalar('train/bc_loss', bc_loss, timestep)
                writer.add_scalar('train/q_loss', q_loss, timestep)
                writer.add_scalar('train/critic_loss', critic_loss, timestep)
        
        # 评估
        if timestep % eval_freq == 0:
            print(f"\n>> 评估模型 (timestep {timestep})...")
            avg_reward, avg_length, success_rate = evaluate_policy(
                agent, env, n_episodes=n_eval_episodes, max_steps=max_episode_steps
            )
            
            print(f"   平均奖励: {avg_reward:.2f}, 平均长度: {avg_length:.1f}, 成功率: {success_rate:.2%}")
            
            with open(eval_csv_file, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([timestep, avg_reward, avg_length, success_rate])
            
            if writer is not None:
                writer.add_scalar('eval/avg_reward', avg_reward, timestep)
                writer.add_scalar('eval/avg_length', avg_length, timestep)
                writer.add_scalar('eval/success_rate', success_rate, timestep)
        
        # 保存模型
        if timestep % save_freq == 0:
            model_dir = osp.join(output_dir, f'timestep_{timestep}')
            agent.save_model(model_dir)
            print(f"\n>> 模型已保存: {model_dir}")
        
        pbar.update(1)
    
    pbar.close()
    
    # 保存最终模型
    final_model_dir = osp.join(output_dir, 'final')
    agent.save_model(final_model_dir)
    print(f"\n>> 最终模型已保存: {final_model_dir}")
    
    if writer is not None:
        writer.close()
    
    # 关闭环境
    env.close()
    
    # 更新配置
    config_dict['training_info']['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    config_dict['training_info']['total_episodes'] = episode_num
    config_dict['training_info']['final_model_path'] = osp.join(final_model_dir, 'actor.pth')
    
    with open(config_yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"\n>> 训练完成! 输出目录: {output_dir}")
    return agent


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='在线强化学习训练 (CARLA) - QL_Diffusion')
    parser.add_argument('--config', type=str, default='../configs/base.yaml', help='环境配置文件路径')
    parser.add_argument('--output_dir', type=str, default='./log/online_train_output', help='输出目录')
    parser.add_argument('--total_timesteps', type=int, default=1000000, help='总训练步数')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--buffer_size', type=int, default=1000000, help='回放缓冲区大小')
    parser.add_argument('--learning_starts', type=int, default=1000, help='开始学习的步数')
    parser.add_argument('--train_freq', type=int, default=1, help='训练频率（每N步）')
    parser.add_argument('--gradient_steps', type=int, default=1, help='每次训练的梯度步数')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--critic_lr', type=float, default=None, help='Critic学习率')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='设备')
    parser.add_argument('--save_freq', type=int, default=10000, help='保存频率（每N步）')
    parser.add_argument('--eval_freq', type=int, default=5000, help='评估频率（每N步）')
    parser.add_argument('--n_eval_episodes', type=int, default=5, help='评估episode数')
    parser.add_argument('--no_tensorboard', action='store_true', help='不使用tensorboard')
    parser.add_argument('--no_critic_update', action='store_true', help='不更新Critic')
    
    # QL_Diffusion 参数
    parser.add_argument('--discount', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--tau', type=float, default=0.005, help='软更新系数')
    parser.add_argument('--eta', type=float, default=1.0, help='Q-learning权重')
    parser.add_argument('--beta_schedule', type=str, default='linear', help='beta调度')
    parser.add_argument('--n_timesteps', type=int, default=100, help='扩散时间步数')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--mode', type=str, default='whole_grad', help='采样模式')
    parser.add_argument('--critic_num_layers', type=int, default=3, help='Critic层数')
    parser.add_argument('--no_timestamp', action='store_true', help='不添加时间戳')
    parser.add_argument('--use_custom_reward', action='store_true', help='使用自定义奖励')
    parser.add_argument('--desired_speed', type=float, default=8.1, help='期望速度')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    
    # 探索参数
    parser.add_argument('--exploration_noise', type=float, default=0.1, help='探索噪声标准差')
    parser.add_argument('--max_episode_steps', type=int, default=1000, help='每个episode最大步数')
    
    # 预训练模型
    parser.add_argument('--pretrained_model', type=str, default=None, help='预训练模型路径')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not osp.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    
    # 开始训练
    agent = train_online_rl(
        config_file=args.config,
        output_dir=args.output_dir,
        total_timesteps=args.total_timesteps,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        lr=args.lr,
        critic_lr=args.critic_lr,
        device=args.device,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        use_tensorboard=not args.no_tensorboard,
        update_critic=not args.no_critic_update,
        discount=args.discount,
        tau=args.tau,
        eta=args.eta,
        beta_schedule=args.beta_schedule,
        n_timesteps=args.n_timesteps,
        hidden_dim=args.hidden_dim,
        mode=args.mode,
        critic_num_layers=args.critic_num_layers,
        add_timestamp=not args.no_timestamp,
        use_custom_reward=args.use_custom_reward,
        desired_speed=args.desired_speed,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        exploration_noise=args.exploration_noise,
        max_episode_steps=args.max_episode_steps,
        pretrained_model_path=args.pretrained_model
    )
    
    print(">> 完成!")

