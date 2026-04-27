#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : train.py
@Description: 离线强化学习训练主函数
"""
import os
import os.path as osp
import yaml
import csv
import random
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

try:
    from .core import QL_Diffusion, DataSampler, carla_env_reward_function
except ImportError:
    try:
        from core import QL_Diffusion, DataSampler, carla_env_reward_function
    except ImportError:
        # 向后兼容：如果core不存在，尝试从旧路径导入
        from models import QL_Diffusion
        from data_sampler import DataSampler
        from reward_functions import carla_env_reward_function


def train_offline_rl(data_file, 
                     output_dir='./log/offline_train_output',
                     epochs=100,
                     iterations_per_epoch=None,
                     batch_size=256,
                     lr=3e-4,
                     device='cuda',
                     save_freq=100,
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
                     reward_tune='no',
                     critic_lr=None,
                     max_grad_norm=1.0,
                     seed=None):
    """
    离线强化学习训练主函数（使用QL_Diffusion）
    
    参数:
        data_file: HDF5数据文件路径
        output_dir: 输出目录（如果add_timestamp=True，会自动添加时间戳）
        epochs: 训练轮数
        iterations_per_epoch: 每轮训练迭代次数（如果为None，则根据数据集大小自动计算）
        batch_size: 批次大小
        lr: 学习率
        device: 设备 ('cuda' 或 'cpu')
        save_freq: 保存频率（每N个epoch保存一次）
        use_tensorboard: 是否使用tensorboard
        update_critic: 是否更新Critic（默认True；若False，则只训练Actor的BC部分，actor_loss=bc_loss）
        discount: 折扣因子
        tau: 软更新系数
        eta: Q-learning权重
        beta_schedule: beta调度方式 ('linear', 'cosine', 'vp')
        n_timesteps: 扩散时间步数
        hidden_dim: Diffusion Policy的隐藏层维度（Critic使用固定的渐进式降维结构）
        mode: 采样模式 ('whole_grad', 't_middle', 't_last', 'last_few')
        critic_num_layers: Critic网络隐藏层数量 (默认3层)
        add_timestamp: 是否在输出目录名中添加时间戳（默认True，避免覆盖）
        use_custom_reward: 是否使用自定义奖励函数重新计算reward（默认False，使用数据集中的reward）
        desired_speed: 期望速度 (m/s)，用于自定义奖励函数，默认8.1
        reward_tune: 奖励调整方式 ('no', 'normalize', 'iql_antmaze')
        critic_lr: Critic学习率（如果为None，则使用lr）
        max_grad_norm: 梯度裁剪的最大范数
        seed: 随机种子（如果为None，则不设置种子，每次运行结果不同）
    """
    # 设置随机种子（如果提供）
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        print(f">> 随机种子已设置: {seed}")
    else:
        print(f">> 未设置随机种子，每次运行结果可能不同")
    
    # 添加时间戳到输出目录（避免覆盖之前的训练结果）
    if add_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 如果output_dir以/结尾，去掉
        output_dir = output_dir.rstrip('/')
        # 添加时间戳
        output_dir = f"{output_dir}_{timestamp}"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f">> 输出目录: {output_dir}")
    
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f">> 使用设备: {device}")
    
    # 决定使用哪个reward函数
    reward_function = None
    if use_custom_reward:
        # 使用与carla-v0_test1环境完全相同的奖励函数
        reward_function = carla_env_reward_function
        print(f">> 使用自定义奖励函数 (与carla-v0_test1环境相同, desired_speed={desired_speed} m/s)")
    else:
        print(f">> 使用数据集中的原始reward")
    
    # 加载数据
    data_sampler = DataSampler(
        data_file, 
        device=device, 
        reward_tune=reward_tune,
        reward_function=reward_function,
        desired_speed=desired_speed
    )
    
    # 计算每轮迭代次数
    if iterations_per_epoch is None:
        iterations_per_epoch = max(1, data_sampler.size // batch_size)
    
    # 保存训练配置到YAML文件
    config_dict = {
        'training_info': {
            'data_file': osp.abspath(data_file),
            'output_dir': osp.abspath(output_dir),
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(device),
        },
        'dataset_info': {
            'dataset_size': int(data_sampler.size),
            'state_dim': int(data_sampler.state_dim),
            'action_dim': int(data_sampler.action_dim),
            'obs_dim': int(data_sampler.obs_dim),
            'reward_range': [float(data_sampler.reward.min().item()), 
                           float(data_sampler.reward.max().item())],
        },
        'training_hyperparameters': {
            'epochs': epochs,
            'iterations_per_epoch': iterations_per_epoch,
            'batch_size': batch_size,
            'lr': float(lr),
            'critic_lr': float(critic_lr) if critic_lr is not None else float(lr),
            'save_freq': save_freq,
            'use_tensorboard': use_tensorboard,
            'update_critic': bool(update_critic),
            'reward_tune': reward_tune,
            'use_custom_reward': use_custom_reward,
            'desired_speed': float(desired_speed),
            'max_grad_norm': float(max_grad_norm),
            'seed': seed,
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
            'max_action': 1.0,  # carla-v0_test1的动作范围
            'max_q_backup': False,
        },
        'model_architecture': {
            'actor': {
                'type': 'Diffusion',
                'state_dim': int(data_sampler.state_dim),
                'action_dim': int(data_sampler.action_dim),
                'hidden_dim': hidden_dim,
                'n_timesteps': n_timesteps,
                'beta_schedule': beta_schedule,
            },
            'critic': {
                'type': 'DualQNetwork',
                'input_dim': int(data_sampler.state_dim + data_sampler.action_dim),
                'num_layers': critic_num_layers,
                'layer_structure': '69→512→256→128→1' if critic_num_layers == 3 else 'custom',
            },
        },
        'computed_values': {
            'total_iterations': epochs * iterations_per_epoch,
            'samples_per_epoch': iterations_per_epoch * batch_size,
        },
        'training_results': {
            'final_bc_loss': None,
            'final_q_loss': None,
            'final_critic_loss': None,
            'total_epochs_completed': None,
            'final_model_path': None,
            'final_q1_new_action': None,
            'final_q2_new_action': None,
        }
    }
    
    # 保存配置到YAML文件
    config_file = osp.join(output_dir, 'training_config.yaml')
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f">> 训练配置已保存到: {config_file}")
    
    # 创建QL_Diffusion agent
    print(f">> Diffusion Policy隐藏层维度: {hidden_dim}维")
    agent = QL_Diffusion(
        state_dim=data_sampler.state_dim,
        action_dim=data_sampler.action_dim,
        max_action=1.0,  # carla-v0_test1的动作范围是[-1, 1]
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
    
    # TensorBoard
    writer = None
    if use_tensorboard:
        log_dir = osp.join(output_dir, 'tensorboard')
        writer = SummaryWriter(log_dir=log_dir)
        print(f">> TensorBoard日志: {log_dir}")
    
    # 训练循环
    print(f"\n>> 开始训练: {epochs} epochs, {iterations_per_epoch} iterations/epoch")
    
    # 保存loss数据到CSV
    csv_file = osp.join(output_dir, 'training_loss.csv')
    with open(csv_file, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['epoch', 'bc_loss', 'q_loss', 'critic_loss', 'q1_new_action', 'q2_new_action', 'avg_reward'])
    
    # 使用tqdm显示训练进度条（增加宽度以显示所有信息）
    epoch_pbar = tqdm(range(1, epochs + 1), desc="Training", unit="epoch", 
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                      ncols=140)  # 增加进度条宽度以显示所有loss信息
    
    for epoch in epoch_pbar:
        # 训练一个epoch
        b_loss, q_loss, critic_loss, q1_new_action, q2_new_action, avg_reward = agent.train(data_sampler, iterations=iterations_per_epoch, batch_size=batch_size)
        
        # 更新进度条显示信息（使用字典格式，tqdm会自动格式化）
        epoch_pbar.set_postfix({
            'BC': f'{b_loss:.4f}',
            'Q': f'{q_loss:.4f}',
            'C': f'{critic_loss:.4f}',  # 使用更短的键名以节省空间
            'Q1': f'{q1_new_action:.4f}',  # q1_new_action的平均值
            'Q2': f'{q2_new_action:.4f}',   # q2_new_action的平均值
            'R': f'{avg_reward:.4f}'  # 平均reward
        })
        
        # 保存到CSV
        with open(csv_file, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([epoch, b_loss, q_loss, critic_loss, q1_new_action, q2_new_action, avg_reward])
        
        if writer is not None:
            writer.add_scalar('train/bc_loss', b_loss, epoch)
            writer.add_scalar('train/q_loss', q_loss, epoch)
            writer.add_scalar('train/critic_loss', critic_loss, epoch)
            writer.add_scalar('train/q1_new_action', q1_new_action, epoch)  # 添加q1_new_action到TensorBoard
            writer.add_scalar('train/q2_new_action', q2_new_action, epoch)  # 添加q2_new_action到TensorBoard
            writer.add_scalar('train/avg_reward', avg_reward, epoch)  # 添加平均reward到TensorBoard
        
        # 保存模型
        if epoch % save_freq == 0:
            model_dir = osp.join(output_dir, f'epoch_{epoch}')
            agent.save_model(model_dir)
            epoch_pbar.write(f">> 模型已保存到: {model_dir}")
    
    epoch_pbar.close()
    
    # 保存最终模型
    final_model_dir = osp.join(output_dir, 'final')
    agent.save_model(final_model_dir)
    
    if writer is not None:
        writer.close()
    
    # 读取最终loss值（从CSV文件）
    final_bc_loss = None
    final_q_loss = None
    final_critic_loss = None
    final_q1_new_action = None
    final_q2_new_action = None
    final_avg_reward = None
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                last_row = rows[-1]
                final_bc_loss = float(last_row['bc_loss'])
                final_q_loss = float(last_row['q_loss'])
                final_critic_loss = float(last_row['critic_loss'])
                if 'q1_new_action' in last_row:
                    final_q1_new_action = float(last_row['q1_new_action'])
                if 'q2_new_action' in last_row:
                    final_q2_new_action = float(last_row['q2_new_action'])
                if 'avg_reward' in last_row:
                    final_avg_reward = float(last_row['avg_reward'])
    except Exception as e:
        print(f"警告: 无法读取最终loss值: {e}")
    
    # 更新训练配置，添加训练结果
    config_dict['training_info']['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    config_dict['training_results'] = {
        'final_bc_loss': final_bc_loss,
        'final_q_loss': final_q_loss,
        'final_critic_loss': final_critic_loss,
        'final_q1_new_action': final_q1_new_action,
        'final_q2_new_action': final_q2_new_action,
        'final_avg_reward': final_avg_reward,
        'total_epochs_completed': epochs,
        'final_model_path': osp.join(final_model_dir, 'actor.pth'),
    }
    
    # 重新保存更新后的配置
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f">> 训练配置已更新: {config_file}")
    
    print(f"\n>> 训练完成! 模型保存在: {output_dir}")
    return agent


if __name__ == '__main__':
    import argparse
    import torch
    
    parser = argparse.ArgumentParser(description='离线强化学习训练 (carla-v0_test1) - QL_Diffusion')
    parser.add_argument('--data_file', type=str, required=True, help='HDF5数据文件路径')
    parser.add_argument('--output_dir', type=str, default='./log/offline_train_output', help='输出目录')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--iterations_per_epoch', type=int, default=None, help='每轮训练迭代次数（默认自动计算）')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--critic_lr', type=float, default=None, help='Critic学习率（默认与lr相同）')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='设备')
    parser.add_argument('--save_freq', type=int, default=1000, help='保存频率（每N个epoch）')
    parser.add_argument('--no_tensorboard', action='store_true', help='不使用tensorboard')
    parser.add_argument('--no_critic_update', action='store_true',
                        help='不更新Critic网络（仅训练Actor的BC部分：actor_loss=bc_loss，去掉q loss）')
    
    # QL_Diffusion 特定参数
    parser.add_argument('--discount', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--tau', type=float, default=0.005, help='软更新系数')
    parser.add_argument('--eta', type=float, default=1.0, help='Q-learning权重')
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear', 'cosine', 'vp'], help='beta调度方式')
    parser.add_argument('--n_timesteps', type=int, default=100, help='扩散时间步数')
    parser.add_argument('--hidden_dim', type=int, default=256, help='隐藏层维度')
    parser.add_argument('--mode', type=str, default='whole_grad', choices=['whole_grad', 't_middle', 't_last', 'last_few'], help='采样模式')
    parser.add_argument('--critic_num_layers', type=int, default=3, help='Critic网络隐藏层数量（默认3层，结构：69→512→256→128→1）')
    parser.add_argument('--no_timestamp', action='store_true', help='不在输出目录名中添加时间戳（默认会添加时间戳避免覆盖）')
    parser.add_argument('--use_custom_reward', action='store_true', help='使用自定义奖励函数重新计算reward（与carla-v0_test1环境相同，默认使用数据集中的reward）')
    parser.add_argument('--desired_speed', type=float, default=8.1, help='期望速度 (m/s)，用于自定义奖励函数，默认8.1')
    parser.add_argument('--reward_tune', type=str, default='no', choices=['no', 'normalize', 'iql_antmaze'], 
                        help='奖励调整方式: no(不调整), normalize(归一化), iql_antmaze(减去1.0)，默认no')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='梯度裁剪的最大范数（默认1.0）')
    parser.add_argument('--seed', type=int, default=None, help='随机种子（如果为None，则不设置种子，每次运行结果不同）')
    
    args = parser.parse_args()
    
    # 检查数据文件是否存在
    if not osp.exists(args.data_file):
        raise FileNotFoundError(f"数据文件不存在: {args.data_file}")
    
    # 开始训练
    agent = train_offline_rl(
        data_file=args.data_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        iterations_per_epoch=args.iterations_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        critic_lr=args.critic_lr,
        device=args.device,
        save_freq=args.save_freq,
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
        reward_tune=args.reward_tune,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed
    )
    
    print(">> 完成!")

