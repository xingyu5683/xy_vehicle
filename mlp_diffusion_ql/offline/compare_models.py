#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : compare_models.py
@Description: 对比评估不同训练方法的模型性能
"""
import os
import sys
import os.path as osp
import argparse
import yaml
import csv
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import gymnasium as gym

# 添加项目路径
current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 添加CARLA agents模块路径
carla_pythonapi_path = "/workspace/volumes/hpc-al-sh01/Carla/CARLA_0.9.13/PythonAPI/carla"
if carla_pythonapi_path not in sys.path:
    sys.path.insert(0, carla_pythonapi_path)

import gym_carla
import carla

try:
    from core import QL_Diffusion, carla_env_reward_function, flatten_obs
except ImportError:
    from models import QL_Diffusion
    from reward_functions import carla_env_reward_function
    from utils import flatten_obs

from util.run_util import load_config, set_seed


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, env, agent, model_name, use_custom_reward=False, desired_speed=8.1, device='cuda', base_seed=None):
        """
        参数:
            env: CARLA环境
            agent: QL_Diffusion agent
            model_name: 模型名称（用于标识）
            use_custom_reward: 是否使用自定义奖励函数
            desired_speed: 期望速度
            device: 设备
            base_seed: 基础随机种子
        """
        self.env = env
        self.agent = agent
        self.model_name = model_name
        self.use_custom_reward = use_custom_reward
        self.desired_speed = desired_speed
        self.device = device
        self.base_seed = base_seed
        
        # 统计数据
        self.episode_rewards = []
        self.episode_lengths = []
        self.collision_counts = []
        self.off_road_counts = []
        self.success_counts = []
        self.min_distances = []
        
    def evaluate(self, n_episodes=10, max_steps=1000, verbose=True):
        """
        评估模型性能
        
        参数:
            n_episodes: 评估episode数
            max_steps: 每个episode最大步数
            verbose: 是否打印详细信息
        
        返回:
            results: 评估结果字典
        """
        self.agent.actor.eval()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"开始评估模型: {self.model_name}")
            print(f"{'='*60}")
        
        pbar = tqdm(range(n_episodes), desc=f"评估 {self.model_name}")
        
        for episode in pbar:
            # 为每个episode设置确定性种子，确保两个模型面对相同的场景
            episode_seed = None
            if self.base_seed is not None:
                episode_seed = self.base_seed + episode
                # 在reset前设置Python和NumPy随机种子
                random.seed(episode_seed)
                np.random.seed(episode_seed)
            
            obs, info = self.env.reset(seed=episode_seed)
            state = flatten_obs(obs)
            
            episode_reward = 0
            episode_length = 0
            collision_count = 0
            off_road_count = 0
            min_distance = float('inf')
            
            done = False
            truncated = False
            
            while not (done or truncated) and episode_length < max_steps:
                # 使用策略选择动作（无噪声）
                action = self.agent.sample_action(state)
                
                # 执行动作
                next_obs, reward, done, truncated, info = self.env.step(action)
                next_state = flatten_obs(next_obs)
                
                # 使用自定义奖励函数（如果需要）
                if self.use_custom_reward:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                    done_tensor = torch.FloatTensor([[float(done)]]).to(self.device)
                    
                    ego_collision = float(info.get('ego_collision', False)) if info.get('ego_collision', False) is not None else 0.0
                    ego_off_road = float(info.get('ego_off_road', False)) if info.get('ego_off_road', False) is not None else 0.0
                    ego_min_dis = float(info.get('ego_min_dis', 100.0)) if info.get('ego_min_dis', None) is not None else 100.0
                    
                    info_tensor = torch.FloatTensor([[ego_collision, ego_off_road, ego_min_dis]]).to(self.device)
                    
                    reward_tensor = carla_env_reward_function(
                        state_tensor, action_tensor, next_state_tensor, done_tensor, info_tensor, self.desired_speed
                    )
                    reward = reward_tensor.item()
                
                # 统计碰撞和偏离道路
                if info.get('ego_collision', False):
                    collision_count += 1
                if info.get('ego_off_road', False):
                    off_road_count += 1
                
                # 记录最小距离
                ego_min_dis = info.get('ego_min_dis', None)
                if ego_min_dis is not None and ego_min_dis < min_distance:
                    min_distance = ego_min_dis
                
                episode_reward += reward
                episode_length += 1
                state = next_state
            
            # 判断是否成功
            success = truncated and not done
            
            # 记录数据
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.collision_counts.append(collision_count)
            self.off_road_counts.append(off_road_count)
            self.success_counts.append(int(success))
            self.min_distances.append(min_distance if min_distance != float('inf') else 0)
            
            pbar.set_postfix({
                'R': f'{episode_reward:.2f}',
                'L': episode_length,
                'Success': int(success)
            })
        
        pbar.close()
        
        # 计算统计结果
        results = {
            'model_name': self.model_name,
            'n_episodes': n_episodes,
            'avg_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'avg_length': np.mean(self.episode_lengths),
            'std_length': np.std(self.episode_lengths),
            'success_rate': np.mean(self.success_counts),
            'avg_collisions': np.mean(self.collision_counts),
            'avg_off_road': np.mean(self.off_road_counts),
            'avg_min_distance': np.mean([d for d in self.min_distances if d > 0]),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
        }
        
        if verbose:
            self._print_results(results)
        
        self.agent.actor.train()
        
        return results
    
    def _print_results(self, results):
        """打印评估结果"""
        print(f"\n{'-'*60}")
        print(f"评估结果: {results['model_name']}")
        print(f"{'-'*60}")
        print(f"评估Episode数: {results['n_episodes']}")
        print(f"平均奖励: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"平均长度: {results['avg_length']:.1f} ± {results['std_length']:.1f}")
        print(f"成功率: {results['success_rate']:.2%}")
        print(f"平均碰撞次数: {results['avg_collisions']:.2f}")
        print(f"平均偏离道路次数: {results['avg_off_road']:.2f}")
        if results['avg_min_distance'] > 0:
            print(f"平均最小距离: {results['avg_min_distance']:.2f}m")
        print(f"{'-'*60}\n")


def compare_models(
        bc_model_path,
        ql_model_path,
        config_file,
        output_dir='./log/model_comparison',
        n_eval_episodes=20,
        max_episode_steps=1000,
        use_custom_reward=False,
        desired_speed=8.1,
        device='cuda',
        seed=None):
    """
    对比两个模型的性能
    
    参数:
        bc_model_path: BC模型路径
        ql_model_path: QL微调后模型路径
        config_file: 环境配置文件
        output_dir: 输出目录
        n_eval_episodes: 评估episode数
        max_episode_steps: 每个episode最大步数
        use_custom_reward: 是否使用自定义奖励
        desired_speed: 期望速度
        device: 设备
        seed: 随机种子
    """
    # 设置随机种子
    if seed is not None:
        set_seed(seed)
        print(f">> 随机种子已设置: {seed}")
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f">> 输出目录: {output_dir}")
    
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f">> 使用设备: {device}")
    
    # 加载环境配置
    params = load_config(config_file)
    
    # 创建临时环境以获取维度信息
    print(f">> 创建临时CARLA环境以获取维度信息...")
    temp_env = gym.make('carla-v0_test1', env_params=params)
    obs, info = temp_env.reset()
    state = flatten_obs(obs)
    state_dim = len(state)
    action_dim = temp_env.action_space.shape[0]
    temp_env.close()
    print(f">> 状态维度: {state_dim}, 动作维度: {action_dim}")
    
    # 为了确保公平对比，每个模型使用独立的环境实例
    
    # ============ 加载BC模型 ============
    print(f"\n>> 加载BC模型: {bc_model_path}")
    bc_agent = QL_Diffusion(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=1.0,
        device=device,
        discount=0.99,
        tau=0.005,
        eta=1.0,
        beta_schedule='linear',
        n_timesteps=100,
        hidden_dim=256,
        lr=3e-4,
        mode='whole_grad',
        critic_num_layers=3,
        update_critic=False,  # BC模型不需要critic
    )
    bc_agent.load_model(bc_model_path)
    
    # ============ 评估BC模型（使用独立环境）============
    print(f"\n{'='*60}")
    print(f"为BC模型创建独立的CARLA环境...")
    print(f"{'='*60}")
    
    # 重要：在创建环境前设置Python和NumPy的随机种子
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        print(f">> BC环境随机种子: {seed}")
    
    bc_env = gym.make('carla-v0_test1', env_params=params)
    
    bc_evaluator = ModelEvaluator(
        env=bc_env,
        agent=bc_agent,
        model_name='BC (纯行为克隆)',
        use_custom_reward=use_custom_reward,
        desired_speed=desired_speed,
        device=device,
        base_seed=seed  # 传递基础种子
    )
    bc_results = bc_evaluator.evaluate(n_episodes=n_eval_episodes, max_steps=max_episode_steps)
    
    # 关闭BC环境，释放资源
    print(f"\n>> 关闭BC环境，释放CARLA资源...")
    bc_env.close()
    
    # 等待一段时间，确保CARLA服务器完全清理
    import time
    print(f">> 等待3秒，确保CARLA服务器状态清理...")
    time.sleep(3)
    
    # ============ 加载QL模型 ============
    print(f"\n>> 加载QL微调模型: {ql_model_path}")
    ql_agent = QL_Diffusion(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=1.0,
        device=device,
        discount=0.99,
        tau=0.005,
        eta=1.0,
        beta_schedule='linear',
        n_timesteps=100,
        hidden_dim=256,
        lr=3e-4,
        mode='whole_grad',
        critic_num_layers=3,
        update_critic=True,
    )
    ql_agent.load_model(ql_model_path)
    
    # ============ 评估QL模型（使用独立环境）============
    print(f"\n{'='*60}")
    print(f"为QL模型创建独立的CARLA环境...")
    print(f"{'='*60}")
    
    # 重要：重新设置Python和NumPy的随机种子，确保与BC评估一致
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        print(f">> QL环境随机种子: {seed}")
    
    ql_env = gym.make('carla-v0_test1', env_params=params)
    
    ql_evaluator = ModelEvaluator(
        env=ql_env,
        agent=ql_agent,
        model_name='QL (Q-learning微调)',
        use_custom_reward=use_custom_reward,
        desired_speed=desired_speed,
        device=device,
        base_seed=seed  # 传递基础种子，确保与BC评估相同
    )
    ql_results = ql_evaluator.evaluate(n_episodes=n_eval_episodes, max_steps=max_episode_steps)
    
    # 关闭QL环境
    print(f"\n>> 关闭QL环境...")
    ql_env.close()
    
    # ============ 生成对比报告 ============
    print(f"\n{'='*60}")
    print(f"模型对比总结")
    print(f"{'='*60}")
    
    comparison = {
        '指标': ['平均奖励', '奖励标准差', '平均长度', '成功率', '平均碰撞', '平均偏离道路'],
        'BC模型': [
            f"{bc_results['avg_reward']:.2f}",
            f"{bc_results['std_reward']:.2f}",
            f"{bc_results['avg_length']:.1f}",
            f"{bc_results['success_rate']:.2%}",
            f"{bc_results['avg_collisions']:.2f}",
            f"{bc_results['avg_off_road']:.2f}",
        ],
        'QL模型': [
            f"{ql_results['avg_reward']:.2f}",
            f"{ql_results['std_reward']:.2f}",
            f"{ql_results['avg_length']:.1f}",
            f"{ql_results['success_rate']:.2%}",
            f"{ql_results['avg_collisions']:.2f}",
            f"{ql_results['avg_off_road']:.2f}",
        ],
        '改进幅度': [
            f"{((ql_results['avg_reward'] - bc_results['avg_reward']) / abs(bc_results['avg_reward']) * 100):.1f}%",
            f"-",
            f"{((ql_results['avg_length'] - bc_results['avg_length']) / bc_results['avg_length'] * 100):.1f}%",
            f"{((ql_results['success_rate'] - bc_results['success_rate']) * 100):.1f}%",
            f"{((bc_results['avg_collisions'] - ql_results['avg_collisions']) / max(bc_results['avg_collisions'], 1e-6) * 100):.1f}%",
            f"{((bc_results['avg_off_road'] - ql_results['avg_off_road']) / max(bc_results['avg_off_road'], 1e-6) * 100):.1f}%",
        ]
    }
    
    # 打印对比表格
    print(f"\n{'-'*80}")
    print(f"{'指标':<20} {'BC模型':<20} {'QL模型':<20} {'改进幅度':<20}")
    print(f"{'-'*80}")
    for i in range(len(comparison['指标'])):
        print(f"{comparison['指标'][i]:<20} {comparison['BC模型'][i]:<20} {comparison['QL模型'][i]:<20} {comparison['改进幅度'][i]:<20}")
    print(f"{'-'*80}\n")
    
    # 保存CSV报告
    csv_file = osp.join(output_dir, 'comparison_summary.csv')
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['指标', 'BC模型', 'QL模型', '改进幅度'])
        for i in range(len(comparison['指标'])):
            writer.writerow([
                comparison['指标'][i],
                comparison['BC模型'][i],
                comparison['QL模型'][i],
                comparison['改进幅度'][i]
            ])
    print(f">> CSV报告已保存: {csv_file}")
    
    # 保存详细结果
    details_file = osp.join(output_dir, 'detailed_results.csv')
    with open(details_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'BC_Reward', 'BC_Length', 'QL_Reward', 'QL_Length'])
        for i in range(n_eval_episodes):
            writer.writerow([
                i + 1,
                bc_results['episode_rewards'][i],
                bc_results['episode_lengths'][i],
                ql_results['episode_rewards'][i],
                ql_results['episode_lengths'][i]
            ])
    print(f">> 详细结果已保存: {details_file}")
    
    # ============ 生成可视化图表 ============
    print(f"\n>> 生成可视化图表...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 每个episode的奖励对比
    ax1 = axes[0, 0]
    episodes = list(range(1, n_eval_episodes + 1))
    ax1.plot(episodes, bc_results['episode_rewards'], 'o-', label='BC Model', linewidth=2, markersize=6)
    ax1.plot(episodes, ql_results['episode_rewards'], 's-', label='QL Model', linewidth=2, markersize=6)
    ax1.axhline(y=bc_results['avg_reward'], color='blue', linestyle='--', alpha=0.5, label='BC Avg')
    ax1.axhline(y=ql_results['avg_reward'], color='orange', linestyle='--', alpha=0.5, label='QL Avg')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('Episode Rewards Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. 每个episode的长度对比
    ax2 = axes[0, 1]
    ax2.plot(episodes, bc_results['episode_lengths'], 'o-', label='BC Model', linewidth=2, markersize=6)
    ax2.plot(episodes, ql_results['episode_lengths'], 's-', label='QL Model', linewidth=2, markersize=6)
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Episode Length', fontsize=12)
    ax2.set_title('Episode Lengths Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. 奖励分布（箱线图）
    ax3 = axes[1, 0]
    box_data = [bc_results['episode_rewards'], ql_results['episode_rewards']]
    bp = ax3.boxplot(box_data, labels=['BC Model', 'QL Model'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax3.set_ylabel('Total Reward', fontsize=12)
    ax3.set_title('Reward Distribution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 关键指标对比（柱状图）
    ax4 = axes[1, 1]
    metrics = ['Success\nRate', 'Avg\nCollisions', 'Avg\nOff-road']
    bc_values = [bc_results['success_rate'], bc_results['avg_collisions'], bc_results['avg_off_road']]
    ql_values = [ql_results['success_rate'], ql_results['avg_collisions'], ql_results['avg_off_road']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, bc_values, width, label='BC Model', color='lightblue')
    bars2 = ax4.bar(x + width/2, ql_values, width, label='QL Model', color='lightcoral')
    
    ax4.set_ylabel('Value', fontsize=12)
    ax4.set_title('Key Metrics Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics, fontsize=10)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = osp.join(output_dir, 'comparison_plot.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f">> 对比图表已保存: {plot_file}")
    
    # 保存配置
    config_dict = {
        'comparison_info': {
            'bc_model_path': bc_model_path,
            'ql_model_path': ql_model_path,
            'config_file': osp.abspath(config_file),
            'output_dir': osp.abspath(output_dir),
            'timestamp': timestamp,
            'device': str(device),
        },
        'evaluation_settings': {
            'n_eval_episodes': n_eval_episodes,
            'max_episode_steps': max_episode_steps,
            'use_custom_reward': use_custom_reward,
            'desired_speed': float(desired_speed),
            'seed': seed,
        },
        'bc_results': {
            'avg_reward': float(bc_results['avg_reward']),
            'std_reward': float(bc_results['std_reward']),
            'avg_length': float(bc_results['avg_length']),
            'success_rate': float(bc_results['success_rate']),
            'avg_collisions': float(bc_results['avg_collisions']),
            'avg_off_road': float(bc_results['avg_off_road']),
        },
        'ql_results': {
            'avg_reward': float(ql_results['avg_reward']),
            'std_reward': float(ql_results['std_reward']),
            'avg_length': float(ql_results['avg_length']),
            'success_rate': float(ql_results['success_rate']),
            'avg_collisions': float(ql_results['avg_collisions']),
            'avg_off_road': float(ql_results['avg_off_road']),
        },
    }
    
    config_yaml_file = osp.join(output_dir, 'comparison_config.yaml')
    with open(config_yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f">> 配置已保存: {config_yaml_file}")
    
    print(f"\n>> 对比评估完成! 输出目录: {output_dir}")
    
    return bc_results, ql_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='模型对比评估 - BC vs QL微调')
    
    # 模型路径
    parser.add_argument('--bc_model', type=str, required=True, help='BC模型路径（actor.pth所在目录）')
    parser.add_argument('--ql_model', type=str, required=True, help='QL微调模型路径（actor.pth所在目录）')
    
    # 环境配置
    parser.add_argument('--config', type=str, default='../configs/base.yaml', help='环境配置文件路径')
    parser.add_argument('--output_dir', type=str, default='./log/model_comparison', help='输出目录')
    
    # 评估参数
    parser.add_argument('--n_eval_episodes', type=int, default=20, help='评估episode数')
    parser.add_argument('--max_episode_steps', type=int, default=1000, help='每个episode最大步数')
    parser.add_argument('--use_custom_reward', action='store_true', help='使用自定义奖励函数')
    parser.add_argument('--desired_speed', type=float, default=8.1, help='期望速度')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='设备')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    
    args = parser.parse_args()
    
    # 检查模型路径
    if not osp.exists(args.bc_model):
        raise FileNotFoundError(f"BC模型路径不存在: {args.bc_model}")
    if not osp.exists(args.ql_model):
        raise FileNotFoundError(f"QL模型路径不存在: {args.ql_model}")
    
    # 检查配置文件
    if not osp.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    
    # 开始对比评估
    bc_results, ql_results = compare_models(
        bc_model_path=args.bc_model,
        ql_model_path=args.ql_model,
        config_file=args.config,
        output_dir=args.output_dir,
        n_eval_episodes=args.n_eval_episodes,
        max_episode_steps=args.max_episode_steps,
        use_custom_reward=args.use_custom_reward,
        desired_speed=args.desired_speed,
        device=args.device,
        seed=args.seed
    )
    
    print("\n>> 完成!")

