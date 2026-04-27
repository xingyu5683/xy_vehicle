#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : collect_offline_data_carla.py
@Description: 使用autopilot在carla-v0_test1环境中收集离线训练数据
              数据保存为HDF5格式，包含observations, actions, rewards, next_observations, dones
"""
import os
import sys
import os.path as osp

# 添加项目根目录到 Python 路径
current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(current_dir)  # offline 的父目录
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import h5py
import numpy as np
import gymnasium as gym
import gym_carla
import carla
import yaml
from datetime import datetime
from util.run_util import load_config, set_seed
from tqdm import tqdm
import time

try:
    from .core import flatten_obs
except ImportError:
    try:
        from core import flatten_obs
    except ImportError:
        # 向后兼容：如果core不存在，尝试从旧路径导入
        from utils import flatten_obs


def get_autopilot_action(env, tm=None, random_action_prob=0.05):
    """
    从环境的ego车辆获取autopilot控制动作，加入一定概率的随机动作以增加数据多样性
    
    参数:
        env: carla环境对象
        tm: TrafficManager对象（可选，如果提供则用于禁止变道）
        random_action_prob: 使用随机动作的概率（默认0.1，即10%）
    
    返回: [throttle, steer] 格式的动作，范围在 [-1, 1]
    """
    # 10%的概率使用随机动作，90%的概率使用autopilot动作
    if np.random.random() < random_action_prob:
        # 生成随机动作
        # longitudinal: [-1, 1] (throttle/brake)
        # steer: [-1, 1] (转向)
        random_action = np.array([
            np.random.uniform(-1.0, 1.0),  # longitudinal
            np.random.uniform(-1.0, 1.0)   # steer
        ], dtype=np.float32)
        return random_action
    
    # 90%的概率使用autopilot动作
    # 获取真实环境对象（可能被wrapped）
    real_env = env
    while hasattr(real_env, "env"):
        real_env = real_env.env
    
    # 启用autopilot
    real_env.ego.set_autopilot(True)
    
    # 禁止自车变道（使用TrafficManager）
    if tm is not None:
        try:
            tm.auto_lane_change(real_env.ego, False)
        except Exception as e:
            # 如果设置失败，不影响主要功能
            pass
    
    # 获取控制信号
    control = real_env.ego.get_control()
    
    # 将throttle和brake合并为longitudinal控制
    if control.throttle >= 0:
        longitudinal = control.throttle
    elif control.brake >= 0:
        longitudinal = -control.brake  # brake为负值
    else:
        longitudinal = 0.0
    
    # 返回 [throttle, steer] 格式，范围在 [-1, 1]
    action = np.array([longitudinal, control.steer], dtype=np.float32)
    
    return action


def collect_data(env, num_episodes=10, max_steps_per_episode=1000, output_file='carla_offline_dataset.hdf5', 
                 env_params=None, config_file_path=None):
    """
    收集离线训练数据
    
    参数:
        env: carla-v0_test1环境
        num_episodes: 收集的episode数量
        max_steps_per_episode: 每个episode的最大步数
        output_file: 输出HDF5文件名
        env_params: 环境参数字典（从配置文件加载）
        config_file_path: 配置文件路径
    """
    print(f">> 开始收集数据: {num_episodes} episodes, 每episode最多 {max_steps_per_episode} 步")
    
    # 打印环境配置信息
    if env_params is not None:
        print(f">> 环境配置:")
        print(f"   配置文件: {osp.abspath(config_file_path) if config_file_path else '未提供'}")
        print(f"   期望速度: {env_params.get('desired_speed', 'N/A')} m/s")
        print(f"   城镇: {env_params.get('town', 'N/A')}")
        print(f"   周车数量: {env_params.get('number_of_vehicles', 'N/A')}")
        print(f"   时间步长: {env_params.get('dt', 'N/A')} s")
        print(f"   最大episode步数: {env_params.get('max_time_episode', 'N/A')}")
    else:
        print(f">> 警告: 环境配置未提供，将无法记录配置信息")
    
    # 打印奖励函数配置信息
    desired_speed = float(env_params.get('desired_speed', 12.0)) if env_params else 12.0
    print(f">> 奖励函数配置:")
    print(f"   函数名称: carla_env._get_reward")
    print(f"   期望速度: {desired_speed} m/s")
    print(f"   碰撞惩罚: -2000.0")
    print(f"   偏离道路惩罚: -1000.0")
    print(f"   速度奖励权重: 1.0 (超速惩罚: -5.0)")
    print(f"   车道偏移惩罚: -1.0")
    print(f"   横向加速度惩罚: -0.2")
    print(f"   转向惩罚: 5.0 * (-steer^2)")
    
    # 初始化TrafficManager（用于禁止变道）
    tm = None
    try:
        # 获取真实环境对象
        real_env = env
        while hasattr(real_env, "env"):
            real_env = real_env.env
        
        # 尝试从环境对象获取world，然后获取client
        if hasattr(real_env, 'world'):
            # 通过world获取client（如果world有这个方法）
            try:
                world = real_env.world
                # 尝试获取client（某些Carla版本支持）
                if hasattr(world, 'get_client'):
                    client = world.get_client()
                else:
                    # 备用方案：创建新的client连接
                    client = carla.Client('localhost', 2000)
                    client.set_timeout(10.0)
                tm = client.get_trafficmanager(port=8000)
                print(">> TrafficManager初始化成功，已启用禁止变道功能")
            except Exception as e:
                print(f">> 警告: 无法初始化TrafficManager: {e}")
    except Exception as e:
        print(f">> 警告: 无法设置禁止变道功能: {e}")
    
    # 初始化数据缓冲区
    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []
    infos = []  # 保存info信息
    
    # 统计信息
    total_steps = 0
    episode_rewards = []
    
    # 收集数据
    for episode in tqdm(range(num_episodes), desc="Collecting episodes"):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        prev_obs = None
        
        while not done and step_count < max_steps_per_episode:
            # 获取autopilot动作（传入TrafficManager以禁止变道）
            action = get_autopilot_action(env, tm=tm)
            
            # 执行动作
            next_obs, reward, done, truncated, info = env.step(action)

            
            # 保存数据（需要前一个观测）
            if prev_obs is not None:
                # 展平观测
                obs_flat = flatten_obs(prev_obs)
                next_obs_flat = flatten_obs(next_obs)
                
                # 将info字典转换为数组格式: [ego_collision, ego_off_road, ego_min_dis]
                # ego_collision和ego_off_road是bool，转换为float (0.0或1.0)
                # ego_min_dis可能是None，需要处理
                info_array = np.array([
                    float(info.get('ego_collision', False)),
                    float(info.get('ego_off_road', False)),
                    float(info.get('ego_min_dis', 0.0)) if info.get('ego_min_dis') is not None else 0.0
                ], dtype=np.float32)
                
                observations.append(obs_flat)
                actions.append(action)
                rewards.append(reward)
                next_observations.append(next_obs_flat)
                dones.append(float(done or truncated))
                infos.append(info_array)
                
                total_steps += 1
                episode_reward += reward
            
            prev_obs = obs
            obs = next_obs
            step_count += 1
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f">> Episode {episode + 1}/{num_episodes}, "
                  f"Total steps: {total_steps}, "
                  f"Avg reward (last 10): {avg_reward:.2f}")
    
    # 转换为numpy数组
    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)
    next_observations = np.array(next_observations, dtype=np.float32)
    dones = np.array(dones, dtype=np.float32)
    infos = np.array(infos, dtype=np.float32)  # shape: (N, 3)
    
    print(f"\n>> 数据收集完成!")
    print(f">> 总步数: {total_steps}")
    print(f">> 观测维度: {observations.shape}")
    print(f">> 动作维度: {actions.shape}")
    print(f">> Info维度: {infos.shape}")
    print(f">> 平均奖励: {np.mean(episode_rewards):.2f}")
    print(f">> 奖励范围: [{np.min(rewards):.2f}, {np.max(rewards):.2f}]")
    
    # 统计info信息
    collision_count = np.sum(infos[:, 0])
    off_road_count = np.sum(infos[:, 1])
    print(f">> 碰撞次数: {int(collision_count)}")
    print(f">> 偏离道路次数: {int(off_road_count)}")
    if np.sum(infos[:, 2] > 0) > 0:
        print(f">> 最小距离范围: [{np.min(infos[infos[:, 2] > 0, 2]):.2f}, {np.max(infos[:, 2]):.2f}]")
    
    # 计算详细统计信息
    total_steps = len(rewards)
    collision_rate = collision_count / total_steps if total_steps > 0 else 0.0
    off_road_rate = off_road_count / total_steps if total_steps > 0 else 0.0
    
    # 奖励统计
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    min_reward = float(np.min(rewards))
    max_reward = float(np.max(rewards))
    median_reward = float(np.median(rewards))
    
    # Episode统计
    mean_episode_reward = float(np.mean(episode_rewards))
    std_episode_reward = float(np.std(episode_rewards))
    min_episode_reward = float(np.min(episode_rewards))
    max_episode_reward = float(np.max(episode_rewards))
    
    # 最小距离统计
    valid_min_dises = infos[infos[:, 2] > 0, 2]
    if len(valid_min_dises) > 0:
        mean_min_dis = float(np.mean(valid_min_dises))
        std_min_dis = float(np.std(valid_min_dises))
        min_min_dis = float(np.min(valid_min_dises))
        max_min_dis = float(np.max(valid_min_dises))
    else:
        mean_min_dis = 0.0
        std_min_dis = 0.0
        min_min_dis = 0.0
        max_min_dis = 0.0
    
    # 动作统计
    mean_action_longitudinal = float(np.mean(actions[:, 0]))
    std_action_longitudinal = float(np.std(actions[:, 0]))
    mean_action_steer = float(np.mean(actions[:, 1]))
    std_action_steer = float(np.std(actions[:, 1]))
    
    # 构建奖励函数配置（从carla_env.py的_get_reward方法提取）
    # 注意：这些权重是硬编码在carla_env.py中的，需要与代码保持一致
    reward_function_config = {
        'function_name': 'carla_env._get_reward',
        'description': 'Carla环境奖励函数（与gym_carla/envs/carla_env.py中的_get_reward一致）',
        'components': {
            'speed_reward': {
                'description': 'Forward driving reward (within speed limit)',
                'formula': '1.0 * speed if speed <= desired_speed else -5.0 * (speed - desired_speed)',
                'speed_weight': 1.0,
                'overspeed_penalty_weight': -5.0,
            },
            'lane_deviation_penalty': {
                'description': 'Lane deviation penalty (penalize offset from lane center)',
                'formula': '-1.0 * lateral_offset',
                'weight': -1.0,
            },
            'lateral_acceleration_penalty': {
                'description': 'Smooth driving penalty (lateral acceleration penalty)',
                'formula': '-0.2 * abs(a_lat) * speed^2',
                'weight': -0.2,
            },
            'steer_penalty': {
                'description': 'Smooth driving penalty (steer penalty)',
                'formula': '5 * (-steer^2)',
                'weight': 5.0,
            },
            'collision_penalty': {
                'description': 'Collision penalty',
                'formula': '-2000.0 if ego_collision else 0',
                'penalty': -2000.0,
            },
            'off_road_penalty': {
                'description': 'Off-road penalty',
                'formula': '-1000.0 if ego_off_road else 0',
                'penalty': -1000.0,
            },
        },
        'desired_speed': float(env_params.get('desired_speed', 12.0)) if env_params else 12.0,
    }
    
    # 构建环境配置信息
    environment_config = {}
    if env_params is not None:
        environment_config = {
            'config_file': osp.abspath(config_file_path) if config_file_path else None,
            'parameters': dict(env_params),  # 复制环境参数字典
        }
    else:
        environment_config = {
            'config_file': None,
            'parameters': {},
            'note': '环境配置未提供',
        }
    
    # 构建统计信息字典
    stats_dict = {
        'dataset_info': {
            'data_file': osp.abspath(output_file),
            'collection_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_episodes': int(num_episodes),
            'total_steps': int(total_steps),
            'mean_steps_per_episode': float(total_steps / num_episodes) if num_episodes > 0 else 0.0,
        },
        'environment_config': environment_config,
        'reward_function_config': reward_function_config,
        'data_dimensions': {
            'obs_dim': int(observations.shape[1]),
            'action_dim': int(actions.shape[1]),
            'info_dim': int(infos.shape[1]),
            'info_fields': 'ego_collision,ego_off_road,ego_min_dis',
        },
        'reward_statistics': {
            'per_step': {
                'mean': mean_reward,
                'std': std_reward,
                'min': min_reward,
                'max': max_reward,
                'median': median_reward,
            },
            'per_episode': {
                'mean': mean_episode_reward,
                'std': std_episode_reward,
                'min': min_episode_reward,
                'max': max_episode_reward,
            },
        },
        'safety_statistics': {
            'collision': {
                'total_count': int(collision_count),
                'rate': float(collision_rate),
                'rate_percentage': float(collision_rate * 100),
            },
            'off_road': {
                'total_count': int(off_road_count),
                'rate': float(off_road_rate),
                'rate_percentage': float(off_road_rate * 100),
            },
            'min_distance': {
                'mean': mean_min_dis,
                'std': std_min_dis,
                'min': min_min_dis,
                'max': max_min_dis,
                'valid_samples': int(len(valid_min_dises)),
            },
        },
        'action_statistics': {
            'longitudinal': {
                'mean': mean_action_longitudinal,
                'std': std_action_longitudinal,
                'range': [float(np.min(actions[:, 0])), float(np.max(actions[:, 0]))],
            },
            'steer': {
                'mean': mean_action_steer,
                'std': std_action_steer,
                'range': [float(np.min(actions[:, 1])), float(np.max(actions[:, 1]))],
            },
        },
    }
    
    # 保存为HDF5文件
    print(f"\n>> 保存数据到 {output_file}...")
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('observations', data=observations, compression='gzip')
        f.create_dataset('actions', data=actions, compression='gzip')
        f.create_dataset('rewards', data=rewards, compression='gzip')
        f.create_dataset('next_observations', data=next_observations, compression='gzip')
        f.create_dataset('done', data=dones, compression='gzip')
        f.create_dataset('info', data=infos, compression='gzip')  # 保存info信息
        
        # 保存元数据
        f.attrs['num_episodes'] = num_episodes
        f.attrs['total_steps'] = total_steps
        f.attrs['obs_dim'] = observations.shape[1]
        f.attrs['action_dim'] = actions.shape[1]
        f.attrs['info_dim'] = infos.shape[1]  # info维度 (3)
        f.attrs['mean_reward'] = mean_reward
        f.attrs['std_reward'] = std_reward
        # 保存info字段说明
        f.attrs['info_fields'] = 'ego_collision,ego_off_road,ego_min_dis'  # 字段名称
        
        # 保存环境配置信息
        if env_params is not None:
            f.attrs['config_file'] = osp.abspath(config_file_path) if config_file_path else 'unknown'
            f.attrs['desired_speed'] = float(env_params.get('desired_speed', 12.0))
            # 保存关键环境参数
            for key in ['number_of_vehicles', 'port', 'town', 'dt', 'max_time_episode', 
                       'desired_speed', 'seed', 'max_waypoints', 'perception_range', 
                       'max_nearby_vehicles']:
                if key in env_params:
                    f.attrs[f'env_{key}'] = env_params[key]
        
        # 保存奖励函数配置（关键参数）
        f.attrs['reward_function'] = 'carla_env._get_reward'
        f.attrs['reward_desired_speed'] = reward_function_config['desired_speed']
        f.attrs['reward_collision_penalty'] = reward_function_config['components']['collision_penalty']['penalty']
        f.attrs['reward_off_road_penalty'] = reward_function_config['components']['off_road_penalty']['penalty']
    
    print(f">> 数据已保存到 {output_file}")
    print(f">> 文件大小: {osp.getsize(output_file) / (1024**2):.2f} MB")
    
    # 保存统计信息到YAML文件
    yaml_file = osp.splitext(output_file)[0] + '_stats.yaml'
    print(f"\n>> 保存统计信息到 {yaml_file}...")
    with open(yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(stats_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f">> 统计信息已保存到 {yaml_file}")
    
    return output_file


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='收集carla-v0_test1环境的离线训练数据')
    # ROOT_DIR 指向项目根目录（offline 的父目录）
    parser.add_argument('--ROOT_DIR', type=str, default=project_root)
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='环境配置文件路径（相对于项目根目录）')
    parser.add_argument('--num_episodes', type=int, default=1000, help='收集的episode数量')
    parser.add_argument('--max_steps', type=int, default=250, help='每个episode的最大步数')
    parser.add_argument('--output', type=str, default='carla_offline_dataset_1217_0.05random', help='输出HDF5文件名')
    parser.add_argument('--seed', type=int, default=None, help='随机种子（如果为None则使用配置文件中的seed）')
    
    args = parser.parse_args()
    
    # 处理配置文件路径（相对于项目根目录）
    if not osp.isabs(args.config):
        config_path = osp.join(args.ROOT_DIR, args.config)
    else:
        config_path = args.config
    
    # 加载环境配置
    env_params = load_config(config_path)
    
    # 设置随机种子
    if args.seed is not None:
        env_params['seed'] = args.seed
    set_seed(env_params['seed'])
    
    # 创建环境
    print(">> 创建环境: carla-v0_test1")
    env = gym.make('carla-v0_test1', env_params=env_params)
    
    # 收集数据
    output_file = collect_data(
        env=env,
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        output_file=args.output,
        env_params=env_params,
        config_file_path=config_path
    )
    
    # 关闭环境
    env.close()
    print(">> 完成!")

