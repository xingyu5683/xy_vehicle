"""
诊断Q值和reward相关性差的问题

检查：
1. Reward的分布和范围
2. Q值的分布和范围
3. Target Q值的计算是否正确
4. Critic loss的变化趋势
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_ql_diffusion import (
    NPZDataset,
    extract_state_features,
    compute_reward,
    collate_fn
)
from torch.utils.data import DataLoader


def analyze_reward_distribution(data_dir, num_samples=1000):
    """分析reward的分布"""
    print("="*60)
    print("分析Reward分布")
    print("="*60)
    
    dataset = NPZDataset(data_dir, device="cpu")
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    rewards = []
    
    for i, (state, action, reward, next_state, not_done) in enumerate(data_loader):
        if i * 32 >= num_samples:
            break
        
        # 提取状态特征
        state_features = extract_state_features(state, use_all_features=False)
        action_flat = action.reshape(action.shape[0], -1)
        
        # 计算reward
        computed_reward = compute_reward(state_features, action_flat)
        
        rewards.extend(computed_reward.numpy().tolist())
    
    rewards = np.array(rewards)
    
    print(f"\n样本数量: {len(rewards)}")
    print(f"Reward统计:")
    print(f"  均值: {rewards.mean():.6f}")
    print(f"  标准差: {rewards.std():.6f}")
    print(f"  最小值: {rewards.min():.6f}")
    print(f"  最大值: {rewards.max():.6f}")
    print(f"  中位数: {np.median(rewards):.6f}")
    print(f"  25%分位数: {np.percentile(rewards, 25):.6f}")
    print(f"  75%分位数: {np.percentile(rewards, 75):.6f}")
    
    # 绘制分布图
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title(f'Reward Distribution (mean={rewards.mean():.6f}, std={rewards.std():.6f})')
    plt.grid(True, alpha=0.3)
    plt.savefig('reward_distribution.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ 分布图已保存: reward_distribution.png")
    
    return rewards


def check_bellman_equation(checkpoint_path, data_dir, num_samples=100):
    """检查Bellman方程的计算"""
    print("\n" + "="*60)
    print("检查Bellman方程计算")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    from validate_q_reward import load_trained_model
    agent, dataset = load_trained_model(checkpoint_path, data_dir, device)
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    bellman_errors = []
    q_value_ranges = []
    reward_ranges = []
    
    agent.critic.eval()
    agent.critic_target.eval()
    
    with torch.no_grad():
        for i, (state, action, reward, next_state, not_done) in enumerate(data_loader):
            if i >= num_samples:
                break
            
            # 移动到设备
            state = {k: v.to(device) for k, v in state.items()}
            next_state = {k: v.to(device) for k, v in next_state.items()}
            action = action.to(device)
            reward = reward.to(device)
            not_done = not_done.to(device)
            
            B = action.shape[0]
            action_flat = action.reshape(B, -1)
            
            # 提取状态特征
            state_features = extract_state_features(state, use_all_features=False)
            next_state_features = extract_state_features(next_state, use_all_features=False)
            
            # 当前Q值
            current_q1, current_q2 = agent.critic(state_features, action_flat)
            current_q = torch.min(current_q1, current_q2)
            
            # 计算target Q值
            next_state_norm = next_state
            if agent.observation_normalizer is not None:
                next_state_norm = agent.observation_normalizer(next_state)
            
            agent.policy_target.eval()
            _, decoder_output = agent.policy_target(next_state_norm)
            next_action = decoder_output['prediction'][:, 0, :, :]
            next_action_flat = next_action.reshape(B, -1)
            
            target_q1, target_q2 = agent.critic_target(next_state_features, next_action_flat)
            target_q_next = torch.min(target_q1, target_q2)
            
            # Bellman目标
            reward_expanded = reward.unsqueeze(-1)
            not_done_expanded = not_done.unsqueeze(-1)
            target_q = reward_expanded + not_done_expanded * agent.discount * target_q_next
            
            # 计算Bellman误差
            bellman_error = (current_q - target_q).abs().mean().item()
            bellman_errors.append(bellman_error)
            
            # 记录范围
            q_value_ranges.append({
                'current_q': current_q.item() if current_q.numel() == 1 else current_q.mean().item(),
                'target_q': target_q.item() if target_q.numel() == 1 else target_q.mean().item(),
                'target_q_next': target_q_next.item() if target_q_next.numel() == 1 else target_q_next.mean().item(),
                'reward': reward.item() if reward.numel() == 1 else reward.mean().item(),
            })
            reward_ranges.append(reward.item() if reward.numel() == 1 else reward.mean().item())
    
    bellman_errors = np.array(bellman_errors)
    current_q_values = np.array([r['current_q'] for r in q_value_ranges])
    target_q_values = np.array([r['target_q'] for r in q_value_ranges])
    target_q_next_values = np.array([r['target_q_next'] for r in q_value_ranges])
    reward_ranges = np.array(reward_ranges)
    
    print(f"\n样本数量: {len(bellman_errors)}")
    print(f"\nBellman误差统计:")
    print(f"  均值: {bellman_errors.mean():.6f}")
    print(f"  标准差: {bellman_errors.std():.6f}")
    print(f"  最大值: {bellman_errors.max():.6f}")
    
    print(f"\nQ值范围:")
    print(f"  当前Q值 - 均值: {current_q_values.mean():.6f}, 标准差: {current_q_values.std():.6f}")
    print(f"  目标Q值 - 均值: {target_q_values.mean():.6f}, 标准差: {target_q_values.std():.6f}")
    print(f"  下一状态Q值 - 均值: {target_q_next_values.mean():.6f}, 标准差: {target_q_next_values.std():.6f}")
    
    print(f"\nReward范围:")
    print(f"  均值: {reward_ranges.mean():.6f}, 标准差: {reward_ranges.std():.6f}")
    print(f"  最小值: {reward_ranges.min():.6f}, 最大值: {reward_ranges.max():.6f}")
    
    # 检查尺度匹配
    q_std = current_q_values.std()
    reward_std = reward_ranges.std()
    scale_ratio = q_std / reward_std if reward_std > 0 else float('inf')
    
    print(f"\n尺度分析:")
    print(f"  Q值标准差 / Reward标准差 = {scale_ratio:.2f}")
    if scale_ratio > 10:
        print(f"  ⚠ 警告: Q值标准差远大于Reward标准差，可能存在尺度不匹配问题")
    elif scale_ratio < 0.1:
        print(f"  ⚠ 警告: Q值标准差远小于Reward标准差，可能存在尺度不匹配问题")
    else:
        print(f"  ✓ 尺度匹配良好")
    
    return bellman_errors, current_q_values, reward_ranges


def main():
    parser = argparse.ArgumentParser(description='诊断Q值和reward相关性差的问题')
    parser.add_argument('--checkpoint_path', type=str, default='./training_log/ql_diffusion/2025-12-10-10:57:39/checkpoints/latest.pth', help='checkpoint路径')
    parser.add_argument('--data_dir', type=str, default='/mnt/data/test/test_data', help='数据目录')
    parser.add_argument('--num_samples', type=int, default=1000, help='分析样本数量')
    
    args = parser.parse_args()
    
    # 1. 分析reward分布
    rewards = analyze_reward_distribution(args.data_dir, args.num_samples)
    
    # 2. 检查Bellman方程
    bellman_errors, q_values, reward_values = check_bellman_equation(
        args.checkpoint_path, args.data_dir, num_samples=min(100, args.num_samples)
    )
    
    # q_values是current_q_values
    q_values = q_values  # 保持变量名一致
    
    print("\n" + "="*60)
    print("诊断总结")
    print("="*60)
    
    # 总结
    if bellman_errors.mean() > 0.1:
        print("⚠ Bellman误差较大，Critic可能训练不充分")
    else:
        print("✓ Bellman误差较小，Critic训练良好")
    
    if rewards.std() < 0.01:
        print("⚠ Reward标准差很小，可能导致Q值学习困难")
        print("  建议: 考虑扩展reward函数，增加reward的变化范围")
    
    q_std = q_values.std()
    reward_std = reward_values.std()
    if q_std / reward_std > 10:
        print("⚠ Q值和Reward的尺度不匹配")
        print("  建议: 检查Q值的初始化或归一化方式")


if __name__ == '__main__':
    main()

