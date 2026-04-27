"""
验证训练好的Q网络输出是否能反映reward的大小

功能：
1. 加载训练好的模型（包括Critic网络）
2. 从数据集中采样多个样本
3. 计算每个样本的Q值和reward
4. 分析Q值和reward的相关性
5. 可视化Q值和reward的关系
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# 尝试导入scipy，如果失败则使用numpy计算相关性
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠ 警告: scipy未安装，将使用numpy计算相关性", flush=True)

# 设置无缓冲输出（在导入train_ql_diffusion之前不设置，避免冲突）
# sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
# sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

print("正在导入模块...", flush=True)

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("正在从train_ql_diffusion导入...", flush=True)
# train_ql_diffusion会自己设置sys.stdout和sys.stderr，所以先导入它
try:
    from train_ql_diffusion import (
        NPZDataset, 
        extract_state_features, 
        compute_reward,
        Critic,
        QL_Diffusion
    )
    print("✓ train_ql_diffusion导入成功", flush=True)
except Exception as e:
    print(f"✗ train_ql_diffusion导入失败: {e}", flush=True)
    import traceback
    traceback.print_exc()
    raise

print("正在从diffusion_planner导入...", flush=True)
try:
    from diffusion_planner.utils.train_utils import opendata
    print("✓ diffusion_planner导入成功", flush=True)
except Exception as e:
    print(f"✗ diffusion_planner导入失败: {e}", flush=True)
    import traceback
    traceback.print_exc()
    raise

print("模块导入完成", flush=True)

# 现在可以安全地设置无缓冲输出（train_ql_diffusion已经设置过了）
# 如果需要，可以在这里再次设置


def setup_chinese_font():
    """配置中文字体"""
    try:
        # 使用非交互式后端，避免卡住
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        
        # 尝试使用中文字体
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'STHeiti']
        for font_name in chinese_fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                # 测试字体是否可用
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.text(0.5, 0.5, '测试', fontsize=12)
                plt.close(fig)
                print(f"✓ 使用中文字体: {font_name}", flush=True)
                return True
            except Exception as e:
                continue
        print("⚠ 未找到中文字体，使用英文标签", flush=True)
        return False
    except Exception as e:
        print(f"⚠ 字体配置失败，使用英文标签: {e}", flush=True)
        return False


def load_trained_model(checkpoint_path: str, data_dir: str, device: str = "cuda"):
    """加载训练好的模型"""
    print(f"\n{'='*60}", flush=True)
    print(f"加载checkpoint: {checkpoint_path}", flush=True)
    print(f"{'='*60}", flush=True)
    
    # 检查checkpoint文件是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")
    
    print(f"正在加载checkpoint文件 (大小: {os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB)...", flush=True)
    
    # 加载checkpoint
    try:
        # PyTorch 2.6默认weights_only=True，但checkpoint可能包含numpy对象，需要设置为False
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print("✓ Checkpoint加载成功", flush=True)
    except Exception as e:
        raise RuntimeError(f"加载checkpoint失败: {e}")
    
    # 获取配置信息
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}", flush=True)
    print(f"Checkpoint step: {checkpoint.get('step', 'unknown')}", flush=True)
    
    # 创建数据集以获取维度信息
    print(f"正在加载数据集: {data_dir}...", flush=True)
    try:
        dataset = NPZDataset(data_dir, device=device)
        print(f"✓ 数据集加载成功，共 {len(dataset)} 个样本", flush=True)
    except Exception as e:
        raise RuntimeError(f"加载数据集失败: {e}")
    
    print("正在获取样本维度信息...", flush=True)
    try:
        sample_state, sample_action, _, _, _ = dataset[0]
        sample_state = {k: v.unsqueeze(0).to(device) for k, v in sample_state.items()}
        print("✓ 样本维度信息获取成功", flush=True)
    except Exception as e:
        raise RuntimeError(f"获取样本维度信息失败: {e}")
    
    # 获取状态特征维度
    USE_ALL_FEATURES = False
    state_feature_dim = extract_state_features(sample_state, use_all_features=USE_ALL_FEATURES).shape[1]
    action_dim = sample_action.shape[0] * sample_action.shape[1]  # T * 4
    
    print(f"State feature dimension: {state_feature_dim}", flush=True)
    print(f"Action dimension: {action_dim}", flush=True)
    
    # 创建PolicyConfig
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
    
    # 加载归一化器（如果存在）
    normalization_file = os.path.join(os.path.dirname(__file__), 'normalization.json')
    if os.path.exists(normalization_file):
        from diffusion_planner.utils.normalizer import StateNormalizer, ObservationNormalizer
        
        class NormalizerArgs:
            def __init__(self, normalization_file_path, predicted_neighbor_num):
                self.normalization_file_path = normalization_file_path
                self.predicted_neighbor_num = predicted_neighbor_num
        
        norm_args = NormalizerArgs(normalization_file, policy_config.predicted_neighbor_num)
        state_normalizer = StateNormalizer.from_json(norm_args)
        observation_normalizer = ObservationNormalizer.from_json(norm_args)
        
        policy_config.state_normalizer = state_normalizer
        policy_config.observation_normalizer = observation_normalizer
        
        print(f"✓ 归一化器已加载", flush=True)
    else:
        state_normalizer = None
        observation_normalizer = None
        print(f"⚠ 归一化文件未找到，不使用归一化", flush=True)
    
    # 创建agent（但不训练）
    agent = QL_Diffusion(
        policy_config=policy_config,
        state_feature_dim=state_feature_dim,
        action_dim=action_dim,
        device=device,
        discount=0.99,
        tau=0.005,
        lr_policy=5e-4,
        lr_critic=3e-4,
        grad_norm=1.0,
        state_normalizer=state_normalizer,
        observation_normalizer=observation_normalizer,
        resume_checkpoint=None,
        eta=1.0,
    )
    
    # 加载checkpoint中的权重
    if 'critic_state_dict' in checkpoint:
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        print("✓ Critic权重已加载", flush=True)
    else:
        print("⚠ Checkpoint中未找到critic_state_dict", flush=True)
    
    if 'policy_state_dict' in checkpoint:
        agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        print("✓ Policy权重已加载", flush=True)
    
    agent.critic.eval()
    agent.policy.eval()
    
    print(f"{'='*60}\n", flush=True)
    
    return agent, dataset


def analyze_q_reward_consistency(agent, dataset, num_samples: int = 1000, device: str = "cuda"):
    """分析Q值和reward的一致性"""
    print(f"\n{'='*60}", flush=True)
    print(f"开始分析Q值和reward的一致性", flush=True)
    print(f"采样数量: {num_samples}", flush=True)
    print(f"{'='*60}", flush=True)
    
    q1_values = []
    q2_values = []
    q_min_values = []
    reward_values = []
    
    # 随机采样
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="处理样本"):
            try:
                state, action, reward, next_state, not_done = dataset[idx]
                
                # 转换为batch格式
                state_batch = {k: v.unsqueeze(0).to(device) for k, v in state.items()}
                action_batch = action.unsqueeze(0).to(device)  # [1, T, 4]
                
                # 提取状态特征
                state_features = extract_state_features(state_batch, use_all_features=False)  # [1, state_feature_dim]
                
                # Flatten action
                action_flat = action_batch.reshape(1, -1)  # [1, T*4]
                
                # 计算Q值
                q1, q2 = agent.critic(state_features, action_flat)  # [1, 1]
                q1_val = q1.item()
                q2_val = q2.item()
                q_min_val = min(q1_val, q2_val)
                
                # 计算reward
                reward_val = compute_reward(state_features, action_flat).item()
                
                q1_values.append(q1_val)
                q2_values.append(q2_val)
                q_min_values.append(q_min_val)
                reward_values.append(reward_val)
                
            except Exception as e:
                print(f"⚠ 处理样本 {idx} 时出错: {e}", flush=True)
                continue
    
    q1_values = np.array(q1_values)
    q2_values = np.array(q2_values)
    q_min_values = np.array(q_min_values)
    reward_values = np.array(reward_values)
    
    print(f"\n有效样本数: {len(q1_values)}", flush=True)
    
    # 统计分析
    print(f"\n{'='*60}", flush=True)
    print(f"统计分析结果", flush=True)
    print(f"{'='*60}", flush=True)
    
    print(f"\nQ1值统计:", flush=True)
    print(f"  均值: {q1_values.mean():.6f}", flush=True)
    print(f"  标准差: {q1_values.std():.6f}", flush=True)
    print(f"  最小值: {q1_values.min():.6f}", flush=True)
    print(f"  最大值: {q1_values.max():.6f}", flush=True)
    
    print(f"\nQ2值统计:", flush=True)
    print(f"  均值: {q2_values.mean():.6f}", flush=True)
    print(f"  标准差: {q2_values.std():.6f}", flush=True)
    print(f"  最小值: {q2_values.min():.6f}", flush=True)
    print(f"  最大值: {q2_values.max():.6f}", flush=True)
    
    print(f"\nQ_min值统计:", flush=True)
    print(f"  均值: {q_min_values.mean():.6f}", flush=True)
    print(f"  标准差: {q_min_values.std():.6f}", flush=True)
    print(f"  最小值: {q_min_values.min():.6f}", flush=True)
    print(f"  最大值: {q_min_values.max():.6f}", flush=True)
    
    print(f"\nReward值统计:", flush=True)
    print(f"  均值: {reward_values.mean():.6f}", flush=True)
    print(f"  标准差: {reward_values.std():.6f}", flush=True)
    print(f"  最小值: {reward_values.min():.6f}", flush=True)
    print(f"  最大值: {reward_values.max():.6f}", flush=True)
    
    # 计算相关性
    print(f"\n{'='*60}", flush=True)
    print(f"相关性分析", flush=True)
    print(f"{'='*60}", flush=True)
    
    # 计算Pearson相关系数的辅助函数
    def pearson_corr(x, y):
        """计算Pearson相关系数"""
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
        if denominator == 0:
            return 0.0, 1.0
        corr = numerator / denominator
        # 简单的p-value估计（对于大样本，p-value通常很小）
        n = len(x)
        if n > 2 and abs(corr) < 0.999:
            t_stat = corr * np.sqrt((n - 2) / (1 - corr ** 2))
            # 使用t分布近似（简化版本，p-value估计）
            p_value = 2 * (1 - min(0.999, abs(t_stat) / 10))  # 简化估计
        else:
            p_value = 1.0 if abs(corr) < 0.001 else 0.0
        return corr, p_value
    
    # Q1和reward的相关性
    if HAS_SCIPY:
        corr_q1_reward, p_value_q1 = stats.pearsonr(q1_values, reward_values)
    else:
        corr_q1_reward, p_value_q1 = pearson_corr(q1_values, reward_values)
    print(f"\nQ1与Reward的相关系数: {corr_q1_reward:.6f} (p-value: {p_value_q1:.6e})", flush=True)
    
    # Q2和reward的相关性
    if HAS_SCIPY:
        corr_q2_reward, p_value_q2 = stats.pearsonr(q2_values, reward_values)
    else:
        corr_q2_reward, p_value_q2 = pearson_corr(q2_values, reward_values)
    print(f"Q2与Reward的相关系数: {corr_q2_reward:.6f} (p-value: {p_value_q2:.6e})", flush=True)
    
    # Q_min和reward的相关性
    if HAS_SCIPY:
        corr_qmin_reward, p_value_qmin = stats.pearsonr(q_min_values, reward_values)
    else:
        corr_qmin_reward, p_value_qmin = pearson_corr(q_min_values, reward_values)
    print(f"Q_min与Reward的相关系数: {corr_qmin_reward:.6f} (p-value: {p_value_qmin:.6e})", flush=True)
    
    # 方向一致性（Q值增加时，reward是否也增加）
    q1_increase = q1_values[1:] > q1_values[:-1]
    reward_increase = reward_values[1:] > reward_values[:-1]
    direction_consistency_q1 = (q1_increase == reward_increase).mean()
    print(f"\nQ1与Reward方向一致性: {direction_consistency_q1:.4f} ({direction_consistency_q1*100:.2f}%)", flush=True)
    
    q2_increase = q2_values[1:] > q2_values[:-1]
    direction_consistency_q2 = (q2_increase == reward_increase).mean()
    print(f"Q2与Reward方向一致性: {direction_consistency_q2:.4f} ({direction_consistency_q2*100:.2f}%)", flush=True)
    
    qmin_increase = q_min_values[1:] > q_min_values[:-1]
    direction_consistency_qmin = (qmin_increase == reward_increase).mean()
    print(f"Q_min与Reward方向一致性: {direction_consistency_qmin:.4f} ({direction_consistency_qmin*100:.2f}%)", flush=True)
    
    # 评估结果
    print(f"\n{'='*60}", flush=True)
    print(f"评估结果", flush=True)
    print(f"{'='*60}", flush=True)
    
    if abs(corr_q1_reward) > 0.5:
        print(f"✓ Q1与Reward相关性较强 (|r|={abs(corr_q1_reward):.4f})", flush=True)
    elif abs(corr_q1_reward) > 0.3:
        print(f"⚠ Q1与Reward相关性中等 (|r|={abs(corr_q1_reward):.4f})", flush=True)
    else:
        print(f"✗ Q1与Reward相关性较弱 (|r|={abs(corr_q1_reward):.4f})", flush=True)
    
    if abs(corr_q2_reward) > 0.5:
        print(f"✓ Q2与Reward相关性较强 (|r|={abs(corr_q2_reward):.4f})", flush=True)
    elif abs(corr_q2_reward) > 0.3:
        print(f"⚠ Q2与Reward相关性中等 (|r|={abs(corr_q2_reward):.4f})", flush=True)
    else:
        print(f"✗ Q2与Reward相关性较弱 (|r|={abs(corr_q2_reward):.4f})", flush=True)
    
    if abs(corr_qmin_reward) > 0.5:
        print(f"✓ Q_min与Reward相关性较强 (|r|={abs(corr_qmin_reward):.4f})", flush=True)
    elif abs(corr_qmin_reward) > 0.3:
        print(f"⚠ Q_min与Reward相关性中等 (|r|={abs(corr_qmin_reward):.4f})", flush=True)
    else:
        print(f"✗ Q_min与Reward相关性较弱 (|r|={abs(corr_qmin_reward):.4f})", flush=True)
    
    return {
        'q1_values': q1_values,
        'q2_values': q2_values,
        'q_min_values': q_min_values,
        'reward_values': reward_values,
        'corr_q1_reward': corr_q1_reward,
        'corr_q2_reward': corr_q2_reward,
        'corr_qmin_reward': corr_qmin_reward,
        'direction_consistency_q1': direction_consistency_q1,
        'direction_consistency_q2': direction_consistency_q2,
        'direction_consistency_qmin': direction_consistency_qmin,
    }


def visualize_q_reward(results: dict, output_dir: str, use_chinese: bool = True):
    """可视化Q值和reward的关系"""
    print(f"\n{'='*60}", flush=True)
    print(f"生成可视化图表", flush=True)
    print(f"{'='*60}", flush=True)
    
    os.makedirs(output_dir, exist_ok=True)
    
    q1_values = results['q1_values']
    q2_values = results['q2_values']
    q_min_values = results['q_min_values']
    reward_values = results['reward_values']
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Q1 vs Reward散点图
    ax = axes[0, 0]
    ax.scatter(reward_values, q1_values, alpha=0.5, s=10)
    ax.set_xlabel('Reward' if not use_chinese else 'Reward (奖励)', fontsize=12)
    ax.set_ylabel('Q1 Value' if not use_chinese else 'Q1值', fontsize=12)
    ax.set_title(f'Q1 vs Reward (r={results["corr_q1_reward"]:.4f})' if not use_chinese 
                 else f'Q1 vs Reward (相关系数={results["corr_q1_reward"]:.4f})', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 添加趋势线
    z = np.polyfit(reward_values, q1_values, 1)
    p = np.poly1d(z)
    ax.plot(reward_values, p(reward_values), "r--", alpha=0.8, linewidth=2, label='Trend Line')
    ax.legend()
    
    # 2. Q2 vs Reward散点图
    ax = axes[0, 1]
    ax.scatter(reward_values, q2_values, alpha=0.5, s=10, color='green')
    ax.set_xlabel('Reward' if not use_chinese else 'Reward (奖励)', fontsize=12)
    ax.set_ylabel('Q2 Value' if not use_chinese else 'Q2值', fontsize=12)
    ax.set_title(f'Q2 vs Reward (r={results["corr_q2_reward"]:.4f})' if not use_chinese 
                 else f'Q2 vs Reward (相关系数={results["corr_q2_reward"]:.4f})', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 添加趋势线
    z = np.polyfit(reward_values, q2_values, 1)
    p = np.poly1d(z)
    ax.plot(reward_values, p(reward_values), "r--", alpha=0.8, linewidth=2, label='Trend Line')
    ax.legend()
    
    # 3. Q_min vs Reward散点图
    ax = axes[1, 0]
    ax.scatter(reward_values, q_min_values, alpha=0.5, s=10, color='orange')
    ax.set_xlabel('Reward' if not use_chinese else 'Reward (奖励)', fontsize=12)
    ax.set_ylabel('Q_min Value' if not use_chinese else 'Q_min值', fontsize=12)
    ax.set_title(f'Q_min vs Reward (r={results["corr_qmin_reward"]:.4f})' if not use_chinese 
                 else f'Q_min vs Reward (相关系数={results["corr_qmin_reward"]:.4f})', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 添加趋势线
    z = np.polyfit(reward_values, q_min_values, 1)
    p = np.poly1d(z)
    ax.plot(reward_values, p(reward_values), "r--", alpha=0.8, linewidth=2, label='Trend Line')
    ax.legend()
    
    # 4. 分布对比
    ax = axes[1, 1]
    ax.hist(reward_values, bins=50, alpha=0.5, label='Reward' if not use_chinese else 'Reward分布', density=True)
    ax2 = ax.twinx()
    ax2.hist(q_min_values, bins=50, alpha=0.5, color='orange', label='Q_min' if not use_chinese else 'Q_min分布', density=True)
    ax.set_xlabel('Value' if not use_chinese else '数值', fontsize=12)
    ax.set_ylabel('Reward Density' if not use_chinese else 'Reward密度', fontsize=12)
    ax2.set_ylabel('Q_min Density' if not use_chinese else 'Q_min密度', fontsize=12)
    ax.set_title('Distribution Comparison' if not use_chinese else '分布对比', fontsize=14)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'q_reward_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存: {output_path}", flush=True)
    plt.close()
    
    # 保存统计结果到文本文件
    report_path = os.path.join(output_dir, 'q_reward_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Q值与Reward一致性分析报告\n")
        f.write("="*60 + "\n\n")
        
        f.write("统计分析:\n")
        f.write(f"  样本数量: {len(q1_values)}\n")
        f.write(f"  Q1均值: {q1_values.mean():.6f}, 标准差: {q1_values.std():.6f}\n")
        f.write(f"  Q2均值: {q2_values.mean():.6f}, 标准差: {q2_values.std():.6f}\n")
        f.write(f"  Q_min均值: {q_min_values.mean():.6f}, 标准差: {q_min_values.std():.6f}\n")
        f.write(f"  Reward均值: {reward_values.mean():.6f}, 标准差: {reward_values.std():.6f}\n\n")
        
        f.write("相关性分析:\n")
        f.write(f"  Q1与Reward相关系数: {results['corr_q1_reward']:.6f}\n")
        f.write(f"  Q2与Reward相关系数: {results['corr_q2_reward']:.6f}\n")
        f.write(f"  Q_min与Reward相关系数: {results['corr_qmin_reward']:.6f}\n\n")
        
        f.write("方向一致性:\n")
        f.write(f"  Q1与Reward方向一致性: {results['direction_consistency_q1']:.4f} ({results['direction_consistency_q1']*100:.2f}%)\n")
        f.write(f"  Q2与Reward方向一致性: {results['direction_consistency_q2']:.4f} ({results['direction_consistency_q2']*100:.2f}%)\n")
        f.write(f"  Q_min与Reward方向一致性: {results['direction_consistency_qmin']:.4f} ({results['direction_consistency_qmin']*100:.2f}%)\n")
    
    print(f"✓ 报告已保存: {report_path}", flush=True)


def main():
    print("程序开始运行...", flush=True)
    
    parser = argparse.ArgumentParser(description='验证Q网络输出是否能反映reward大小')
    parser.add_argument('--checkpoint_path', type=str, required=True, default='./training_log/ql_diffusion/2025-12-10-10:57:39/checkpoints/latest.pth', help='训练好的checkpoint路径')
    parser.add_argument('--data_dir', type=str, default='/mnt/data/test/test_data', help='数据目录')
    parser.add_argument('--num_samples', type=int, default=100, help='采样数量')
    parser.add_argument('--output_dir', type=str, default='./q_reward_validation', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    args = parser.parse_args()
    print(f"参数解析完成: checkpoint={args.checkpoint_path}, data_dir={args.data_dir}, num_samples={args.num_samples}", flush=True)
    
    # 配置中文字体
    print("正在配置中文字体...", flush=True)
    use_chinese = setup_chinese_font()
    print(f"字体配置完成: use_chinese={use_chinese}", flush=True)
    
    # 加载模型
    print("开始加载模型...", flush=True)
    agent, dataset = load_trained_model(args.checkpoint_path, args.data_dir, args.device)
    print("模型加载完成", flush=True)
    
    # 分析Q值和reward的一致性
    print("开始分析Q值和reward的一致性...", flush=True)
    results = analyze_q_reward_consistency(agent, dataset, args.num_samples, args.device)
    print("分析完成", flush=True)
    
    # 可视化
    print("开始生成可视化图表...", flush=True)
    output_dir = os.path.abspath(args.output_dir)
    visualize_q_reward(results, output_dir, use_chinese)
    print("可视化完成", flush=True)
    
    print(f"\n{'='*60}", flush=True)
    print(f"分析完成！结果保存在: {output_dir}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断", flush=True)
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n程序运行出错: {e}", flush=True)
        print("错误详情:", flush=True)
        traceback.print_exc()
        sys.exit(1)

