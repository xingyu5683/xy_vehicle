"""
可视化 Diffusion_Planner 的输出轨迹与原轨迹对比
包括轨迹对比和加速度分析
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib import font_manager

# 配置中文字体
def setup_chinese_font():
    """配置 matplotlib 使用中文字体"""
    # 尝试使用常见的中文字体
    chinese_fonts = [
        'SimHei',           # 黑体 (Windows)
        'Microsoft YaHei',  # 微软雅黑 (Windows)
        'WenQuanYi Micro Hei',  # 文泉驿微米黑 (Linux)
        'WenQuanYi Zen Hei',    # 文泉驿正黑 (Linux)
        'Noto Sans CJK SC',     # Noto 字体 (Linux)
        'STHeiti',          # 华文黑体 (macOS)
        'Arial Unicode MS', # Arial Unicode (跨平台)
    ]
    
    # 获取系统可用字体
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    
    # 查找第一个可用的中文字体
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            print(f"Using Chinese font: {font_name}")
            return True
    
    # 如果找不到中文字体，使用英文标签
    print("Warning: No Chinese font found, using English labels")
    return False

# 设置字体
USE_CHINESE = setup_chinese_font()

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.normalizer import StateNormalizer, ObservationNormalizer
from diffusion_planner.utils.train_utils import opendata


def compute_acceleration(trajectory):
    """
    计算轨迹的加速度（二阶差分）
    参考 reward 函数中的计算方式
    
    Args:
        trajectory: [T, 4] 或 [T, 2] - 轨迹 (x, y, cos_h, sin_h) 或 (x, y)
    
    Returns:
        accelerations: [T-2, 2] - 加速度向量
        accel_magnitude: [T-2] - 加速度大小
    """
    if trajectory.shape[1] == 4:
        positions = trajectory[:, :2]  # [T, 2] (x, y)
    else:
        positions = trajectory  # [T, 2]
    
    T = positions.shape[0]
    
    if T < 3:
        return None, None
    
    # 位置差分得到速度
    velocities = positions[1:, :] - positions[:-1, :]  # [T-1, 2]
    
    if velocities.shape[0] < 2:
        return None, None
    
    # 速度差分得到加速度
    accelerations = velocities[1:, :] - velocities[:-1, :]  # [T-2, 2]
    accel_magnitude = np.linalg.norm(accelerations, axis=-1)  # [T-2]
    
    return accelerations, accel_magnitude


def load_checkpoint(model, checkpoint_path, device):
    """加载checkpoint到模型"""
    print(f"\n{'='*60}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"{'='*60}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 尝试多种格式
        state_dict = None
        
        # 格式1: QL训练格式（包含'policy_state_dict'）
        if 'policy_state_dict' in checkpoint:
            state_dict = checkpoint['policy_state_dict']
            print("Found 'policy_state_dict' in checkpoint")
        
        # 格式2: 原始diffusion_planner格式（包含'model'键）
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("Found 'model' in checkpoint")
        
        # 格式3: 包含'ema_state_dict'（如果启用EMA）
        elif 'ema_state_dict' in checkpoint:
            state_dict = checkpoint['ema_state_dict']
            print("Found 'ema_state_dict' in checkpoint")
        
        # 格式4: 直接是state_dict
        else:
            if isinstance(checkpoint, dict) and any('encoder' in k or 'decoder' in k for k in checkpoint.keys()):
                state_dict = checkpoint
                print("Checkpoint appears to be a direct state_dict")
        
        if state_dict is None:
            raise ValueError("Could not find valid state_dict in checkpoint")
        
        # 处理DDP格式（移除'module.'前缀）
        if any(k.startswith('module.') for k in state_dict.keys()):
            print("Removing 'module.' prefix from state_dict keys (DDP format)")
            state_dict = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
        
        # 加载state_dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {len(missing_keys)} keys")
            if len(missing_keys) <= 10:
                for key in missing_keys:
                    print(f"  - {key}")
            else:
                for key in missing_keys[:10]:
                    print(f"  - {key}")
                print(f"  ... and {len(missing_keys) - 10} more")
        
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    print(f"  - {key}")
            else:
                for key in unexpected_keys[:10]:
                    print(f"  - {key}")
                print(f"  ... and {len(unexpected_keys) - 10} more")
        
        print(f"✓ Checkpoint loaded successfully!")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise


def visualize_trajectories(original_traj, predicted_traj, accelerations_orig, accelerations_pred, 
                          accel_mag_orig, accel_mag_pred, save_path=None):
    """
    可视化轨迹对比和加速度分析
    
    Args:
        original_traj: [T, 4] - 原始轨迹
        predicted_traj: [T, 4] - 预测轨迹
        accelerations_orig: [T-2, 2] - 原始轨迹加速度
        accelerations_pred: [T-2, 2] - 预测轨迹加速度
        accel_mag_orig: [T-2] - 原始轨迹加速度大小
        accel_mag_pred: [T-2] - 预测轨迹加速度大小
        save_path: 保存路径
    """
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 轨迹对比图（2D平面）
    ax1 = plt.subplot(2, 2, 1)
    orig_pos = original_traj[:, :2]
    pred_pos = predicted_traj[:, :2]
    
    # 根据字体支持情况选择标签
    if USE_CHINESE:
        orig_label, pred_label, start_label, end_label = '原始轨迹', '预测轨迹', '起点', '终点'
        title1 = '轨迹对比 (2D平面)'
    else:
        orig_label, pred_label, start_label, end_label = 'Original Trajectory', 'Predicted Trajectory', 'Start', 'End'
        title1 = 'Trajectory Comparison (2D Plane)'
    
    ax1.plot(orig_pos[:, 0], orig_pos[:, 1], 'b-', linewidth=2, label=orig_label, alpha=0.7)
    ax1.plot(pred_pos[:, 0], pred_pos[:, 1], 'r--', linewidth=2, label=pred_label, alpha=0.7)
    ax1.scatter(orig_pos[0, 0], orig_pos[0, 1], c='green', s=100, marker='o', label=start_label, zorder=5)
    ax1.scatter(orig_pos[-1, 0], orig_pos[-1, 1], c='red', s=100, marker='s', label=end_label, zorder=5)
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title(title1, fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. 加速度大小对比（时间序列）
    ax2 = plt.subplot(2, 2, 2)
    if accel_mag_orig is not None and accel_mag_pred is not None:
        time_steps = np.arange(len(accel_mag_orig))
        if USE_CHINESE:
            orig_accel_label, pred_accel_label = '原始轨迹加速度', '预测轨迹加速度'
            xlabel2, ylabel2, title2 = '时间步', '加速度大小 (m/s²)', '加速度大小对比'
            mean_label = '预测平均值'
        else:
            orig_accel_label, pred_accel_label = 'Original Trajectory Acceleration', 'Predicted Trajectory Acceleration'
            xlabel2, ylabel2, title2 = 'Time Step', 'Acceleration Magnitude (m/s²)', 'Acceleration Magnitude Comparison'
            mean_label = 'Predicted Mean'
        
        ax2.plot(time_steps, accel_mag_orig, 'b-', linewidth=2, label=orig_accel_label, marker='o', markersize=4)
        ax2.plot(time_steps, accel_mag_pred, 'r--', linewidth=2, label=pred_accel_label, marker='s', markersize=4)
        
        # 计算并标注预测轨迹加速度的平均值
        pred_mean = accel_mag_pred.mean()
        ax2.axhline(y=pred_mean, color='red', linestyle=':', linewidth=2, alpha=0.7, label=f'{mean_label}: {pred_mean:.4f} m/s²')
        
        ax2.set_xlabel(xlabel2, fontsize=12)
        ax2.set_ylabel(ylabel2, fontsize=12)
        ax2.set_title(title2, fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    else:
        no_data_text = '加速度数据不足' if USE_CHINESE else 'Insufficient Acceleration Data'
        title2 = '加速度大小对比' if USE_CHINESE else 'Acceleration Magnitude Comparison'
        ax2.text(0.5, 0.5, no_data_text, ha='center', va='center', fontsize=14)
        ax2.set_title(title2, fontsize=14, fontweight='bold')
    
    # 3. 加速度向量场（原始轨迹）
    ax3 = plt.subplot(2, 2, 3)
    if accelerations_orig is not None:
        orig_pos_for_accel = orig_pos[1:-1]  # [T-2, 2] 对应加速度的位置
        traj_label = '轨迹' if USE_CHINESE else 'Trajectory'
        title3 = '原始轨迹加速度向量场' if USE_CHINESE else 'Original Trajectory Acceleration Vector Field'
        ax3.plot(orig_pos[:, 0], orig_pos[:, 1], 'b-', linewidth=1.5, alpha=0.5, label=traj_label)
        # 绘制加速度向量
        scale = 0.1  # 缩放因子，使向量可见
        ax3.quiver(orig_pos_for_accel[:, 0], orig_pos_for_accel[:, 1],
                  accelerations_orig[:, 0] * scale, accelerations_orig[:, 1] * scale,
                  angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.6, width=0.003)
        ax3.set_xlabel('X (m)', fontsize=12)
        ax3.set_ylabel('Y (m)', fontsize=12)
        ax3.set_title(title3, fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
    else:
        no_data_text = '加速度数据不足' if USE_CHINESE else 'Insufficient Acceleration Data'
        title3 = '原始轨迹加速度向量场' if USE_CHINESE else 'Original Trajectory Acceleration Vector Field'
        ax3.text(0.5, 0.5, no_data_text, ha='center', va='center', fontsize=14)
        ax3.set_title(title3, fontsize=14, fontweight='bold')
    
    # 4. 加速度向量场（预测轨迹）
    ax4 = plt.subplot(2, 2, 4)
    if accelerations_pred is not None:
        pred_pos_for_accel = pred_pos[1:-1]  # [T-2, 2] 对应加速度的位置
        traj_label = '轨迹' if USE_CHINESE else 'Trajectory'
        title4 = '预测轨迹加速度向量场' if USE_CHINESE else 'Predicted Trajectory Acceleration Vector Field'
        ax4.plot(pred_pos[:, 0], pred_pos[:, 1], 'r--', linewidth=1.5, alpha=0.5, label=traj_label)
        # 绘制加速度向量
        scale = 0.1  # 缩放因子，使向量可见
        ax4.quiver(pred_pos_for_accel[:, 0], pred_pos_for_accel[:, 1],
                  accelerations_pred[:, 0] * scale, accelerations_pred[:, 1] * scale,
                  angles='xy', scale_units='xy', scale=1, color='red', alpha=0.6, width=0.003)
        ax4.set_xlabel('X (m)', fontsize=12)
        ax4.set_ylabel('Y (m)', fontsize=12)
        ax4.set_title(title4, fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
    else:
        no_data_text = '加速度数据不足' if USE_CHINESE else 'Insufficient Acceleration Data'
        title4 = '预测轨迹加速度向量场' if USE_CHINESE else 'Predicted Trajectory Acceleration Vector Field'
        ax4.text(0.5, 0.5, no_data_text, ha='center', va='center', fontsize=14)
        ax4.set_title(title4, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize Diffusion_Planner trajectory')
    parser.add_argument('--data_dir', type=str, default='/mnt/data/test/test_data', 
                       help='path to npz data directory')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='path to checkpoint file (e.g., training_log/ql_diffusion/2025-12-05-18:14:08/checkpoints/latest.pth)')
    parser.add_argument('--data_idx', type=int, default=0,
                       help='index of npz file to visualize (default: 0)')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--normalization_file', type=str, default='normalization.json',
                       help='normalization file path')
    parser.add_argument('--output_dir', type=str, default='./visualization_output',
                       help='directory to save visualization results')
    parser.add_argument('--save_name', type=str, default=None,
                       help='output filename (default: auto-generated)')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    npz_files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if len(npz_files) == 0:
        raise ValueError(f"No npz files found in {args.data_dir}")
    
    if args.data_idx >= len(npz_files):
        print(f"Warning: data_idx {args.data_idx} >= number of files {len(npz_files)}, using index 0")
        args.data_idx = 0
    
    npz_file = npz_files[args.data_idx]
    print(f"\n{'='*60}")
    print(f"Loading data from: {npz_file}")
    print(f"File index: {args.data_idx}/{len(npz_files)}")
    print(f"{'='*60}\n")
    
    data = opendata(npz_file)
    
    # 准备state输入
    state = {
        'ego_current_state': torch.from_numpy(data['ego_current_state']).float().unsqueeze(0).to(device),
        'neighbor_agents_past': torch.from_numpy(data['neighbor_agents_past']).float().unsqueeze(0).to(device),
        'lanes': torch.from_numpy(data['lanes']).float().unsqueeze(0).to(device),
        'lanes_speed_limit': torch.from_numpy(data['lanes_speed_limit']).float().unsqueeze(0).to(device),
        'lanes_has_speed_limit': torch.from_numpy(data['lanes_has_speed_limit']).bool().unsqueeze(0).to(device),
        'route_lanes': torch.from_numpy(data['route_lanes']).float().unsqueeze(0).to(device),
        'route_lanes_speed_limit': torch.from_numpy(data['route_lanes_speed_limit']).float().unsqueeze(0).to(device),
        'route_lanes_has_speed_limit': torch.from_numpy(data['route_lanes_has_speed_limit']).bool().unsqueeze(0).to(device),
        'static_objects': torch.from_numpy(data['static_objects']).float().unsqueeze(0).to(device),
    }
    
    # 原始轨迹
    ego_future = data['ego_agent_future']  # [T, 3] (x, y, heading)
    heading = ego_future[:, 2]
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    original_trajectory = np.concatenate([ego_future[:, :2], cos_h[:, None], sin_h[:, None]], axis=1)  # [T, 4]
    
    print(f"Original trajectory shape: {original_trajectory.shape}")
    
    # 创建模型配置
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
        
        print(f"Normalizers loaded from {normalization_file}")
    else:
        print(f"Warning: Normalization file not found: {normalization_file}")
        state_normalizer = None
        observation_normalizer = None
    
    # 初始化模型
    print(f"\n{'='*60}")
    print("Initializing Diffusion_Planner model...")
    print(f"{'='*60}\n")
    
    model = Diffusion_Planner(policy_config).to(device)
    model.eval()
    
    # 加载checkpoint
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
    load_checkpoint(model, args.checkpoint_path, device)
    
    # 归一化state
    state_norm = state
    if observation_normalizer is not None:
        state_norm = observation_normalizer(state)
    
    # 运行推理
    print(f"\n{'='*60}")
    print("Running inference...")
    print(f"{'='*60}\n")
    
    with torch.no_grad():
        _, decoder_output = model(state_norm)
        predicted_trajectory = decoder_output['prediction'][0, 0, :, :].cpu().numpy()  # [T, 4]
    
    print(f"Predicted trajectory shape: {predicted_trajectory.shape}")
    
    # 计算加速度
    print("\nComputing accelerations...")
    accel_orig, accel_mag_orig = compute_acceleration(original_trajectory)
    accel_pred, accel_mag_pred = compute_acceleration(predicted_trajectory)
    
    if accel_mag_orig is not None:
        print(f"Original trajectory - Mean acceleration: {accel_mag_orig.mean():.4f} m/s², Max: {accel_mag_orig.max():.4f} m/s²")
    if accel_mag_pred is not None:
        print(f"Predicted trajectory - Mean acceleration: {accel_mag_pred.mean():.4f} m/s², Max: {accel_mag_pred.max():.4f} m/s²")
    
    # 可视化
    if args.save_name is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 从 checkpoint_path 中提取信息用于文件名
        checkpoint_info = ""
        if args.checkpoint_path:
            # 获取 checkpoint 文件名（不含路径和扩展名）
            checkpoint_basename = os.path.basename(args.checkpoint_path)
            checkpoint_name = os.path.splitext(checkpoint_basename)[0]
            
            # 清理文件名，移除特殊字符，只保留字母、数字、下划线和连字符
            import re
            checkpoint_name_clean = re.sub(r'[^a-zA-Z0-9_-]', '_', checkpoint_name)
            
            # 如果文件名太长，截断
            if len(checkpoint_name_clean) > 50:
                checkpoint_name_clean = checkpoint_name_clean[:50]
            
            checkpoint_info = f"_{checkpoint_name_clean}"
        
        filename = f"trajectory_vis_{args.data_idx}_{timestamp}{checkpoint_info}.png"
    else:
        filename = args.save_name
    
    save_path = os.path.join(args.output_dir, filename)
    
    print(f"\n{'='*60}")
    print("Generating visualization...")
    print(f"{'='*60}\n")
    
    visualize_trajectories(
        original_trajectory,
        predicted_trajectory,
        accel_orig,
        accel_pred,
        accel_mag_orig,
        accel_mag_pred,
        save_path=save_path
    )
    
    print(f"\n{'='*60}")
    print("Visualization completed!")
    print(f"{'='*60}")
    print(f"Output saved to: {save_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

