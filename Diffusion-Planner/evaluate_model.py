"""
评估模型脚本
参考 visualize_trajectory.py 的加载和使用模型逻辑
从固定的100个场景计算平均reward
"""

import os
import sys
import glob
import argparse
import hashlib
import numpy as np
import torch

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.normalizer import StateNormalizer, ObservationNormalizer
from diffusion_planner.utils.train_utils import opendata
from train_ql_diffusion import compute_reward, extract_state_features


def load_checkpoint(model, checkpoint_path, device):
    """加载checkpoint到模型（参考 visualize_trajectory.py）"""
    print(f"\n{'='*60}", flush=True)
    print(f"Loading checkpoint from: {checkpoint_path}", flush=True)
    print(f"{'='*60}", flush=True)
    
    try:
        # PyTorch 2.6+需要设置weights_only=False
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 尝试多种格式
        state_dict = None
        
        # 格式1: QL训练格式（包含'policy_state_dict'）
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
            if isinstance(checkpoint, dict) and any('encoder' in k or 'decoder' in k for k in checkpoint.keys()):
                state_dict = checkpoint
                print("Checkpoint appears to be a direct state_dict", flush=True)
        
        if state_dict is None:
            raise ValueError("Could not find valid state_dict in checkpoint")
        
        # 处理DDP格式（移除'module.'前缀）
        if any(k.startswith('module.') for k in state_dict.keys()):
            print("Removing 'module.' prefix from state_dict keys (DDP format)", flush=True)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[len('module.'):]] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        
        # 验证checkpoint中的权重hash（用于调试）
        if len(state_dict) > 0:
            first_key_checkpoint = list(state_dict.keys())[0]
            first_weight_checkpoint = state_dict[first_key_checkpoint]
            checkpoint_hash = hashlib.md5(first_weight_checkpoint.cpu().numpy().tobytes()).hexdigest()[:8]
            print(f"Checkpoint中第一个参数的hash: {checkpoint_hash}", flush=True)
        
        # 加载state_dict
        print(f"State dict包含 {len(state_dict)} 个参数", flush=True)
        print(f"模型需要 {len(model.state_dict())} 个参数", flush=True)
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        # 统计成功加载的参数数量
        model_keys = set(model.state_dict().keys())
        state_dict_keys = set(state_dict.keys())
        matched_keys = model_keys & state_dict_keys
        print(f"成功匹配 {len(matched_keys)}/{len(model_keys)} 个参数", flush=True)
        
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {len(missing_keys)} keys", flush=True)
            if len(missing_keys) <= 10:
                for key in missing_keys:
                    print(f"  - {key}", flush=True)
            else:
                for key in missing_keys[:10]:
                    print(f"  - {key}", flush=True)
                print(f"  ... and {len(missing_keys) - 10} more", flush=True)
        else:
            print("✓ 所有模型参数都已从checkpoint加载", flush=True)
        
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys", flush=True)
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    print(f"  - {key}", flush=True)
            else:
                for key in unexpected_keys[:10]:
                    print(f"  - {key}", flush=True)
                print(f"  ... and {len(unexpected_keys) - 10} more", flush=True)
        
        print(f"✓ Checkpoint loaded successfully!", flush=True)
        print(f"{'='*60}\n", flush=True)
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}", flush=True)
        raise


def evaluate_model(checkpoint_path: str, data_dir: str = "/mnt/datanpz0.01", 
                  num_scenarios: int = 100, device: str = "cuda", 
                  fixed_seed: int = 42, normalization_file: str = "normalization.json"):
    """
    评估模型
    
    Args:
        checkpoint_path: checkpoint文件路径
        data_dir: 数据目录
        num_scenarios: 评估的场景数量（固定100个）
        device: 设备类型
        fixed_seed: 固定随机种子，确保每次选择相同的场景
        normalization_file: 归一化文件路径
    """
    # 设置随机种子，确保每次选择相同的场景
    np.random.seed(fixed_seed)
    torch.manual_seed(fixed_seed)
    
    print(f"\n{'='*60}", flush=True)
    print(f"模型评估", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Checkpoint路径: {checkpoint_path}", flush=True)
    print(f"数据目录: {data_dir}", flush=True)
    print(f"评估场景数: {num_scenarios}", flush=True)
    print(f"设备: {device}", flush=True)
    print(f"固定随机种子: {fixed_seed}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # 检查checkpoint文件是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    # 加载数据文件列表
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if len(npz_files) == 0:
        raise ValueError(f"数据目录中没有找到npz文件: {data_dir}")
    
    total_scenarios = len(npz_files)
    print(f"数据集总场景数: {total_scenarios}", flush=True)
    
    if total_scenarios < num_scenarios:
        print(f"警告: 数据集只有 {total_scenarios} 个场景，少于请求的 {num_scenarios} 个", flush=True)
        num_scenarios = total_scenarios
    
    # 固定选择场景索引（使用固定随机种子确保可重复）
    selected_indices = np.random.choice(total_scenarios, size=num_scenarios, replace=False)
    selected_indices = np.sort(selected_indices)  # 排序以便于查看
    
    print(f"\n选择的场景索引（前10个）: {selected_indices[:10]}", flush=True)
    print(f"选择的场景索引（后10个）: {selected_indices[-10:]}", flush=True)
    print(f"", flush=True)
    
    # 创建模型配置（参考 visualize_trajectory.py）
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
    
    # 创建归一化器（参考 visualize_trajectory.py）
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
        
        print(f"归一化器已加载: {normalization_file}", flush=True)
    else:
        print(f"警告: 归一化文件未找到: {normalization_file}", flush=True)
        state_normalizer = None
        observation_normalizer = None
    
    # 初始化模型（参考 visualize_trajectory.py）
    print(f"\n{'='*60}", flush=True)
    print("初始化 Diffusion_Planner 模型...", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    model = Diffusion_Planner(policy_config).to(device)
    
    # 保存加载前的权重（用于验证）
    import hashlib
    model_before = model.state_dict()
    before_hash = None
    if len(model_before) > 0:
        first_key_before = list(model_before.keys())[0]
        before_hash = hashlib.md5(model_before[first_key_before].cpu().numpy().tobytes()).hexdigest()[:8]
        print(f"加载前模型权重hash（第一个参数）: {before_hash}", flush=True)
    
    # 先检查checkpoint中的权重hash
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'policy_state_dict' in checkpoint:
            policy_dict = checkpoint['policy_state_dict']
            # 移除module.前缀
            clean_dict = {}
            for k, v in policy_dict.items():
                if k.startswith('module.'):
                    clean_dict[k[7:]] = v
                else:
                    clean_dict[k] = v
            if len(clean_dict) > 0:
                first_key_ckpt = list(clean_dict.keys())[0]
                checkpoint_hash = hashlib.md5(clean_dict[first_key_ckpt].cpu().numpy().tobytes()).hexdigest()[:8]
                print(f"Checkpoint文件中第一个参数的hash: {checkpoint_hash}", flush=True)
    except Exception as e:
        print(f"警告: 无法读取checkpoint文件进行验证: {e}", flush=True)
    
    # 加载checkpoint（在设置eval模式之前加载，确保权重正确加载）
    load_checkpoint(model, checkpoint_path, device)
    
    # 验证权重是否真的被加载了
    model_after = model.state_dict()
    after_hash = None
    if len(model_after) > 0:
        first_key_after = list(model_after.keys())[0]
        after_hash = hashlib.md5(model_after[first_key_after].cpu().numpy().tobytes()).hexdigest()[:8]
        print(f"加载后模型权重hash（第一个参数）: {after_hash}", flush=True)
        
        # 比较checkpoint hash和加载后的hash
        if 'checkpoint_hash' in locals() and checkpoint_hash:
            if checkpoint_hash == after_hash:
                print(f"✓ 验证: 加载后的权重与checkpoint中的权重匹配", flush=True)
                
                # 检查是否和base_weight相同
                base_weight_path = os.path.join(os.path.dirname(__file__), 'base_weight', 'model_epoch_500_trainloss_0.0486.pth')
                if os.path.exists(base_weight_path):
                    try:
                        base_ckpt = torch.load(base_weight_path, map_location=device, weights_only=False)
                        if 'model' in base_ckpt:
                            base_dict = base_ckpt['model']
                            base_clean = {}
                            for k, v in base_dict.items():
                                if k.startswith('module.'):
                                    base_clean[k[7:]] = v
                                else:
                                    base_clean[k] = v
                            if first_key_after in base_clean:
                                base_hash = hashlib.md5(base_clean[first_key_after].cpu().numpy().tobytes()).hexdigest()[:8]
                                if checkpoint_hash == base_hash:
                                    print(f"⚠ 重要警告: Checkpoint中的policy权重与base_weight相同！", flush=True)
                                    print(f"   这意味着训练过程中policy权重没有更新。", flush=True)
                                    print(f"   建议使用其他checkpoint或检查训练代码。", flush=True)
                    except Exception:
                        pass
            else:
                print(f"⚠ 警告: 加载后的权重与checkpoint中的权重不匹配！", flush=True)
                print(f"   Checkpoint: {checkpoint_hash}, 加载后: {after_hash}", flush=True)
        
        # 比较加载前后的hash
        if before_hash and after_hash:
            if before_hash == after_hash:
                print("⚠ 警告: 模型权重可能没有正确加载（权重未改变）", flush=True)
                print("   这可能导致所有模型评估结果相同！", flush=True)
            else:
                print("✓ 模型权重已成功加载（权重已改变）", flush=True)
    
    # 设置模型为评估模式
    model.eval()
    print("模型已设置为评估模式", flush=True)
    
    # 评估
    print(f"开始评估...", flush=True)
    rewards = []
    
    with torch.no_grad():
        for i, idx in enumerate(selected_indices):
            npz_file = npz_files[idx]
            
            # 加载数据
            data = opendata(npz_file)
            
            # 准备state输入（参考 visualize_trajectory.py）
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
            
            # 归一化state
            state_norm = state
            if observation_normalizer is not None:
                state_norm = observation_normalizer(state)
            
            # 使用模型生成自车轨迹（不使用数据集中的真实轨迹）
            _, decoder_output = model(state_norm)
            predicted_action = decoder_output['prediction'][0, 0, :, :]  # [T, 4] 模型生成的自车轨迹
            predicted_action_flat = predicted_action.reshape(1, -1)  # [1, T*4]
            
            # 提取状态特征
            state_features = extract_state_features(state, use_all_features=False)  # [1, state_feature_dim]
            
            # 使用模型生成的自车轨迹计算reward（不归一化，使用原始reward）
            computed_reward = compute_reward(state_features, predicted_action_flat, normalize=False)
            
            if computed_reward.dim() == 0:
                reward_value = computed_reward.item()
            else:
                reward_value = computed_reward.mean().item()
            
            rewards.append(reward_value)
            
            if (i + 1) % 10 == 0:
                print(f"已评估 {i + 1}/{num_scenarios} 个场景，当前平均reward: {np.mean(rewards):.6f}", flush=True)
    
    # 计算统计量
    rewards_array = np.array(rewards)
    mean_reward = float(rewards_array.mean())
    std_reward = float(rewards_array.std())
    min_reward = float(rewards_array.min())
    max_reward = float(rewards_array.max())
    
    # 输出结果
    print(f"\n{'='*60}", flush=True)
    print(f"评估结果", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"评估场景数: {num_scenarios}", flush=True)
    print(f"平均Reward: {mean_reward:.6f}", flush=True)
    print(f"Reward标准差: {std_reward:.6f}", flush=True)
    print(f"最小Reward: {min_reward:.6f}", flush=True)
    print(f"最大Reward: {max_reward:.6f}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'min_reward': min_reward,
        'max_reward': max_reward,
        'num_scenarios': num_scenarios,
    }


def main():
    parser = argparse.ArgumentParser(description='评估模型')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Checkpoint文件路径（必需）')
    parser.add_argument('--data_dir', type=str, default='/mnt/datanpz0.01',
                       help='数据目录（默认: /mnt/datanpz0.01）')
    parser.add_argument('--num_scenarios', type=int, default=100,
                       help='评估的场景数量（默认: 100）')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备类型（默认: cuda）')
    parser.add_argument('--fixed_seed', type=int, default=42,
                       help='固定随机种子，确保每次选择相同的场景（默认: 42）')
    parser.add_argument('--normalization_file', type=str, default='normalization.json',
                       help='归一化文件路径（默认: normalization.json）')
    
    args = parser.parse_args()
    
    # 评估模型
    results = evaluate_model(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        num_scenarios=args.num_scenarios,
        device=args.device,
        fixed_seed=args.fixed_seed,
        normalization_file=args.normalization_file,
    )
    
    print(f"评估完成！平均Reward: {results['mean_reward']:.6f}", flush=True)


if __name__ == '__main__':
    main()
