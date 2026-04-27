"""
检查多个checkpoint的policy权重是否相同
"""
import torch
import hashlib
import os
import sys

def check_checkpoint(ckpt_path):
    """检查checkpoint的policy权重hash"""
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        # 获取policy权重
        policy_dict = None
        if 'policy_state_dict' in ckpt:
            policy_dict = ckpt['policy_state_dict']
        elif 'model' in ckpt:
            policy_dict = ckpt['model']
        else:
            return None, "未找到policy权重"
        
        # 移除module.前缀
        clean_dict = {}
        for k, v in policy_dict.items():
            if k.startswith('module.'):
                clean_dict[k[7:]] = v
            else:
                clean_dict[k] = v
        
        if len(clean_dict) == 0:
            return None, "权重字典为空"
        
        # 计算前3个参数的hash
        hashes = []
        for i, (k, v) in enumerate(list(clean_dict.items())[:3]):
            h = hashlib.md5(v.numpy().tobytes()).hexdigest()[:16]
            hashes.append(f"{k[:30]}:{h}")
        
        # 计算所有权重的总体hash
        all_weights_hash = hashlib.md5(
            b''.join([v.numpy().tobytes() for v in list(clean_dict.values())[:10]])
        ).hexdigest()[:16]
        
        return all_weights_hash, hashes
    except Exception as e:
        return None, f"错误: {e}"

if __name__ == '__main__':
    # 检查base_weight
    base_weight = './base_weight/model_epoch_500_trainloss_0.0486.pth'
    if os.path.exists(base_weight):
        hash_val, details = check_checkpoint(base_weight)
        print(f"\n{base_weight}:")
        print(f"  总体hash: {hash_val}")
        if details:
            for d in details:
                print(f"  {d}")
    
    # 检查训练checkpoint
    training_dir = './training_log/ql_diffusion'
    if os.path.exists(training_dir):
        print(f"\n检查训练checkpoint:")
        for root, dirs, files in os.walk(training_dir):
            for file in files:
                if file.endswith('.pth'):
                    ckpt_path = os.path.join(root, file)
                    hash_val, details = check_checkpoint(ckpt_path)
                    if hash_val:
                        print(f"\n{ckpt_path}:")
                        print(f"  总体hash: {hash_val}")
                        if details and len(details) > 0:
                            print(f"  前3个参数hash: {details[0]}")

