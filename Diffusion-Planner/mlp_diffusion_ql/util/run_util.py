import os
import os.path as osp
import yaml
import random
import numpy as np
import torch

def load_config(config_path="default_config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """统一设置随机种子，保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    print(f"✅ Random seed set to {seed}")