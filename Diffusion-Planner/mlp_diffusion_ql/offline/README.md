# Offline RL Training Module

本模块包含用于Carla环境的离线强化学习训练代码，已按功能拆分为多个文件。

## 文件结构

```
offline/
├── __init__.py              # 模块初始化文件，导出主要类和函数
├── core/                    # 核心模块（模型、工具、数据等）
│   ├── __init__.py          # core模块初始化
│   ├── utils.py             # 辅助函数和工具类
│   │   ├── flatten_obs()   # 观测展平函数
│   │   ├── SinusoidalPosEmb # 正弦位置编码
│   │   ├── beta_schedule函数 # Beta调度函数
│   │   ├── WeightedL2       # 加权L2损失
│   │   └── EMA              # 指数移动平均
│   ├── models.py            # 模型定义
│   │   ├── MLP              # 扩散模型的MLP网络
│   │   ├── Diffusion        # 扩散模型
│   │   ├── Critic           # Critic网络（Q函数）
│   │   └── QL_Diffusion     # QL_Diffusion Agent
│   ├── reward_functions.py  # 奖励函数
│   │   └── carla_env_reward_function()  # Carla环境奖励函数
│   └── data_sampler.py      # 数据采样器
│       └── DataSampler      # HDF5数据采样器类
├── train.py                 # 训练主函数和入口
│   └── train_offline_rl()   # 离线RL训练主函数
├── collect_offline_data_carla.py  # 数据收集脚本
├── test_trained_model.py    # 模型测试脚本
├── plot_training_loss.py    # 训练loss可视化脚本
├── offline_train_carla.sh   # 训练启动脚本
├── log/                     # 训练输出目录（所有训练结果保存在此）
│   └── offline_train_output_*/  # 带时间戳的训练输出文件夹
└── dataset/                 # 数据集目录
```

## 使用方法

### 训练模型

```bash
# 使用shell脚本
./offline_train_carla.sh

# 或直接运行Python脚本
python train.py --data_file carla_offline_dataset.hdf5 --epochs 600 --batch_size 4096

# 注意：所有训练输出默认保存在 ./log/offline_train_output_YYYYMMDD_HHMMSS/ 目录下
```

### 导入模块

```python
# 方式1：作为模块导入
from offline import QL_Diffusion, flatten_obs, DataSampler

# 方式2：从core模块导入
from offline.core import QL_Diffusion, flatten_obs, DataSampler

# 方式3：直接导入（向后兼容，推荐使用方式1或2）
from offline.core.models import QL_Diffusion
from offline.core.utils import flatten_obs
from offline.core.data_sampler import DataSampler
```

## 主要改进

1. **模块化设计**：代码按功能拆分为多个文件，便于维护和扩展
2. **清晰的职责分离**：
   - `core/utils.py`: 工具函数
   - `core/models.py`: 模型定义
   - `core/data_sampler.py`: 数据处理
   - `core/reward_functions.py`: 奖励函数
   - `train.py`: 训练逻辑
3. **核心模块集中**：所有核心功能（模型、工具、数据、奖励）集中在 `core/` 文件夹中
4. **向后兼容**：导入路径支持多种方式，保持灵活性
5. **灵活的导入**：支持相对导入和绝对导入

## 输出目录

- **默认输出目录**: `./log/offline_train_output_YYYYMMDD_HHMMSS/`
- 所有训练结果（模型、TensorBoard日志、CSV文件、配置文件）都保存在 `log/` 文件夹下
- 每次训练会自动添加时间戳，避免覆盖之前的训练结果

## 注意事项

- 所有文件都在 `offline/` 目录下，可以作为一个Python包使用
- `__init__.py` 文件导出了主要的类和函数，方便使用
- 如果遇到导入错误，确保 `offline/` 目录在Python路径中
- 训练输出会自动保存到 `log/` 文件夹，保持目录整洁

