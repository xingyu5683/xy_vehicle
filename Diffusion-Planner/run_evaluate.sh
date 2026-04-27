#!/bin/bash

###################################
# 模型评估脚本
# 从固定的100个场景计算平均reward
###################################

# 设置工作目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认参数
DATA_DIR="/mnt/datanpz0.01"
DEVICE="cuda"
NUM_SCENARIOS=100
FIXED_SEED=42  # 固定随机种子，确保每次选择相同的场景
NORMALIZATION_FILE="normalization.json"

# Checkpoint路径：如果提供了命令行参数则使用参数，否则使用默认路径
if [ -n "$1" ]; then
    CHECKPOINT_PATH="$1"
    echo "使用命令行参数指定的checkpoint路径"
else
    CHECKPOINT_PATH="./training_log/ql_diffusion/2025-12-08-18:49:12/checkpoints/latest.pth"
    # CHECKPOINT_PATH="./training_log/ql_diffusion/2025-12-18-10:41:15/checkpoints/latest.pth"
    echo "使用默认checkpoint路径（可通过命令行参数覆盖）"
fi

echo "=========================================="
echo "模型评估脚本"
echo "=========================================="
echo "Checkpoint路径: $CHECKPOINT_PATH"
echo "数据目录: $DATA_DIR"
echo "设备: $DEVICE"
echo "评估场景数: $NUM_SCENARIOS"
echo "固定随机种子: $FIXED_SEED"
echo "归一化文件: $NORMALIZATION_FILE"
echo "=========================================="
echo ""

# 构建评估命令
EVAL_CMD="python -u evaluate_model.py \
    --checkpoint_path \"$CHECKPOINT_PATH\" \
    --data_dir \"$DATA_DIR\" \
    --num_scenarios \"$NUM_SCENARIOS\" \
    --device \"$DEVICE\" \
    --fixed_seed \"$FIXED_SEED\" \
    --normalization_file \"$NORMALIZATION_FILE\""

# 运行评估
eval $EVAL_CMD

