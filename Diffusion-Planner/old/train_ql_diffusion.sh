#!/bin/bash

###################################
# 简单的训练测试脚本
# 支持通过命令行参数调整batch_size
###################################

# 设置工作目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认参数
BATCH_SIZE=300  # 第一个参数为batch_size，默认为8

# 其他默认参数
DATA_DIR="/mnt/datanpzmini"
DEVICE="cuda"
NUM_EPOCHS=500
LOG_INTERVAL=10
# 随机种子设置（用于实验可重复性）
RANDOM_SEED=0  # 设置为 None 或留空则不设置随机种子
# Max Q Backup设置（用于减少Q值过估计）
USE_MAX_Q_BACKUP=true  # 设置为 true 启用 Max Q Backup
MAX_Q_BACKUP_SAMPLES=10  # Max Q Backup时采样的动作数量

echo "=========================================="
echo "训练测试脚本"
echo "=========================================="
echo "Batch大小: $BATCH_SIZE"
echo "数据目录: $DATA_DIR"
echo "设备: $DEVICE"
echo "训练轮数: $NUM_EPOCHS"
if [ -n "$RANDOM_SEED" ] && [ "$RANDOM_SEED" != "None" ]; then
    echo "随机种子: $RANDOM_SEED"
else
    echo "随机种子: 未设置（使用系统默认）"
fi
echo "Max Q Backup: $USE_MAX_Q_BACKUP"
if [ "$USE_MAX_Q_BACKUP" = "true" ]; then
    echo "Max Q Backup采样数: $MAX_Q_BACKUP_SAMPLES"
fi
echo "=========================================="
echo ""

# 构建训练命令
TRAIN_CMD="python -u train_ql_diffusion.py \
    --data_dir \"$DATA_DIR\" \
    --device \"$DEVICE\" \
    --batch_size \"$BATCH_SIZE\" \
    --num_epochs \"$NUM_EPOCHS\" \
    --log_interval \"$LOG_INTERVAL\""

# 如果设置了随机种子，添加相应参数
if [ -n "$RANDOM_SEED" ] && [ "$RANDOM_SEED" != "None" ]; then
    TRAIN_CMD="$TRAIN_CMD --seed $RANDOM_SEED"
fi

# 如果启用Max Q Backup，添加相应参数
if [ "$USE_MAX_Q_BACKUP" = "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --max_q_backup --max_q_backup_samples $MAX_Q_BACKUP_SAMPLES"
fi

# 可选：使用r_fun（取消注释以启用）
# TRAIN_CMD="$TRAIN_CMD --use_r_fun"

# 运行训练
eval $TRAIN_CMD
