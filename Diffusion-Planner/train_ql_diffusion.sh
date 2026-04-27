#!/bin/bash

###################################
# 简单的训练测试脚本
# 支持通过命令行参数调整batch_size
# 支持两卡DDP训练（torchrun）
###################################

# 设置工作目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认参数（总batch_size；DDP下会自动按GPU数均分）
BATCH_SIZE=300
NUM_GPUS=2
MASTER_PORT=29529

# 可选：允许第一个参数覆盖batch_size
if [ -n "$1" ]; then
    BATCH_SIZE="$1"
fi

# 默认使用两卡（若未显式指定CUDA_VISIBLE_DEVICES，则默认用0,1）
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
fi

export PYTHONUNBUFFERED=1

# 其他默认参数
DATA_DIR="/mnt/datanpzmini"
DEVICE="cuda"
NUM_EPOCHS=5000
LOG_INTERVAL=10
# 随机种子设置（用于实验可重复性）
RANDOM_SEED=0  # 设置为 None 或留空则不设置随机种子
# Max Q Backup设置（用于减少Q值过估计）
USE_MAX_Q_BACKUP=true  # 设置为 true 启用 Max Q Backup
MAX_Q_BACKUP_SAMPLES=10  # Max Q Backup时采样的动作数量
# 折扣因子（discount factor）
DISCOUNT=0.99  # 默认0.99，降低可以提高训练稳定性
# Q loss权重（eta）
ETA=0.1  # Q loss 权重，较小值让 BC loss 主导
# 是否使用 r_fun 模式（直接用 reward 梯度，绕过 Q 网络）
USE_R_FUN=false
# CQL 权重（防止 Q 值过估计，但太大会压死 Q 值）
ALPHA_CQL=0  # 0.1 太大会压死 Q 值，0 会导致 Q 值爆炸，0.01 是折中
# Q 值 tanh 限制范围（防止 Q 值爆炸）
MAX_Q=50  # 限制 Q 值在 [-50, 50]，比 10 更大的空间
# Critic网络结构配置
CRITIC_HIDDEN_DIM=512  # Critic隐藏层维度
CRITIC_NUM_LAYERS=3    # Critic层数
# Critic网络checkpoint路径（如果提供，critic将加载固定权重并不再更新）
CRITIC_CHECKPOINT_PATH=""  # 设置为空字符串或留空则不使用固定critic，例如: "./training_log/ql_diffusion/xxx/checkpoints/latest.pth"
# 断点续训：从已有checkpoint继续训练
RESUME_CHECKPOINT=""

echo "=========================================="
echo "训练测试脚本"
echo "=========================================="
echo "GPU数量: $NUM_GPUS"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
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
echo "Discount Factor: $DISCOUNT"
echo "Eta (Q loss weight): $ETA"
echo "Use R_Fun: $USE_R_FUN"
if [ "$USE_R_FUN" = "false" ]; then
    echo "Alpha CQL: $ALPHA_CQL"
    echo "Max Q: $MAX_Q"
fi
echo "Critic Hidden Dim: $CRITIC_HIDDEN_DIM"
echo "Critic Num Layers: $CRITIC_NUM_LAYERS"
if [ -n "$CRITIC_CHECKPOINT_PATH" ]; then
    echo "Critic Checkpoint路径: $CRITIC_CHECKPOINT_PATH (Critic将被固定)"
else
    echo "Critic Checkpoint路径: 未设置 (Critic将正常更新)"
fi
echo "=========================================="
echo ""

# DDP时检查batch能否均分
if [ "$NUM_GPUS" -gt 1 ]; then
    if [ $((BATCH_SIZE % NUM_GPUS)) -ne 0 ]; then
        echo "ERROR: DDP模式下 BATCH_SIZE($BATCH_SIZE) 必须能被 NUM_GPUS($NUM_GPUS) 整除"
        exit 1
    fi
fi

# 构建训练命令
if [ "$NUM_GPUS" -gt 1 ]; then
    TRAIN_CMD="torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train_ql_diffusion.py \
    --ddp --port $MASTER_PORT \
    --data_dir \"$DATA_DIR\" \
    --device \"$DEVICE\" \
    --batch_size \"$BATCH_SIZE\" \
    --num_epochs \"$NUM_EPOCHS\" \
    --log_interval \"$LOG_INTERVAL\""
else
    TRAIN_CMD="python -u train_ql_diffusion.py \
    --data_dir \"$DATA_DIR\" \
    --device \"$DEVICE\" \
    --batch_size \"$BATCH_SIZE\" \
    --num_epochs \"$NUM_EPOCHS\" \
    --log_interval \"$LOG_INTERVAL\""
fi

# 如果设置了随机种子，添加相应参数
if [ -n "$RANDOM_SEED" ] && [ "$RANDOM_SEED" != "None" ]; then
    TRAIN_CMD="$TRAIN_CMD --seed $RANDOM_SEED"
fi

# 如果启用Max Q Backup，添加相应参数
if [ "$USE_MAX_Q_BACKUP" = "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --max_q_backup --max_q_backup_samples $MAX_Q_BACKUP_SAMPLES"
fi

# 添加discount参数
TRAIN_CMD="$TRAIN_CMD --discount $DISCOUNT"

# 添加eta参数（Q loss权重）
TRAIN_CMD="$TRAIN_CMD --eta $ETA"

# 断点续训checkpoint
if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume_checkpoint \"$RESUME_CHECKPOINT\""
fi

# 如果设置了Critic checkpoint路径，添加相应参数
if [ -n "$CRITIC_CHECKPOINT_PATH" ]; then
    TRAIN_CMD="$TRAIN_CMD --critic_checkpoint_path \"$CRITIC_CHECKPOINT_PATH\""
fi

# 使用r_fun模式（直接用 reward 梯度，绕过 Q 网络）
if [ "$USE_R_FUN" = "true" ]; then
    TRAIN_CMD="$TRAIN_CMD --use_r_fun"
    # r_fun 模式下关闭 CQL 和 max_q 限制
    TRAIN_CMD="$TRAIN_CMD --alpha_cql 0 --max_q 0"
else
    # Q 网络模式：添加 CQL 和 max_q 参数
    TRAIN_CMD="$TRAIN_CMD --alpha_cql $ALPHA_CQL --max_q $MAX_Q"
fi

# 添加 Critic 网络结构参数
TRAIN_CMD="$TRAIN_CMD --critic_hidden_dim $CRITIC_HIDDEN_DIM --critic_num_layers $CRITIC_NUM_LAYERS"

# 运行训练
eval $TRAIN_CMD
