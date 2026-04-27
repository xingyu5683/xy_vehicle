#!/bin/bash

# 验证Q网络输出是否能反映reward大小的脚本

# 默认参数
CHECKPOINT_PATH="/workspace/planner/Diffusion-Planner/training_log/ql_diffusion/2026-01-14-17:04:54/checkpoints/checkpoint_epoch_5000.pth"
DATA_DIR="${2:-/mnt/datanpzmini}"
NUM_SAMPLES="1000"
OUTPUT_DIR="${4:-./q_reward_validation}"
DEVICE="${5:-cuda}"

python validate_q_reward.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --data_dir "$DATA_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

