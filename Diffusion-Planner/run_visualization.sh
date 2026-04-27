#!/bin/bash
# 运行轨迹可视化脚本的示例

# 设置参数
DATA_DIR="/mnt/datanpzmini"
# CHECKPOINT_PATH="./base_weight/model_epoch_500_trainloss_0.0486.pth"
# CHECKPOINT_PATH="./training_log/ql_diffusion/2025-12-11-18:45:56/checkpoints/checkpoint_epoch_100.pth"
# CHECKPOINT_PATH="./training_log/ql_diffusion/2025-12-12-15:42:06/checkpoints/latest.pth"
# CHECKPOINT_PATH="./training_log/ql_diffusion/2025-12-12-15:42:06/checkpoints/checkpoint_epoch_100.pth"
# CHECKPOINT_PATH="./training_log/ql_diffusion/2025-12-15-16:45:51/checkpoints/checkpoint_epoch_50.pth"
# CHECKPOINT_PATH="./training_log/ql_diffusion/2025-12-15-18:34:39/checkpoints/latest.pth"
# CHECKPOINT_PATH="./training_log/ql_diffusion/2025-12-16-13:34:09/checkpoints/checkpoint_epoch_100.pth"
CHECKPOINT_PATH="/workspace/planner/Diffusion-Planner/training_log/ql_diffusion/2026-01-20-10:19:03/checkpoints/checkpoint_epoch_5000.pth"
DATA_IDX=3  # 选择第几个npz文件（从0开始）
OUTPUT_DIR="./visualization_output"
NORMALIZATION_FILE="normalization.json"

# 运行可视化
python visualize_trajectory.py \
    --data_dir "$DATA_DIR" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --data_idx $DATA_IDX \
    --output_dir "$OUTPUT_DIR" \
    --normalization_file "$NORMALIZATION_FILE" \
    --device cuda

echo "Visualization completed! Check output in $OUTPUT_DIR"