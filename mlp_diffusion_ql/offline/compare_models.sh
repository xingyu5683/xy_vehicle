#!/bin/bash
# 模型对比评估脚本

# 激活conda环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate carla_py37

echo "================================"
echo "    模型对比评估 - BC vs QL"
echo "================================"
echo ""

# 设置模型路径（请根据实际情况修改）
BC_MODEL_PATH="./log/offline_train_output_20260110_145220/final"  # BC模型路径
QL_MODEL_PATH="./log/offline_train_output_20260113_190040/final"  # QL微调后的模型路径

# 检查模型路径是否存在
if [ ! -d "$BC_MODEL_PATH" ]; then
    echo "❌ 错误: BC模型路径不存在: $BC_MODEL_PATH"
    echo "请修改脚本中的 BC_MODEL_PATH 变量"
    exit 1
fi

if [ ! -d "$QL_MODEL_PATH" ]; then
    echo "❌ 错误: QL模型路径不存在: $QL_MODEL_PATH"
    echo "请修改脚本中的 QL_MODEL_PATH 变量"
    exit 1
fi

echo ">> BC模型路径: $BC_MODEL_PATH"
echo ">> QL模型路径: $QL_MODEL_PATH"
echo ""

# 运行对比评估
python compare_models.py \
    --bc_model "$BC_MODEL_PATH" \
    --ql_model "$QL_MODEL_PATH" \
    --config ../configs/base.yaml \
    --output_dir ./log/model_comparison \
    --n_eval_episodes 20 \
    --max_episode_steps 1000 \
    --use_custom_reward \
    --desired_speed 8.1 \
    --device cuda \
    --seed 42

echo ""
echo "================================"
echo "    对比评估完成！"
echo "================================"

