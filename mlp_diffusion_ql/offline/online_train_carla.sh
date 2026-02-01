#!/bin/bash

# 在线强化学习训练脚本
# 与CARLA环境实时交互并训练QL_Diffusion模型

# 激活conda环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate carla_py37

# 切换到脚本所在目录
cd "$(dirname "$0")"

echo "🚀 开始在线训练..."
echo "当前Python: $(which python)"
echo "当前目录: $(pwd)"

# 在线训练参数
python online_train_carla.py \
    --config ../configs/base.yaml \
    --output_dir ./log/online_train_output \
    --total_timesteps 500000 \
    --batch_size 256 \
    --buffer_size 100000 \
    --learning_starts 5000 \
    --train_freq 1 \
    --gradient_steps 1 \
    --lr 0.0003 \
    --save_freq 10000 \
    --eval_freq 5000 \
    --n_eval_episodes 3 \
    --discount 0.99 \
    --tau 0.005 \
    --eta 1.0 \
    --beta_schedule linear \
    --n_timesteps 100 \
    --hidden_dim 256 \
    --mode whole_grad \
    --critic_num_layers 3 \
    --use_custom_reward \
    --desired_speed 8.1 \
    --max_grad_norm 1.0 \
    --exploration_noise 0.1 \
    --max_episode_steps 1000 \
    --seed 0

# 可选：从预训练模型开始训练
# 取消下面的注释并设置正确的模型路径
# --pretrained_model ./log/offline_train_output_20260107_185848/final

echo "✅ 训练完成！"

