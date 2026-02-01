python train.py \
    --data_file ./dataset/carla_offline_dataset.hdf5\
    --epochs 600 \
    --batch_size 5000 \
    --reward_tune normalize \
    --save_freq 100 \
    --use_custom_reward \
    --seed 0 \
    --no_critic_update \

# 注意：batch_size 10000
