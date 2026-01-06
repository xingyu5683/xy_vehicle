"""

    All the much needed hyper-parameters needed for the algorithm implementation. 

"""
# common
MODEL_LOAD = False  # resume
SEED = 0
IM_WIDTH = 160
IM_HEIGHT = 80
GAMMA = 0.998

# VAE Bottleneck
LATENT_DIM = 95

# Dueling DQN parameters
EPISODES = 1000
MEMORY_SIZE = 1000#原1000
DQN_LEARNING_RATE = 0.0003
EPSILON = 1.0#原0.8
EPSILON_END = 0.01#原0.1
EPSILON_DECREMENT = (EPSILON - EPSILON_END) / 50000  # 总衰减步数5万步，可根据你的场景调整 原来：0.000005
BATCH_SIZE = 64
WARMING_UP = 100
REPLACE_NETWORK = 5#原5次
DQN_CHECKPOINT_DIR = 'preTrained_models/ddqn/'
MODEL_ONLINE = 'carla_dueling_dqn_online.pth'
MODEL_TARGET = 'carla_dueling_dqn_target.pth'
NUM_ACTIONS = 7  # 8, 7

# Proximal Policy Optimization(PPO) parameters
EPISODE_LENGTH = 7500
TOTAL_TIMESTEPS = 20e6
ACTION_STD_INIT = 0.2
TEST_TIMESTEPS = 5e4
PPO_LEARNING_RATE = 1e-4
PPO_CHECKPOINT_DIR = 'preTrained_models/ppo/'
POLICY_CLIP = 0.2

# SAC parameters
SAC_CHECKPOINT_DIR = 'preTrained_models/sac/'
obs_dim = 9
action_dim = 2
train_total_steps = 5e10
test_interval_steps = 2e3
save_interval_steps = 4e4
warmup_steps = 4e2
eval_episodes = 3
memory_size = 8e5
batch_size = 256 * 4
gamma = 0.99
tau = 0.005
alpha = 0.2
actor_lr = 3e-4
critic_lr = 3e-4

