#定义初始化环境，注册Carla的gym环境，包括id和entry_point
from gymnasium.envs.registration import register

register(
    id='carla-v0_test1',
    entry_point='gym_carla.envs:CarlaEnv'
)