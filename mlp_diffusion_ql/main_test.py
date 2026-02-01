import random
import time
import os.path as osp

import gymnasium as gym
import carla
import gym_carla # 导入环境
from util.run_util import load_config, set_seed


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.realpath(__file__))))
    args = parser.parse_args()
    args_dict = vars(args)

    # refence to FREA, adding a "list loop" can execute multiple scenarios
    agent_config_path = osp.join(args.ROOT_DIR, 'configs', 'base.yaml')
    env_params =  load_config(agent_config_path)

    # set random seed
    set_seed(env_params['seed'])

    env = gym.make('carla-v0_test1', env_params=env_params)
    obs, info = env.reset()


    def get_action(env, obs):
        """Use autopilot action."""

        # Use autopilot (Expert mode)
        real_env = env
        while hasattr(real_env, "env"):
            real_env = real_env.env


        real_env.ego.set_autopilot(True)
        control = real_env.ego.get_control()
        if control.throttle >=0:
            longitudinal = control.throttle
        elif control.brake >=0:
            longitudinal = control.brake
        action = [longitudinal, control.steer]

        return action

    # if train is True:


    for episode in range(env_params['total_episodes']):  # Run 10 episodes
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = get_action(env, obs)
            next_obs, reward, done, _, info = env.step(action)
            # time.sleep(0.05)

            # print(f"Step: {env.time_step}, Reward: {reward:.2f}, Done: {done}")
            obs = next_obs
            total_reward += reward

        print(f"Episode {episode} finished. Total reward: {total_reward:.2f}")

    env.close()