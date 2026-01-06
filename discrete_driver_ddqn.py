import os
import sys
import time
import random
import numpy as np
import argparse
import logging
import pickle
import torch
import torchvision
from distutils.util import strtobool
from threading import Thread
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from simulation.connection import ClientConnection
from simulation.environment import CarlaEnvironment
from networks.off_policy.ddqn.agent import DQNAgent
from encoder_init import EncodeState
from parameters import *
import torchvision.models as models
from torchvision.models import ResNet34_Weights

# -------------------- 新增：导入模仿学习模型相关 --------------------
import torchvision.transforms as transforms
from PIL import Image
# 在文件顶部添加 deque 导入
from collections import deque


# 定义模仿学习模型结构（必须与你的预训练模型完全一致）
class ResNet34(torch.nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        # 使用torchvision0.12.0的ResNet34（Python3.7兼容）
        self.resnet = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.resnet.fc = torch.nn.Identity()  # 移除原全连接层

        # 运动状态分支 - 使用与训练代码一致的名称：kinematics_fc
        self.kinematics_fc = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32)
        )

        # 融合层 - 使用与训练代码一致的独立层名称，而非Sequential容器
        self.fc1 = torch.nn.Linear(512 + 32, 256)
        self.fc1_bn = torch.nn.BatchNorm1d(256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc2_bn = torch.nn.BatchNorm1d(128)
        self.relu = torch.nn.ReLU()

        # 动作输出层 - 使用与训练代码一致的名称：fc3
        self.fc3 = torch.nn.Linear(128, 3)

    def forward(self, x, kinematics):
        """
        前向传播函数
        Args:
            x: 图像输入，shape [batch_size, 3, H, W]
            kinematics: 运动状态输入，shape [batch_size, 2]
        Returns:
            control: 控制输出，shape [batch_size, 3]，其中：
                     - 前两列：油门/刹车，范围0~1
                     - 第三列：方向盘，范围-1~1
        """
        # 图像特征提取
        img_feat = self.resnet(x)  # shape: [batch_size, 512]

        # 运动状态特征提取
        kin_feat = self.kinematics_fc(kinematics)  # shape: [batch_size, 32]

        # 特征融合
        fusion_feat = torch.cat([img_feat, kin_feat], dim=1)  # shape: [batch_size, 544]
        fusion_feat = self.relu(self.fc1_bn(self.fc1(fusion_feat)))  # shape: [batch_size, 256]
        fusion_feat = self.relu(self.fc2_bn(self.fc2(fusion_feat)))  # shape: [batch_size, 128]

        # 动作输出
        control = self.fc3(fusion_feat)  # shape: [batch_size, 3]

        # 激活函数处理
        # 油门/刹车：sigmoid激活，范围0~1
        # 方向盘：tanh激活，范围-1~1
        throttle_brake = torch.sigmoid(control[:, :2])  # shape: [batch_size, 2]
        steering = torch.tanh(control[:, 2:3])  # shape: [batch_size, 1]

        # 拼接完整控制输出
        control = torch.cat([throttle_brake, steering], dim=1)  # shape: [batch_size, 3]

        return control


# 核心修改1：适配环境的5个离散动作空间（-0.5/-0.3/0.0/0.3/0.5转向）
def continuous_to_discrete(continuous_action, num_actions=5):
    """
    将连续动作转换为离散动作（适配环境的5个动作索引）
    Args:
        continuous_action: 连续动作，shape [3]（油门、刹车、转向）
        num_actions: 离散动作数量（固定为5，对应环境的5个动作）
    Returns:
        discrete_action: 离散动作索引（0~4）
    """
    # 1. 提取转向角并添加死区阈值
    steer = continuous_action[2]  # 转向角，范围-1~1
    dead_zone = 0.1  # 死区阈值：绝对值小于0.1的转向角视为直行
    if abs(steer) < dead_zone:
        steer = 0.0  # 轻微转向直接置为0，强制直行

    # 2. 映射到5个动作索引（适配环境的动作空间顺序）
    # 环境动作空间顺序：0(-0.5), 1(-0.3), 2(0.0), 3(0.3), 4(0.5)
    if steer <= -0.4:
        discrete_action = 0  # 大左转
    elif steer <= -0.2:
        discrete_action = 1  # 中左转
    elif abs(steer) < 0.2:
        discrete_action = 2  # 直行（核心）
    elif steer <= 0.4:
        discrete_action = 3  # 中右转
    else:
        discrete_action = 4  # 大右转

    # 3. 确保索引在有效范围内
    discrete_action = max(0, min(num_actions - 1, discrete_action))
    return discrete_action


# 图像预处理（与模仿学习一致）
def get_imitation_transform():
    return transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.1840, 0.1659, 0.1613),
            std=(0.2540, 0.2386, 0.2599)
        )
    ])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='ddqn', help='name of the experiment')
    parser.add_argument('--env-name', type=str, default='carla', help='name of the simulation environment')
    parser.add_argument('--learning-rate', type=float, default=DQN_LEARNING_RATE, help='learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=SEED, help='seed of the experiment')
    parser.add_argument('--total-episodes', type=int, default=EPISODES, help='total timesteps of the experiment')
    parser.add_argument('--train', type=bool, default=True, help='is it training?')
    parser.add_argument('--town', type=str, default="Town02", help='which town do you like?')
    parser.add_argument('--load-checkpoint', type=bool, default=MODEL_LOAD, help='resume training?')
    # 核心修改2：默认动作数改为5（适配环境的5个动作空间）
    parser.add_argument('--num-actions', type=int, default=5, help='num of discrete actions')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by deafult')
    parser.add_argument('--imitation-ckpt', type=str, default='',
                        help='path to imitation learning checkpoint')
    parser.add_argument('--convergence-episodes', type=int, default=50,
                        help='连续多少个episode奖励稳定则认为收敛')
    parser.add_argument('--convergence-threshold', type=float, default=0.05,
                        help='奖励波动阈值（方差/均值 < 阈值则认为稳定）')
    parser.add_argument('--min-episodes', type=int, default=200,
                        help='最小训练episode数（避免过早停止）')

    args = parser.parse_args()
    return args


def runner():
    # ========================================================================
    #                           BASIC PARAMETER & LOGGING SETUP
    # ========================================================================

    args = parse_args()
    exp_name = args.exp_name
    town = args.town
    train = args.train
    checkpoint_load = args.load_checkpoint
    # 核心修改3：动作数从参数读取（默认5）
    num_actions = args.num_actions

    reward_history = deque(maxlen=args.convergence_episodes)
    is_converged = False

    imitation_ckpt = args.imitation_ckpt

    try:
        if exp_name == 'ddqn':
            run_name = f"DDQN"
    except Exception as e:
        print(e)
        sys.exit()

    if train == True:
        writer = SummaryWriter(f"runs/{run_name}/{town}")
    else:
        writer = SummaryWriter(f"runs/{run_name}_TEST/{town}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key, value in vars(args).items()])))

    # Seeding to reproduce the results
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # 核心修改4：n_actions改为num_actions（5个）
    n_actions = num_actions  # 现在固定为5个动作：0(-0.5),1(-0.3),2(0.0),3(0.3),4(0.5)
    epoch = 0
    cumulative_score = 0
    episodic_length = list()
    scores = list()
    deviation_from_center = 0
    distance_covered = 0

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"使用设备：{device}")

    # ========================================================================
    #                           CREATING THE SIMULATION
    # ========================================================================
    try:
        client, world = ClientConnection(town).setup()
        logging.info("Connection has been setup successfully.")
    except Exception as e:
        logging.error(f"Connection has been refused by the server: {e}")
        sys.exit()
    if train:
        # 核心修改5：确认环境使用离散动作（continuous_action=False）
        env = CarlaEnvironment(client, world, town, continuous_action=False, algorithm='dqn', route_mode='1')
    else:
        env = CarlaEnvironment(client, world, town, checkpoint_frequency=None, continuous_action=False, algorithm='dqn',
                               route_mode='1')
    encode = EncodeState(LATENT_DIM)

    time.sleep(0.5)

    # ========================================================================
    # 核心修改：计算编码后的状态维度（适配弯道距离特征）
    # ========================================================================
    sample_observation = env.reset()
    sample_encoded = encode.process(sample_observation)
    state_dim = len(sample_encoded)
    print(f"\n✅ 编码后的状态维度：{state_dim}（包含弯道距离特征）")
    print(f"✅ 离散动作数量：{n_actions}（适配环境的5个动作空间）")

    # ========================================================================
    #                           ALGORITHM
    # ========================================================================
    if train is False:  # Test
        # 核心修改6：传入正确的动作数（5）和状态维度
        agent = DQNAgent(town, n_actions, state_dim=state_dim)
        agent.load_model()
        for params in agent.q_network_eval.parameters():
            params.requires_grad = False
        for params in agent.q_network_target.parameters():
            params.requires_grad = False
    else:  # Training
        if checkpoint_load:
            agent = DQNAgent(town, n_actions, state_dim=state_dim)
            agent.load_model()
        else:
            agent = DQNAgent(town, n_actions, state_dim=state_dim)

    # -------------------- 使用模仿学习模型填充经验池 --------------------
    if exp_name == 'ddqn' and not checkpoint_load:
        print(f"\n{'=' * 50}")
        print(f"开始使用模仿学习模型填充经验池...")
        print(f"模仿学习权重路径：{imitation_ckpt}")

        # 1. 加载模仿学习模型
        imitation_model = ResNet34().to(device)
        try:
            checkpoint = torch.load(imitation_ckpt, map_location=device)
            if 'model_state_dict' in checkpoint:
                imitation_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                imitation_model.load_state_dict(checkpoint)
            imitation_model.eval()
            print(f"✅ 成功加载模仿学习模型")
        except Exception as e:
            print(f"❌ 加载模仿学习模型失败：{e}，使用随机动作填充经验池")
            # 核心修改7：随机动作范围改为0~4（5个动作）
            while agent.replay_buffer.counter < agent.replay_buffer.buffer_size:
                observation = env.reset()
                observation_encoded = encode.process(observation)
                done = False
                while not done:
                    action = random.randint(0, n_actions - 1)  # 0~4
                    new_observation, reward, done, _ = env.step(action)
                    if new_observation is None:
                        break
                    new_observation_encoded = encode.process(new_observation)
                    agent.save_transition(observation_encoded, action, reward, new_observation_encoded, int(done))
                    observation_encoded = new_observation_encoded
            print(f"✅ 已使用随机动作填充经验池")
        else:
            # 2. 准备图像预处理
            transform = get_imitation_transform()

            # 3. 使用模仿学习模型填充经验池
            filled_count = 0
            max_filled = agent.replay_buffer.buffer_size
            while filled_count < max_filled:
                observation = env.reset()
                done = False
                while not done and filled_count < max_filled:
                    try:
                        raw_image, nav_features = observation
                        velocity_kmh = nav_features[1]
                        velocity = velocity_kmh / 3.6
                        waypoint_dist = nav_features[5]
                        kinematics_np = np.array([velocity, waypoint_dist], dtype=np.float32)
                        kinematics_tensor = torch.tensor(kinematics_np, dtype=torch.float32).unsqueeze(0).to(device)
                    except Exception as e:
                        print(f"解析环境输出失败：{e}，使用随机动作")
                        action = random.randint(0, n_actions - 1)  # 0~4
                    else:
                        # 4. 使用模仿学习模型生成连续动作
                        with torch.no_grad():
                            image_pil = Image.fromarray(raw_image.astype(np.uint8))
                            image_tensor = transform(image_pil).unsqueeze(0).to(device)
                            continuous_action = imitation_model(image_tensor, kinematics_tensor).cpu().numpy()[0]
                            # 核心修改8：转换为5个离散动作索引
                            action = continuous_to_discrete(continuous_action, num_actions=n_actions)
                            print(f"模型输出 | 转向: {continuous_action[2]:.3f} → 动作索引: {action}")

                    # 执行动作
                    new_observation, reward, done, _ = env.step(action)
                    if new_observation is None:
                        break

                    # 保存到经验池
                    observation_encoded = encode.process(observation)
                    new_observation_encoded = encode.process(new_observation)
                    agent.save_transition(observation_encoded, action, reward, new_observation_encoded, int(done))
                    filled_count += 1

                    # 更新观察
                    observation = new_observation

                print(f"已填充经验池：{filled_count}/{max_filled}")

            print(f"✅ 已使用模仿学习模型填充经验池")
    elif exp_name == 'ddqn' and checkpoint_load:
        # 核心修改9：随机动作范围改为0~4
        while agent.replay_buffer.counter < agent.replay_buffer.buffer_size:
            observation = env.reset()
            observation = encode.process(observation)
            done = False
            while not done:
                action = random.randint(0, n_actions - 1)  # 0~4
                new_observation, reward, done, _ = env.step(action)
                new_observation = encode.process(new_observation)
                agent.save_transition(observation, action, reward, new_observation, int(done))
                observation = new_observation

    # -------------------- 训练循环 --------------------
    if args.train:
        for step in range(epoch + 1, EPISODES + 1):
            if is_converged:
                break

            # Reset
            done = False
            observation = env.reset()
            observation = encode.process(observation)
            current_ep_reward = 0

            # Episode start: timestamp
            t1 = datetime.now()

            while not done:
                # 核心修改10：agent输出0~4的动作索引，直接传给环境
                action = agent.get_action(args.train, observation)
                # 防御性检查：确保动作索引在0~4范围内
                action = max(0, min(n_actions - 1, action))
                new_observation, reward, done, info = env.step(action)

                if new_observation is None:
                    break
                new_observation = encode.process(new_observation)
                current_ep_reward += reward

                agent.save_transition(observation, action, reward, new_observation, int(done))
                if agent.get_len_buffer() > WARMING_UP:
                    agent.learn()

                observation = new_observation

            # Episode end : timestamp
            t2 = datetime.now()
            t3 = t2 - t1
            episodic_length.append(abs(t3.total_seconds()))

            deviation_from_center += info[1]
            distance_covered += info[0]

            scores.append(current_ep_reward)
            reward_history.append(current_ep_reward)

            # 收敛判断逻辑
            if step >= args.min_episodes:
                if len(reward_history) == args.convergence_episodes:
                    mean_reward = np.mean(reward_history)
                    var_reward = np.var(reward_history)

                    if mean_reward != 0:
                        cv_reward = var_reward / abs(mean_reward)
                    else:
                        cv_reward = float('inf')

                    if cv_reward < args.convergence_threshold:
                        is_converged = True
                        print(f"\n{'=' * 60}")
                        print(f"🎉 模型已收敛！")
                        print(f"收敛指标：")
                        print(f"- 连续 {args.convergence_episodes} 个episode的平均奖励：{mean_reward:.2f}")
                        print(f"- 奖励方差：{var_reward:.2f}")
                        print(f"- 变异系数：{cv_reward:.4f}（< {args.convergence_threshold}）")
                        print(f"训练总episode数：{step}")
                        print(f"{'=' * 60}\n")

                        agent.save_model(current_ep_reward, step)
                        data_obj = {'cumulative_score': cumulative_score, 'epsilon': agent.epsilon, 'epoch': step}
                        os.makedirs(f'checkpoints/DDQN/{town}', exist_ok=True)
                        with open(f'checkpoints/DDQN/{town}/checkpoint_ddqn.pickle', 'wb') as handle:
                            pickle.dump(data_obj, handle)
                        break

            if checkpoint_load:
                cumulative_score = ((cumulative_score * (step - 1)) + current_ep_reward) / (step)
            else:
                cumulative_score = np.mean(scores)

            print('Starting Episode: ', step, ', Epsilon Now:  {:.3f}'.format(agent.epsilon),
                  'Reward:  {:.2f}'.format(current_ep_reward), ', Average Reward:  {:.2f}'.format(cumulative_score))
            agent.save_model(current_ep_reward, step)

            if step >= 10 and step % 10 == 0:
                if exp_name == 'ddqn':
                    data_obj = {'cumulative_score': cumulative_score, 'epsilon': agent.epsilon, 'epoch': step}
                    os.makedirs(f'checkpoints/DDQN/{town}', exist_ok=True)
                    with open(f'checkpoints/DDQN/{town}/checkpoint_ddqn.pickle', 'wb') as handle:
                        pickle.dump(data_obj, handle)

                writer.add_scalar("Cumulative Reward/info", cumulative_score, step)
                writer.add_scalar("Epsilon/info", agent.epsilon, step)
                writer.add_scalar("Episodic Reward/episode", scores[-1], step)
                writer.add_scalar("Average Episodic Reward/info", np.mean(scores[-10]), step)
                writer.add_scalar("Episode Length (s)/info", np.mean(episodic_length), step)
                writer.add_scalar("Average Deviation from Center/episode", deviation_from_center / 10, step)
                writer.add_scalar("Average Distance Covered (m)/episode", distance_covered / 10, step)

                episodic_length = list()
                deviation_from_center = 0
                distance_covered = 0

        print("Terminating the run.")
        sys.exit()
    else:
        # Testing
        for step in range(epoch + 1, EPISODES + 1):
            # Reset
            done = False
            observation = env.reset()
            observation = encode.process(observation)
            current_ep_reward = 0

            # Episode start: timestamp
            t1 = datetime.now()

            while not done:
                # 核心修改11：测试阶段同样输出0~4的动作索引
                action = agent.get_action(args.train, observation)
                action = max(0, min(n_actions - 1, action))
                new_observation, reward, done, info = env.step(action)

                if new_observation is None:
                    break
                new_observation = encode.process(new_observation)
                current_ep_reward += reward
                observation = new_observation

            # Episode end : timestamp
            t2 = datetime.now()
            t3 = t2 - t1
            episodic_length.append(abs(t3.total_seconds()))

            deviation_from_center += info[1]
            distance_covered += info[0]

            scores.append(current_ep_reward)

            if checkpoint_load:
                cumulative_score = ((cumulative_score * (step - 1)) + current_ep_reward) / (step)
            else:
                cumulative_score = np.mean(scores)

            print('Starting Episode: ', step, ', Epsilon Now:  {:.3f}'.format(agent.epsilon),
                  'Reward:  {:.2f}'.format(current_ep_reward), ', Average Reward:  {:.2f}'.format(cumulative_score))

            writer.add_scalar("TEST: Episodic Reward/episode", scores[-1], step)
            writer.add_scalar("TEST: Cumulative Reward/info", cumulative_score, step)
            writer.add_scalar("TEST: Episode Length (s)/info", np.mean(episodic_length), step)
            writer.add_scalar("TEST: Deviation from Center/episode", deviation_from_center, step)
            writer.add_scalar("TEST: Distance Covered (m)/episode", distance_covered, step)

            episodic_length = list()
            deviation_from_center = 0
            distance_covered = 0

        print("Terminating the run.")
        sys.exit()


if __name__ == "__main__":
    runner()