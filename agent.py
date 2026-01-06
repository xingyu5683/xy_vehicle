import torch
import numpy as np
from encoder_init import EncodeState
from networks.off_policy.ddqn.dueling_dqn import DuelingDQnetwork
from networks.off_policy.ddqn.replay_buffer import ReplayBuffer
from parameters import *

import os
import pickle
from parameters import DQN_CHECKPOINT_DIR  # 导入路径配置


class DQNAgent(object):

    def __init__(self, town, n_actions, state_dim=None):
        """
        初始化DQN代理
        :param town: 城镇名称
        :param n_actions: 离散动作数量
        :param state_dim: 编码后的状态维度（新增：用于适配弯道距离特征后的维度）
        """
        self.gamma = GAMMA
        self.alpha = DQN_LEARNING_RATE
        self.epsilon = EPSILON
        self.epsilon_end = EPSILON_END
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = MEMORY_SIZE
        self.batch_size = BATCH_SIZE
        self.train_step = 0
        self.bestval = -1e10

        # 核心修改1：动态设置ReplayBuffer的状态维度（优先使用传入的state_dim，否则用默认100）
        self.state_dim = state_dim if state_dim is not None else 100
        self.replay_buffer = ReplayBuffer(MEMORY_SIZE, self.state_dim, n_actions)

        # 核心修改2：确保DuelingDQnetwork的输入维度与state_dim一致（需配合DuelingDQnetwork修改）
        self.q_network_eval = DuelingDQnetwork(town, n_actions, MODEL_ONLINE, input_dim=self.state_dim)
        self.q_network_target = DuelingDQnetwork(town, n_actions, MODEL_TARGET, input_dim=self.state_dim)

        self.town = town  # 新增：记录城镇名称（用于路径拼接）
        # 新增：元数据保存路径
        self.meta_path = os.path.join(DQN_CHECKPOINT_DIR, self.town, 'dqn_meta.pkl')
        print(f"action num: {n_actions}")
        print(f"state dimension: {self.state_dim}")  # 新增：打印状态维度，便于调试

    def save_transition(self, observation, action, reward, new_observation, done):
        # 新增：验证状态维度，避免存入错误维度的数据
        if len(observation) != self.state_dim:
            # 若维度不匹配，调整维度（截断或补0）
            observation = self._adjust_state_dim(observation, self.state_dim)
            new_observation = self._adjust_state_dim(new_observation, self.state_dim)
        self.replay_buffer.save_transition(observation, action, reward, new_observation, done)

    def get_len_buffer(self):
        return len(self.replay_buffer)

    def get_action(self, train, observation):
        # 新增：确保观测维度与网络输入一致
        if len(observation) != self.state_dim:
            observation = self._adjust_state_dim(observation, self.state_dim)
            # 转换为张量（如果observation是numpy数组）
            if not torch.is_tensor(observation):
                observation = torch.tensor(observation, dtype=torch.float).to(self.q_network_eval.device)

        if train is True:
            if np.random.random() > self.epsilon:
                advantage = self.q_network_eval.forward(observation)
                action = torch.argmax(advantage).item()
            else:
                action = np.random.choice(self.action_space)
        else:
            advantage = self.q_network_eval.forward(observation)
            action = torch.argmax(advantage).item()

        return action

    def decrese_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= EPSILON_DECREMENT
        else:
            self.epsilon = self.epsilon_end

    def save_model(self, current_ep_reward, Episodic):
        save_best = False
        if current_ep_reward > self.bestval:
            self.bestval = current_ep_reward
            save_best = True
        if save_best:
            self.q_network_eval.save_checkpoint()
            self.q_network_target.save_checkpoint()
            # 新增：保存元数据（bestval、epsilon、train_step、state_dim）
            meta_data = {
                'bestval': self.bestval,
                'epsilon': self.epsilon,
                'train_step': self.train_step,
                'episodic': Episodic,
                'state_dim': self.state_dim  # 新增：保存状态维度，加载时恢复
            }
            # 确保目录存在
            os.makedirs(os.path.dirname(self.meta_path), exist_ok=True)
            with open(self.meta_path, 'wb') as f:
                pickle.dump(meta_data, f)
            print('Overwrote best model:', Episodic)

    def load_model(self):
        # 1. 加载网络权重
        try:
            self.q_network_eval.load_checkpoint()
            self.q_network_target.load_checkpoint()
            print("✅ 成功加载模型权重")
        except Exception as e:
            print(f"⚠️ 加载模型权重失败：{e}")
            return

        # 2. 加载元数据（恢复bestval、epsilon、train_step、state_dim）
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'rb') as f:
                meta_data = pickle.load(f)
            self.bestval = meta_data['bestval']
            self.epsilon = meta_data['epsilon']  # 恢复探索率，避免从头开始衰减
            self.train_step = meta_data['train_step']  # 恢复训练步数，避免重新计数
            # 新增：恢复状态维度（若存在）
            if 'state_dim' in meta_data:
                self.state_dim = meta_data['state_dim']
                # 更新ReplayBuffer的状态维度（若需要）
                self.replay_buffer = ReplayBuffer(MEMORY_SIZE, self.state_dim, len(self.action_space))
            print(f"✅ 成功恢复元数据：")
            print(f"- 历史最优奖励：{self.bestval:.2f}")
            print(f"- 当前探索率：{self.epsilon:.3f}")
            print(f"- 已训练步数：{self.train_step}")
            print(f"- 状态维度：{self.state_dim}")
        else:
            print(f"⚠️ 未找到元数据文件（{self.meta_path}），使用默认初始值")

    def learn(self):
        if self.replay_buffer.counter < self.batch_size:
            return

        observation, action, reward, new_observation, done = self.replay_buffer.sample_buffer()
        batch_idx = np.arange(self.batch_size)

        # 新增：确保张量设备与网络一致（避免CPU/GPU不匹配）
        device = self.q_network_eval.device
        observation = observation.to(device)
        action = action.to(device)
        reward = reward.to(device)
        new_observation = new_observation.to(device)
        done = done.to(device)

        with torch.no_grad():
            q_ = self.q_network_eval.forward(new_observation)
            next_actions = torch.argmax(q_, dim=-1)
            q_ = self.q_network_target.forward(new_observation)
            q_[done] = 0.0
            target = reward + self.gamma * q_[batch_idx, next_actions]

        q = self.q_network_eval.forward(observation)[batch_idx, action]

        loss = self.q_network_eval.loss(q, target.detach()).to(device)
        self.q_network_eval.optimizer.zero_grad()
        loss.backward()
        self.q_network_eval.optimizer.step()

        # # 新增：更新学习率（仅在评估网络更新时）
        # self.q_network_eval.update_learning_rate()

        self.train_step += 1

        if self.train_step % REPLACE_NETWORK == 0:
            self.q_network_target.load_state_dict(self.q_network_eval.state_dict())

        self.decrese_epsilon()

    # 新增：辅助函数：调整状态维度（截断或补0）
    def _adjust_state_dim(self, state, target_dim):
        """
        调整状态维度到目标维度
        :param state: 原始状态（numpy数组或torch张量）
        :param target_dim: 目标维度
        :return: 调整后的状态
        """
        # 转换为numpy数组处理
        if torch.is_tensor(state):
            is_tensor = True
            state_np = state.cpu().numpy()
        else:
            is_tensor = False
            state_np = np.array(state)

        # 展平为一维
        state_np = state_np.flatten()
        current_dim = len(state_np)

        if current_dim < target_dim:
            # 补0
            state_np = np.pad(state_np, (0, target_dim - current_dim), mode='constant')
        elif current_dim > target_dim:
            # 截断
            state_np = state_np[:target_dim]

        # 转换回原类型
        if is_tensor:
            state = torch.tensor(state_np, dtype=torch.float).to(self.q_network_eval.device)
        else:
            state = state_np

        return state