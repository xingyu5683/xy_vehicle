import os
import torch
import torch.nn as nn
import torch.optim as optim
from parameters import DQN_LEARNING_RATE, DQN_CHECKPOINT_DIR


class DuelingDQnetwork(nn.Module):
    def __init__(self, town, n_actions, model, input_dim=100):  # 核心修改：新增input_dim参数，默认100（兼容旧代码）
        super(DuelingDQnetwork, self).__init__()
        self.town = town
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(DQN_CHECKPOINT_DIR + self.town, model)

        # 核心修改：将固定的95+5改为动态的input_dim
        self.Linear1 = nn.Sequential(
            nn.Linear(input_dim, 256),  # 替换原95+5=100为input_dim
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.q = nn.Linear(64, self.n_actions)

        # 学习率调度相关参数（保留原有注释，若需要可启用）
        self.initial_lr = DQN_LEARNING_RATE  # 从parameters导入
        # self.current_lr = self.initial_lr * 0.1  # 初始学习率为正常的10%
        # self.lr_warmup_steps = 5000  # 热身步数（可调整）
        # self.current_step = 0

        self.optimizer = optim.Adam(self.parameters(), lr=DQN_LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    # 保留原有注释的学习率更新方法（若需要可启用）
    # def update_learning_rate(self):
    #     """线性增长学习率：从初始值到目标值"""
    #     self.current_step += 1
    #     if self.current_step <= self.lr_warmup_steps:
    #         # 线性插值：初始lr * 0.1 → 初始lr
    #         progress = self.current_step / self.lr_warmup_steps
    #         new_lr = self.initial_lr * (0.1 + 0.9 * progress)  # 0.1是初始缩放因子
    #         for param_group in self.optimizer.param_groups:
    #             param_group['lr'] = new_lr
    #         self.current_lr = new_lr

    def forward(self, x):
        # 新增：鲁棒性处理——确保输入是一维张量（避免多维输入导致维度不匹配）
        # 例如：如果x是形状为[batch_size, d1, d2]的张量，展平为[batch_size, d1*d2]
        if len(x.shape) > 2:
            x = x.flatten(1)  # 保留batch维度，展平其余维度
        # 若x是一维张量（单样本），保持不变；若x是二维（batch+feature），直接传入
        fc = self.Linear1(x)
        q = self.q(fc)
        return q

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        # 新增：加载权重时指定设备（避免CPU/GPU不匹配）
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))