#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : models.py
@Description: 模型定义（Diffusion模型、Critic网络、QL_Diffusion Agent）
"""
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import (
        SinusoidalPosEmb, extract, cosine_beta_schedule, 
        linear_beta_schedule, vp_beta_schedule, WeightedL2, EMA
    )


# ==================== Diffusion Model ====================

class MLP(nn.Module):
    """扩散模型的MLP网络"""
    def __init__(self, state_dim, action_dim, device, t_dim=16, hidden_dim=256):
        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish()
        )

        self.final_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, time, state):
        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)
        return self.final_layer(x)


class Diffusion(nn.Module):
    """扩散模型"""
    def __init__(self, state_dim, action_dim, model, max_action,
                 beta_schedule='linear', n_timesteps=100,
                 loss_type='l2', clip_denoised=True, predict_epsilon=True):
        super(Diffusion, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.model = model
        self.model_frozen = copy.deepcopy(model)

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = WeightedL2()

    def predict_start_from_noise(self, x_t, t, noise):
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s, grad=True):
        if grad:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.model_frozen(x, t, s))

        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, s, grad=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s, grad=grad)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(self, state, shape, verbose=False):
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)

        return x

    def sample(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, shape, *args, **kwargs)
        # 如果模型处于eval模式，使用frozen模型（grad=False）
        # 否则使用当前模型（grad=True，用于训练）
        # 注意：self.training 是 PyTorch 的属性，在 eval() 时为 False，在 train() 时为 True
        return action.clamp_(-self.max_action, self.max_action)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.model(x_noisy, t, state)
        assert noise.shape == x_recon.shape
        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)
        return loss

    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t, weights)

    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)

    def step_frozen(self):
        """同步model的参数到model_frozen"""
        # 方法1：直接复制参数（更快，但只同步参数）
        # for param, target_param in zip(self.model.parameters(), self.model_frozen.parameters()):
        #     target_param.data.copy_(param.data)
        
        # 方法2：使用state_dict（更完整，同步所有参数和缓冲区）
        # 确保model_frozen处于eval模式
        self.model_frozen.eval()
        # 加载model的state_dict到model_frozen
        self.model_frozen.load_state_dict(self.model.state_dict())

    def sample_t_middle(self, state):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        device = self.betas.device
        x = torch.randn(shape, device=device)
        t_middle = self.n_timesteps // 2
        for i in reversed(range(t_middle, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)
        return x.clamp_(-self.max_action, self.max_action)

    def sample_t_last(self, state):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        device = self.betas.device
        x = torch.randn(shape, device=device)
        t_last = self.n_timesteps - 1
        timesteps = torch.full((batch_size,), t_last, device=device, dtype=torch.long)
        x = self.p_sample(x, timesteps, state)
        return x.clamp_(-self.max_action, self.max_action)

    def sample_last_few(self, state):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        device = self.betas.device
        x = torch.randn(shape, device=device)
        last_few = max(1, self.n_timesteps // 10)
        for i in reversed(range(self.n_timesteps - last_few, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)
        return x.clamp_(-self.max_action, self.max_action)


# ==================== Critic Network ====================

class Critic(nn.Module):
    """Critic网络（Q函数）- 优化结构：渐进式降维"""
    def __init__(self, state_dim, action_dim, num_layers=3):
        """
        参数:
            state_dim: 状态维度 (67 for carla-v0_test1)
            action_dim: 动作维度 (2 for carla-v0_test1)
            num_layers: 隐藏层数量 (默认3层，推荐方案)
        
        网络结构（方案1，固定）:
        - 3层: 69 → 512 → 256 → 128 → 1
        
        注意: hidden_dim参数已移除，因为Critic使用固定的渐进式降维结构
        """
        super(Critic, self).__init__()
        
        input_dim = state_dim + action_dim  # 69维输入
        
        # 固定使用方案1的渐进式降维结构
        if num_layers == 3:
            # 方案1（推荐）：69 → 512 → 256 → 128 → 1
            layer_dims = [512, 256, 128]
        else:
            raise ValueError(f"目前只支持3层结构，num_layers={num_layers}。如需其他层数，请取消注释相关代码。")
        
        # 构建Q1网络
        q1_layers = []
        prev_dim = input_dim
        
        for dim in layer_dims:
            q1_layers.append(nn.Linear(prev_dim, dim))
            q1_layers.append(nn.Mish())
            prev_dim = dim
        
        # 输出层（1维）
        q1_layers.append(nn.Linear(prev_dim, 1))
        self.q1_model = nn.Sequential(*q1_layers)
        
        # Q2网络：与Q1相同的结构
        q2_layers = []
        prev_dim = input_dim
        
        for dim in layer_dims:
            q2_layers.append(nn.Linear(prev_dim, dim))
            q2_layers.append(nn.Mish())
            prev_dim = dim
        
        q2_layers.append(nn.Linear(prev_dim, 1))
        self.q2_model = nn.Sequential(*q2_layers)
        
        # 打印网络结构信息
        total_params_q1 = sum(p.numel() for p in self.q1_model.parameters())
        print(f">> Q网络结构: {input_dim} -> {' -> '.join(map(str, layer_dims))} -> 1")
        print(f">>   参数量: Q1={total_params_q1:,}, Q2={total_params_q1:,}, 总计={total_params_q1 * 2:,}")

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


# ==================== QL_Diffusion Agent ====================

class QL_Diffusion(object):
    """QL_Diffusion 算法实现"""
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount=0.99,
                 tau=0.005,
                 max_q_backup=False,
                 eta=1.0,
                 model_type='MLP',
                 beta_schedule='linear',
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 hidden_dim=256,
                 r_fun=None,
                 mode='whole_grad',
                 critic_num_layers=3,
                 critic_lr=None,
                 max_grad_norm=1,
                 update_critic: bool = True):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device, hidden_dim=hidden_dim)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        if r_fun is None:
            # Critic使用固定的渐进式降维结构，不需要hidden_dim参数
            self.critic = Critic(state_dim, action_dim, num_layers=critic_num_layers).to(device)
            self.critic_target = copy.deepcopy(self.critic)
            critic_lr = critic_lr if critic_lr is not None else lr
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta
        self.device = device
        self.max_q_backup = max_q_backup
        self.r_fun = r_fun
        self.mode = mode
        self.max_grad_norm = max_grad_norm
        # 是否更新critic（若False：跳过critic训练，actor仅使用BC loss）
        self.update_critic = bool(update_critic)

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(self, replay_buffer, iterations, batch_size=100):
        """训练函数"""
        total_critic_loss = 0.0
        total_bc_loss = 0.0
        total_q_loss = 0.0
        total_q1_new_action = 0.0  # 累计q1_new_action的值
        total_q2_new_action = 0.0  # 累计q2_new_action的值
        total_reward = 0.0  # 累计reward的值
        
        for step in range(iterations):
            # 采样批次
            state, action, reward, next_state, done = replay_buffer.sample(batch_size)
            
            # 累计reward
            total_reward += reward.mean().item()

            # Critic训练（使用Bellman方程）
            # 可选：禁用critic更新（只训练actor的BC部分）
            if self.r_fun is None and self.update_critic:
                # 计算当前Q值
                current_q1, current_q2 = self.critic(state, action)
                
                # 使用target critic计算next_state的Q值
                with torch.no_grad():
                    # 使用EMA actor生成next_action（用于计算target Q值，更稳定）
                    if self.step >= self.step_start_ema:
                        actor_for_target = self.ema_model
                    else:
                        actor_for_target = self.actor
                    
                    if self.mode == 'whole_grad':
                        next_action = actor_for_target(next_state)
                    elif self.mode == 't_middle':
                        next_action = actor_for_target.sample_t_middle(next_state)
                    elif self.mode == 't_last':
                        next_action = actor_for_target.sample_t_last(next_state)
                    elif self.mode == 'last_few':
                        next_action = actor_for_target.sample_last_few(next_state)
                    else:
                        next_action = actor_for_target(next_state)
                    
                    # 使用target critic计算target Q值
                    target_q1, target_q2 = self.critic_target(next_state, next_action)
                    target_q = torch.min(target_q1, target_q2)
                    
                    # Bellman target: r + gamma * (1 - done) * max Q(s', a')
                    target = reward + self.discount * (1 - done) * target_q
                
                # Critic损失：当前Q值与target的MSE
                critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                # 软更新target critic
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
                total_critic_loss += critic_loss.item()
            else:
                # r_fun != None：没有critic
                # update_critic=False：用户选择不更新critic
                pass

            # Policy训练
            bc_loss = self.actor.loss(action, state)

            if self.update_critic:
                # 正常QL_Diffusion：actor_loss = bc_loss + q_loss
                if self.mode == 'whole_grad':
                    new_action = self.actor(state)
                elif self.mode == 't_middle':
                    new_action = self.actor.sample_t_middle(state)
                elif self.mode == 't_last':
                    new_action = self.actor.sample_t_last(state)
                elif self.mode == 'last_few':
                    new_action = self.actor.sample_last_few(state)
                else:
                    new_action = self.actor(state)

                if self.r_fun is None:
                    q1_new_action, q2_new_action = self.critic(state, new_action)
                    # 记录q1_new_action和q2_new_action的平均值（用于TensorBoard）
                    total_q1_new_action += q1_new_action.mean().item()
                    total_q2_new_action += q2_new_action.mean().item()

                    if torch.rand(1).item() > 0.5:
                        lmbda = self.eta / q2_new_action.abs().mean().detach()
                        q_loss = - lmbda * q1_new_action.mean()
                    else:
                        lmbda = self.eta / q1_new_action.abs().mean().detach()
                        q_loss = - lmbda * q2_new_action.mean()
                else:
                    q_new_action = self.r_fun(new_action)
                    lmbda = self.eta / q_new_action.abs().mean().detach()
                    q_loss = - lmbda * q_new_action.mean()
                    # 如果使用r_fun，q1_new_action和q2_new_action设为0（不适用）
                    total_q1_new_action += 0.0
                    total_q2_new_action += 0.0

                actor_loss = bc_loss + q_loss  # TODO：加loss权重
            else:
                # 不更新critic：actor_loss 仅等于 bc_loss，去掉 q loss 部分
                q_loss = torch.zeros((), device=self.device, dtype=bc_loss.dtype)
                total_q1_new_action += 0.0
                total_q2_new_action += 0.0
                actor_loss = bc_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor.step_frozen()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            self.step += 1
            total_bc_loss += bc_loss.item()
            total_q_loss += q_loss.item()

        # 返回平均损失和q1_new_action、q2_new_action平均值
        avg_critic_loss = total_critic_loss / iterations if (self.r_fun is None and self.update_critic) else 0.0
        avg_bc_loss = total_bc_loss / iterations
        avg_q_loss = total_q_loss / iterations
        avg_q1_new_action = total_q1_new_action / iterations
        avg_q2_new_action = total_q2_new_action / iterations
        avg_reward = total_reward / iterations  # 平均reward
        
        return avg_bc_loss, avg_q_loss, avg_critic_loss, avg_q1_new_action, avg_q2_new_action, avg_reward

    def sample_action(self, state):
        """采样动作（用于环境交互，使用稳定的frozen模型）"""
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            # 设置为eval模式，sample方法会自动使用model_frozen
            # self.actor.eval()
            action = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def save_model(self, dir):
        """保存模型"""
        os.makedirs(dir, exist_ok=True)
        torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
        if self.r_fun is None:
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')
        print(f">> 模型已保存到: {dir}")

    def load_model(self, dir, load_critic: bool = True):
        """加载模型

        参数:
            dir: 模型目录，需包含 actor.pth（以及可选的 critic.pth）
            load_critic: 是否加载critic权重（默认True）。
                         若为False，则只加载actor，critic保持当前初始化权重。
        """
        # 加载actor模型
        actor_path = f'{dir}/actor.pth'
        if not os.path.exists(actor_path):
            raise FileNotFoundError(f"Actor模型文件不存在: {actor_path}")
        
        actor_state_dict = torch.load(actor_path, map_location=self.device)
        self.actor.load_state_dict(actor_state_dict, strict=False)
        self.actor.eval()  # 设置为评估模式
        # 同步model_frozen（用于采样时使用frozen模型）
        self.actor.step_frozen()  # 关键：确保model_frozen与model同步
        print(f">> Actor模型已从 {dir} 加载")
        
        # 加载critic模型
        if self.r_fun is None and load_critic:
            critic_path = f'{dir}/critic.pth'
            if not os.path.exists(critic_path):
                raise FileNotFoundError(f"Critic模型文件不存在: {critic_path}")
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
            self.critic.eval() # 设置为评估模式
            print(f">> Critic模型已从 {dir} 加载")
        elif self.r_fun is None and (not load_critic):
            print(f">> 跳过Critic加载：将使用当前初始化的Critic权重")

