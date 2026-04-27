#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : reward_functions.py
@Description: 奖励函数定义
"""
import torch


def carla_env_reward_function(state, action, next_state, done, info, desired_speed=8.1):
    """
    与carla-v0_test1环境完全相同的奖励函数（基于carla_env.py中的_get_reward函数）
    
    参数:
        state: torch.Tensor, shape (N, 67) - 当前状态（展平的观测）
        action: torch.Tensor, shape (N, 2) - 动作 [longitudinal, steer]
        next_state: torch.Tensor, shape (N, 67) - 下一状态（展平的观测）
        done: torch.Tensor, shape (N, 1) - 是否终止
        info: torch.Tensor, shape (N, 3) - info信息 [ego_collision, ego_off_road, ego_min_dis]
        desired_speed: float - 期望速度 (m/s)，默认8.1
    
    返回:
        reward: torch.Tensor, shape (N, 1) - 奖励值
    
    注意:
        观测格式 (67维):
        - ego_state[0:9]: 自车状态
          - [3]: speed (m/s) - 从velocity计算得到
          - [6]: a_lat (横向加速度)
          - [7]: front_distance (m)
        - nearby_vehicles[9:29]: 周围车辆
        - waypoints[29:65]: 路径点
        - lane_info[65:67]: 车道信息
          - [65]: lane_width
          - [66]: lateral_offset
        
        info格式 (3维):
        - [0]: ego_collision (0.0或1.0)
        - [1]: ego_off_road (0.0或1.0)
        - [2]: ego_min_dis
    """
    # 初始化reward为0
    reward = torch.zeros(state.shape[0], 1, device=state.device, dtype=state.dtype)
    
    # 从state中提取所需信息
    speed = state[:, 3]  # ego_state[3]: speed (m/s)
    a_lat = state[:, 6]  # ego_state[6]: a_lat (横向加速度)
    lane_width = state[:, 65]  # lane_info[0]: lane_width
    lateral_offset = state[:, 66]  # lane_info[1]: lateral_offset
    
    # 从action中获取steer值
    steer = action[:, 1]  # action[1]: steer
    
    # 从info中获取碰撞和离路信息
    ego_collision = info[:, 0]  # ego_collision (0.0或1.0)
    ego_off_road = info[:, 1]  # ego_off_road (0.0或1.0)
    
    # ========== 旧版本的reward函数（已注释） ==========
    # # 1. Collision penalty
    # r_collision = torch.where(ego_collision > 0.5, torch.tensor(-1.0, device=state.device), torch.tensor(0.0, device=state.device))
    # 
    # # 2. Steering penalty
    # r_steer = -steer ** 2
    # 
    # # 3. Out of lane penalty
    # out_lane_thres = 2.0
    # r_out = torch.where(torch.abs(lateral_offset) > out_lane_thres, 
    #                     torch.tensor(-1.0, device=state.device), 
    #                     torch.tensor(0.0, device=state.device))
    # 
    # # 4. Speed reward and too fast penalty
    # r_fast = torch.where(speed > desired_speed, 
    #                      torch.tensor(-1.0, device=state.device), 
    #                      torch.tensor(0.0, device=state.device))
    # 
    # # 5. Lateral acceleration penalty
    # r_lat = -torch.abs(a_lat) * speed ** 2
    # 
    # # 6. Combine all rewards
    # reward = (10.0 * r_collision.unsqueeze(1) + 
    #          1.0 * speed.unsqueeze(1) + 
    #          10.0 * r_fast.unsqueeze(1) + 
    #          1.0 * r_out.unsqueeze(1) + 
    #          5.0 * r_steer.unsqueeze(1) + 
    #          0.2 * r_lat.unsqueeze(1) - 
    #          0.1)
    # ================================================
    
    # ========== 当前环境中使用的reward函数（与carla_env.py中的_get_reward一致） ==========
    
    # 1. Forward driving reward (within speed limit and along lane direction)
    # 如果速度在期望速度以内，奖励为速度值；否则惩罚超速
    speed_2d = speed.unsqueeze(1)  # (N, 1)
    speed_condition = speed <= desired_speed  # (N,)
    speed_reward = torch.where(speed_condition.unsqueeze(1),
                               1.0 * speed_2d,
                               -5.0 * (speed_2d - desired_speed))
    reward += speed_reward
    
    # 2. Lane deviation penalty (penalize offset from lane center)
    reward += -1.0 * lateral_offset.unsqueeze(1)
    
    # 3. Smooth driving penalty (lateral acceleration penalty)
    reward += -0.2 * torch.abs(a_lat).unsqueeze(1) * speed_2d ** 2
    
    # 3*. Smooth driving penalty (steer penalty)
    r_steer = -steer ** 2
    reward += 5.0 * r_steer.unsqueeze(1)
    
    # 4. Stationary penalty (已注释，不启用)
    # front_distance = state[:, 7]  # ego_state[7]: front_distance
    # stationary_mask = (front_distance > 10.0) & (speed < 0.1)
    # reward[stationary_mask] += -2.0
    
    # 5. Collision penalty
    collision_mask = (ego_collision > 0.5).unsqueeze(1)  # (N, 1)
    reward[collision_mask] += -200.0
    
    # 6. Off-road penalty
    off_road_mask = (ego_off_road > 0.5).unsqueeze(1)  # (N, 1)
    reward[off_road_mask] += -100.0
    
    # 7. Sparse terminal reward (已注释，不启用)
    # done_mask = done.squeeze(1) > 0.5
    # safe_done_mask = done_mask & (~collision_mask) & (~off_road_mask)
    # reward[safe_done_mask] += 200.0
    
    return reward

