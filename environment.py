import time
import random
import numpy as np
import pygame
from simulation.connection import carla
from simulation.sensors import CameraSensor, CameraSensorEnv, CollisionSensor
from simulation.settings import *
import math
from simulation.route_planner import RoutePlanner
from simulation.global_path_plan import global_path_planner
from scenarios.Ghost_probe import Ghost_probe_vehicle, Ghost_probe_walk
from scenarios.Slam_brake import Slam_brake_vehicle
from my_agents.tools.misc import draw_waypoints, distance_vehicle
from my_agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error


class CarlaEnvironment():
    def __init__(self, client, world, town, checkpoint_frequency=100, continuous_action=True, algorithm='ppo',
                 route_mode='1') -> None:

        self.client = client
        self.world = world
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.debug = self.world.debug  # type: carla.DebugHelper

        # 1. 速度与控制参数初始化（优先初始化，避免动作空间报错）
        self.target_speed = 25  # km/h
        self.max_speed = 30.0
        self.min_speed = 20.0
        self.velocity = float(0.0)
        self.max_distance_from_center = 2.5
        self.previous_throttle = float(0.8)  # 核心修改：初始油门设为高值，避免平滑稀释
        self.previous_steer = float(0.0)     # 初始方向为直行
        self.previous_brake = float(0.0)

        # 2. 动作空间初始化：强制优先直行，高油门
        self.action_space = self.get_init_action_space()
        self.num_actions = len(self.action_space)
        self.straight_action_idx = 2  # 直行动作固定索引（steer=0.0的位置）

        # 原有属性初始化
        self.continous_action_space = continuous_action
        self.display_on = VISUAL_DISPLAY
        self.vehicle = None
        self.settings = None
        self.current_waypoint_index = 0
        self.checkpoint_waypoint_index = 0
        self.fresh_start = True
        self.checkpoint_frequency = checkpoint_frequency
        self.route_waypoints = None
        self.town = town

        # Objects to be kept alive
        self.camera_obj = None
        self.env_camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None
        self.walker_flag = False
        self.global_route_plan = global_path_planner(world_map=self.map, sampling_resolution=2)  # 实例化全局规划器
        self.algorithm = algorithm
        self.route_mode = route_mode
        self.scenarios = "Random"

        # Two very important lists for keeping track of our actors and their observations.
        self.sensor_list = list()
        self.actor_list = list()
        self.walker_list = list()
        self.vehicle_list = list()

        # 新增：记录已通过的路径点索引和绘制ID（核心）
        self.passed_waypoint_indices = set()  # 标记已通过的路径点索引
        self.waypoint_draw_ids = {}  # 存储路径点的绘制ID（key:索引，value:draw_id）

    def get_init_action_space(self):
        """初始化动作空间：强制高油门，直行优先"""
        steer_options = [-0.5, -0.3, 0.0, 0.3, 0.5]
        action_space = []
        for steer in steer_options:
            if steer == 0.0:
                # 直行动作：起步阶段强制满油门
                action_space.append((steer, 1.0, 0.0))  # 油门拉满，无刹车
            elif steer < 0.0:
                # 左转动作：极低油门，避免误触发
                action_space.append((steer, 0.1, 0.25))
            else:
                # 右转动作：极低油门，避免误触发
                action_space.append((steer, 0.1, 0.25))
        return action_space

    def get_discrete_action_space(self):
        """运行时动态动作空间：起步阶段强制高油门"""
        steer_options = [-0.5, -0.3, 0.0, 0.3, 0.5]
        action_space = []

        current_speed = self.velocity
        max_speed = self.max_speed
        target_speed = self.target_speed

        for steer in steer_options:
            if steer == 0.0:
                # 核心：速度<10km/h时，强制满油门起步
                if current_speed < 10.0:
                    throttle = 0.6
                    brake = 0.0
                elif current_speed >= max_speed:
                    throttle = 0.0
                    brake = 0.3
                elif current_speed >= target_speed:
                    throttle = 0.4  # 仍保持高油门，避免掉速
                    brake = 0.0
                else:
                    throttle = min(1.0, max(0.5, (target_speed - current_speed)/target_speed * 0.8))
                    brake = 0.0
                action_space.append((steer, throttle, brake))
            else:
                action_space.append((steer, 0.2, 0.1))
        return action_space

    # A reset function for reseting our environment.
    def reset(self):
        # 新增：清空已通过的路径点索引和绘制ID（每次重置环境时初始化）
        self.passed_waypoint_indices.clear()
        self.waypoint_draw_ids.clear()

        # remove all actors
        if len(self.actor_list) != 0 or len(self.vehicle_list) != 0 or len(self.sensor_list) != 0:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicle_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.vehicle_list.clear()

        self.remove_sensors()

        # 每次环境reset会被重置
        if self.scenarios == "Ghost_probe":
            self.walker_flag = False
            self.set_ghost_scenarios()
        elif self.scenarios == "Random":
            self.set_other_vehicles()
        elif self.scenarios == "Slam_brake":
            self.slam_vehicle = None
            self.set_slam_scenarios()

        # Blueprint of our main vehicle
        vehicle_bp = self.get_vehicle(CAR_NAME)

        # 选择出生点和总距离
        if self.town == "Town07":
            transform = self.map.get_spawn_points()[38]
            self.total_distance = 750  # train max distance
        elif self.town == "Town02":
            transform = self.map.get_spawn_points()[28]
            self.total_distance = 780  # train max distance 333原来780
        elif self.town == "Town03":
            transform = self.map.get_spawn_points()[49]
            self.total_distance = 300  # train max distance
        else:
            # 随机选择初始位置训练，不好训，别用
            transform = random.choice(self.map.get_spawn_points())
            self.total_distance = 250  # train max distance

        self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
        self.actor_list.append(self.vehicle)

        # 选择目标点
        if self.town == "Town07":
            self.goal_location = self.map.get_spawn_points()[1]
        elif self.town == "Town02":
            self.goal_location = self.map.get_spawn_points()[0]
        elif self.town == "Town03":
            self.goal_location = self.map.get_spawn_points()[132]
        else:
            self.goal_location = random.choice(self.map.get_spawn_points())

        # Camera Sensor
        self.camera_obj = CameraSensor(self.vehicle)

        # Third person view of our vehicle in the Simulated env
        if self.display_on:
            self.env_camera_obj = CameraSensorEnv(self.vehicle)
            self.sensor_list.append(self.env_camera_obj.sensor)

        # Collision sensor
        self.collision_obj = CollisionSensor(self.vehicle)
        self.collision_history = self.collision_obj.collision_data
        self.sensor_list.append(self.collision_obj.sensor)

        # 重置运行时参数：强制初始油门为高值
        self.timesteps = 0
        self.rotation = self.vehicle.get_transform().rotation.yaw
        self.previous_location = self.vehicle.get_location()
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0

        self.velocity = float(0.0)
        self.distance_from_center = float(0.0)
        self.angle = float(0.0)
        self.center_lane_deviation = 0.0
        self.distance_covered = 0.0

        # 核心：重置后强制初始高油门，直行
        self.previous_throttle = float(0.8)
        self.previous_steer = float(0.0)
        self.previous_brake = float(0.0)

        if self.fresh_start:  # first run
            self.current_waypoint_index = 0

            # ppo保持原来的训练方法，resume时候会从最新的waypoint训练
            if self.route_mode == 'other':
                self.route_waypoints = list()
                # Waypoint nearby angle and distance from it
                self.waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True,
                                                      lane_type=(carla.LaneType.Driving))
                current_waypoint = self.waypoint
                self.route_waypoints.append(current_waypoint)
                for x in range(self.total_distance):
                    if self.town == "Town07":
                        if x < 650:
                            next_waypoint = current_waypoint.next(1.0)[0]
                        else:
                            next_waypoint = current_waypoint.next(1.0)[-1]
                    elif self.town == "Town02":
                        if x < 650:
                            next_waypoint = current_waypoint.next(1.0)[-1]
                        else:
                            next_waypoint = current_waypoint.next(1.0)[0]
                    else:
                        next_waypoint = current_waypoint.next(1.0)[0]

                    self.route_waypoints.append(next_waypoint)
                    current_waypoint = next_waypoint
                    # 新代码：保存draw_id，设置life_time=0.5
                    draw_id = self.debug.draw_point(
                        next_waypoint.transform.location + carla.Location(0, 0, 2),
                        size=0.05, color=carla.Color(0, 255, 0), life_time=0.5
                    )
                    self.waypoint_draw_ids[len(self.route_waypoints) - 1] = draw_id
            else:  # dqn sac ppo
                self.route_waypoints = self.global_route_plan.search_path_way(
                    origin=transform.location,
                    destination=self.goal_location.location)
                # 绘制路径点并保存draw_id
                for idx, waypoint in enumerate(self.route_waypoints):
                    draw_id = self.debug.draw_point(
                        waypoint.transform.location + carla.Location(0, 0, 2),
                        size=0.05, color=carla.Color(0, 255, 0), life_time=0.5
                    )
                    self.waypoint_draw_ids[idx] = draw_id
        else:
            # 绘制路径点并保存draw_id
            for idx, waypoint in enumerate(self.route_waypoints):
                draw_id = self.debug.draw_point(
                    waypoint.transform.location + carla.Location(0, 0, 2),
                    size=0.05, color=carla.Color(0, 255, 0), life_time=0.5
                )
                self.waypoint_draw_ids[idx] = draw_id
            # Teleport vehicle to last checkpoint
            waypoint = self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)]
            transform = waypoint.transform
            self.vehicle.set_transform(transform)
            self.current_waypoint_index = self.checkpoint_waypoint_index

        self.routeplanner = RoutePlanner(self.vehicle, 12)
        self.waypoints_routeplanner, _, self.vehicle_front = self.routeplanner.run_step()

        # 等待相机数据
        while len(self.camera_obj.front_camera) == 0:
            time.sleep(0.01)
        self.image_obs = self.camera_obj.front_camera.pop(-1)
        self.sensor_list.append(self.camera_obj.sensor)

        # 构建导航观测
        if self.algorithm == 'sac':
            transform = self.vehicle.get_transform()
            location = transform.location
            rotation = transform.rotation
            x_pos = location.x
            y_pos = location.y
            z_pos = location.z
            pitch = rotation.pitch
            yaw = rotation.yaw
            roll = rotation.roll
            acceleration = self.vector_to_scalar(self.vehicle.get_acceleration())
            angular_velocity = self.vector_to_scalar(self.vehicle.get_angular_velocity())
            velocity = self.vector_to_scalar(self.vehicle.get_velocity())
            self.navigation_obs = np.array([x_pos,
                                            y_pos,
                                            z_pos,
                                            pitch,
                                            yaw,
                                            roll,
                                            acceleration,
                                            angular_velocity,
                                            velocity], dtype=np.float64)
        else:
            # 新增：计算初始弯道距离
            distance_to_curve = self.get_distance_to_next_curve()
            self.navigation_obs = np.array(
                [self.previous_throttle, self.velocity, self.previous_steer, self.distance_from_center, self.angle,
                 distance_to_curve])

        # 核心：重置后立即施加高油门，强制起步
        self.vehicle.apply_control(carla.VehicleControl(
            steer=0.0,
            throttle=0.9,
            brake=0.0,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False
        ))
        time.sleep(0.5)  # 给车辆足够的起步时间
        self.collision_history.clear()
        self.episode_start_time = time.time()

        # 重置动作空间为初始化的高油门空间
        self.action_space = self.get_init_action_space()
        self.num_actions = len(self.action_space)

        return [self.image_obs, self.navigation_obs]

    # Step method is used for implementing actions taken by our agent
    def step(self, action_idx):
        self.timesteps += 1
        self.fresh_start = False

        # 处理鬼探头场景
        if self.scenarios == "Ghost_probe":
            TargetLocation = [-10.737004, -189.928192]
            current_position = np.array([self.walker_probe.get_location().x, self.walker_probe.get_location().y])
            dist_from_TargetLocation = np.linalg.norm(TargetLocation - current_position)
            vehicle_location = [self.vehicle.get_location().x, self.vehicle.get_location().y]
            dist_from_vehicle_location = np.linalg.norm(vehicle_location - current_position)

            if self.walker_flag is False and dist_from_vehicle_location < 8:
                pedestrain_control = carla.WalkerControl()
                pedestrain_control.speed = 1.0
                pedestrain_rotation = carla.Rotation(0, 90, 0)
                pedestrain_control.direction = pedestrain_rotation.get_forward_vector()
                self.walker_probe.apply_control(pedestrain_control)
                self.walker_flag = True

            if self.walker_flag is True and (dist_from_TargetLocation < 1.0):
                control = carla.WalkerControl()
                control.direction.x = 0
                control.direction.z = 0
                control.direction.y = 0
                self.walker_probe.apply_control(control)
                self.walker_flag = False

        # 处理急刹场景
        elif self.scenarios == "Slam_brake":
            current_position = np.array([self.slam_vehicle.get_location().x, self.slam_vehicle.get_location().y])
            vehicle_location = [self.vehicle.get_location().x, self.vehicle.get_location().y]
            dist_from_vehicle_location = np.linalg.norm(vehicle_location - current_position)
            if dist_from_vehicle_location < 3:
                self.agent_vehicle.set_speed(5)
            else:
                self.agent_vehicle.set_speed(10)

            control = self.agent_vehicle.run_step()
            control.manual_gear_shift = False
            self.slam_vehicle.apply_control(control)

        # 核心：先更新动作空间，再计算速度（避免首次step用旧空间）
        self.action_space = self.get_discrete_action_space()
        self.num_actions = len(self.action_space)

        # 计算当前速度（km/h）
        velocity = self.vehicle.get_velocity()
        self.velocity = np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) * 3.6

        # 执行动作
        if self.continous_action_space:
            if self.algorithm == 'ppo':
                steer = float(action_idx[0])
                steer = max(min(steer, 1.0), -1.0)
                throttle = float((action_idx[1] + 1.0) / 2)
                throttle = max(min(throttle, 1.0), 0.0)
                # 核心：起步阶段强制高油门，弱化平滑
                if self.velocity < 10.0:
                    smooth_steer = steer * 0.5 + self.previous_steer * 0.5
                    smooth_throttle = throttle * 0.8 + self.previous_throttle * 0.2
                else:
                    smooth_steer = self.previous_steer * 0.9 + steer * 0.1
                    smooth_throttle = self.previous_throttle * 0.9 + throttle * 0.1
                self.vehicle.apply_control(carla.VehicleControl(steer=smooth_steer,
                                                                throttle=smooth_throttle))
                self.previous_steer = steer
                self.previous_throttle = throttle
            elif self.algorithm == 'sac':
                # calculate actions
                steer = float(action_idx[0])
                steer = max(min(steer, 1.0), -1.0)
                throttle_brake = float((action_idx[1] + 1.0) / 2)
                throttle_brake = max(min(throttle_brake, 1.0), 0.0)

                if throttle_brake >= 0.0:
                    throttle = throttle_brake
                    brake = 0.0
                else:
                    throttle = 0.0
                    brake = -throttle_brake

                # 核心：起步阶段弱化平滑
                if self.velocity < 10.0:
                    smooth_steer = steer * 0.6 + self.previous_steer * 0.4
                    smooth_throttle = throttle * 0.8 + self.previous_throttle * 0.2
                    smooth_brake = brake * 0.2 + self.previous_brake * 0.8
                else:
                    smooth_steer = self.previous_steer * 0.9 + steer * 0.1
                    smooth_throttle = self.previous_throttle * 0.9 + throttle * 0.1
                    smooth_brake = self.previous_brake * 0.9 + brake * 0.1

                # apply control to simulation
                vehicle_control = carla.VehicleControl(
                    throttle=smooth_throttle,
                    steer=smooth_steer,
                    brake=smooth_brake,
                    hand_brake=False,
                    reverse=False,
                    manual_gear_shift=False
                )
                self.vehicle.apply_control(vehicle_control)
                self.previous_steer = steer
                self.previous_throttle = throttle
                self.previous_brake = brake
            else:
                steer = float(action_idx[0])
                steer = max(min(steer, 1.0), -1.0)
                throttle = float(action_idx[1])
                throttle = max(min(throttle, 1.0), 0.0)
                brake = float(action_idx[2])
                brake = max(min(brake, 1.0), 0.0)
                self.vehicle.apply_control(carla.VehicleControl(steer=steer, throttle=throttle, brake=brake))
                self.previous_steer = steer
                self.previous_throttle = throttle
                self.previous_brake = brake
        else:
            # 核心：起步阶段强制选择直行动作
            if self.velocity < 10.0:
                action_idx = self.straight_action_idx  # 强制直行
                # print(f"🚗 起步阶段：强制直行动作，当前速度{self.velocity:.1f}km/h")
            else:
                action_idx = action_idx % self.num_actions  # 正常索引

            steer, throttle, brake = self.action_space[action_idx]

            # 超速闭环修正（仅在超速时生效）
            if self.velocity > self.max_speed:
                throttle = 0.0
                brake = 0.5
                print(f"⚠️ 超速修正：当前速度{self.velocity:.1f}km/h → 断油+刹车0.5")
            elif self.velocity > self.target_speed:
                throttle = throttle * 0.9  # 轻微降低，不影响动力
                print(f"⚠️ 接近超速：当前速度{self.velocity:.1f}km/h → 油门调整为{throttle:.2f}")

            # 核心：起步阶段弱化平滑，保证油门有效
            if self.velocity < 10.0:
                smooth_steer = steer * 0.7 + self.previous_steer * 0.3
                smooth_throttle = throttle * 0.9 + self.previous_throttle * 0.1
                smooth_brake = brake * 0.1 + self.previous_brake * 0.9
            else:
                smooth_steer = self.previous_steer * 0.9 + steer * 0.1
                smooth_throttle = self.previous_throttle * 0.9 + throttle * 0.1
                smooth_brake = self.previous_brake * 0.9 + brake * 0.1

            # 应用控制
            self.vehicle.apply_control(carla.VehicleControl(
                steer=smooth_steer,
                throttle=smooth_throttle,
                brake=smooth_brake,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False
            ))

            # 更新历史控制参数
            self.previous_steer = smooth_steer
            self.previous_throttle = smooth_throttle
            self.previous_brake = smooth_brake

        # 处理交通灯
        if self.vehicle.is_at_traffic_light():
            traffic_light = self.vehicle.get_traffic_light()
            if traffic_light.get_state() == carla.TrafficLightState.Red:
                traffic_light.set_state(carla.TrafficLightState.Green)

        # 更新碰撞历史
        self.collision_history = self.collision_obj.collision_data

        # 更新车辆旋转和位置
        self.rotation = self.vehicle.get_transform().rotation.yaw
        self.location = self.vehicle.get_location()

        # 更新当前路径点索引
        waypoint_index = self.current_waypoint_index
        for _ in range(len(self.route_waypoints)):
            next_waypoint_index = waypoint_index + 1
            wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
            dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2],
                         self.vector(self.location - wp.transform.location)[:2])
            if dot > 0.0:
                waypoint_index += 1
            else:
                break
        self.current_waypoint_index = waypoint_index

        # 计算车道中心偏差
        self.current_waypoint = self.route_waypoints[self.current_waypoint_index % len(self.route_waypoints)]
        self.next_waypoint = self.route_waypoints[(self.current_waypoint_index + 1) % len(self.route_waypoints)]
        self.distance_from_center = self.distance_to_line(self.vector(self.current_waypoint.transform.location),
                                                          self.vector(self.next_waypoint.transform.location),
                                                          self.vector(self.location))
        self.center_lane_deviation += self.distance_from_center

        # 计算角度偏差
        fwd = self.vector(self.vehicle.get_velocity())
        wp_fwd = self.vector(self.current_waypoint.transform.rotation.get_forward_vector())
        self.angle = self.angle_diff(fwd, wp_fwd)

        # 更新检查点
        if not self.fresh_start and self.algorithm == 'other':
            if self.checkpoint_frequency is not None:
                self.checkpoint_waypoint_index = (
                                                     self.current_waypoint_index // self.checkpoint_frequency) * self.checkpoint_frequency

        # 计算到目标点的距离
        point1 = (self.goal_location.location.x, self.goal_location.location.y)
        point2 = (self.vehicle.get_location().x, self.vehicle.get_location().y)
        distance = self.euclidean_distance(point1, point2)

        # 奖励和结束条件计算
        done = False
        reward = 0

        # 碰撞惩罚
        if len(self.collision_history) != 0:
            print("Collision detected!")
            done = True
            reward = -5
        # 车道偏离惩罚
        elif self.distance_from_center > self.max_distance_from_center:
            print("Lane deviation exceeded!")
            done = True
            reward = -10
        # 低速超时惩罚：延长超时时间，给足够起步时间
        elif self.episode_start_time + 20 < time.time() and self.velocity < 1.0:
            print("Vehicle stuck at low speed!")
            reward = -5
            done = True
        # 超速惩罚
        elif self.velocity > self.max_speed:
            print(f"Over speed! Current speed: {self.velocity:.1f} km/h")
            reward = -5
            done = True
        # 到达目标奖励
        if distance < 0.8:
            print(f"Reach goal! Distance: {distance:.2f} m")
            reward = 20
            done = True

        # 计算奖励因子
        centering_factor = max(1.0 - self.distance_from_center / self.max_distance_from_center, 0.0)
        angle_factor = max(1.0 - abs(self.angle / np.deg2rad(40)), 0.0)
        goal_factor = max(1.0, 1.0 / abs(distance))

        # 提前转向惩罚（优化版：消除左转/右转偏置）
        current_steer = abs(self.previous_steer)
        curve_dist = self.get_distance_to_next_curve()
        normalized_curve_dist = curve_dist / 100.0

        # 核心优化1：动态调整惩罚阈值——弯道越远，允许的转向幅度越小
        max_allowed_steer = 0.1 + (normalized_curve_dist * 0.1)  # 远离弯道时，最大允许转向0.1；靠近弯道时，允许到0.2
        # 核心优化2：只惩罚“无必要的大幅转向”，而非所有远离弯道的转向
        if normalized_curve_dist > 0.3 and current_steer > max_allowed_steer:
            # 核心优化3：降低惩罚系数（从5.0→2.0），避免惩罚过重导致模型只敢选左转
            early_steer_penalty = current_steer * (1 - normalized_curve_dist) * 2.0
            reward -= early_steer_penalty
            print(
                f"⚠️ 提前转向惩罚：转向幅度{current_steer:.2f}，弯道距离{curve_dist:.1f}m，扣{early_steer_penalty:.2f}分")

        # 非结束状态的奖励计算
        if not done:
            if self.continous_action_space:
                if self.velocity < self.min_speed:
                    reward = (self.velocity / self.min_speed) * centering_factor * angle_factor * goal_factor
                elif self.velocity > self.target_speed:
                    reward = (1.0 - (self.velocity - self.target_speed) / (
                            self.max_speed - self.target_speed)) * centering_factor * angle_factor * goal_factor
                    # 超速额外惩罚
                    if self.velocity > self.max_speed:
                        speed_reward = -0.5
                        reward += speed_reward
                else:
                    reward = 1.0 * centering_factor * angle_factor * goal_factor
            else:
                # 离散动作奖励：强化超速惩罚
                if self.velocity < self.min_speed:
                    reward = (self.velocity / self.min_speed) * goal_factor
                elif self.velocity > self.target_speed:
                    speed_over = self.velocity - self.target_speed
                    speed_penalty = speed_over * 2.0
                    reward = (1.0 - (self.velocity - self.target_speed) / (
                            self.max_speed - self.target_speed)) * goal_factor - speed_penalty
                    if self.velocity > self.max_speed:
                        reward -= 10.0
                else:
                    reward = goal_factor

        # 超时结束
        if self.timesteps >= 7500:
            print("Episode timeout!")
            done = True
        # 路径点遍历完毕结束
        elif self.current_waypoint_index >= len(self.route_waypoints) - 2:
            print("All waypoints processed!")
            done = True
            self.fresh_start = True
            if self.checkpoint_frequency is not None and self.algorithm == 'other':
                if self.checkpoint_frequency < self.total_distance // 2:
                    self.checkpoint_frequency += 2
                else:
                    self.checkpoint_frequency = None
                    self.checkpoint_waypoint_index = 0

        # 等待相机数据
        while len(self.camera_obj.front_camera) == 0:
            time.sleep(0.0001)
        self.image_obs = self.camera_obj.front_camera.pop(-1)

        # 归一化观测值
        normalized_velocity = self.velocity / self.target_speed
        normalized_distance_from_center = self.distance_from_center / self.max_distance_from_center
        normalized_angle = abs(self.angle / np.deg2rad(40))

        # 构建导航观测
        if self.algorithm == 'sac':
            transform = self.vehicle.get_transform()
            location = transform.location
            rotation = transform.rotation
            x_pos = location.x
            y_pos = location.y
            z_pos = location.z
            pitch = rotation.pitch
            yaw = rotation.yaw
            roll = rotation.roll
            acceleration = self.vector_to_scalar(self.vehicle.get_acceleration())
            angular_velocity = self.vector_to_scalar(self.vehicle.get_angular_velocity())
            velocity = self.vector_to_scalar(self.vehicle.get_velocity())
            self.navigation_obs = np.array([x_pos,
                                            y_pos,
                                            z_pos,
                                            pitch,
                                            yaw,
                                            roll,
                                            acceleration,
                                            angular_velocity,
                                            velocity], dtype=np.float64)
        else:
            # 计算弯道距离
            distance_to_curve = self.get_distance_to_next_curve()
            normalized_curve_dist = distance_to_curve / 100.0
            normalized_curve_dist = max(min(normalized_curve_dist, 1.0), 0.0)

            self.navigation_obs = np.array(
                [self.previous_throttle, self.velocity, normalized_velocity, normalized_distance_from_center,
                 normalized_angle, normalized_curve_dist],
                dtype=float)

        # 结束时清理资源
        if done:
            self.center_lane_deviation = self.center_lane_deviation / self.timesteps
            self.distance_covered = abs(self.current_waypoint_index - self.checkpoint_waypoint_index)

            for sensor in self.sensor_list:
                sensor.destroy()

            self.remove_sensors()

            for actor in self.actor_list:
                actor.destroy()

        return [self.image_obs, self.navigation_obs], reward, done, [self.distance_covered,
                                                                     self.center_lane_deviation]

    # 计算到下一个弯道的距离
    def get_distance_to_next_curve(self):
        current_idx = self.current_waypoint_index
        max_search_steps = 100
        distance_to_curve = 0.0

        for i in range(1, max_search_steps + 1):
            next_idx = (current_idx + i) % len(self.route_waypoints)
            curr_wp_yaw = self.route_waypoints[current_idx].transform.rotation.yaw
            next_wp_yaw = self.route_waypoints[next_idx].transform.rotation.yaw
            yaw_diff = abs(curr_wp_yaw - next_wp_yaw)
            yaw_diff = min(yaw_diff, 360 - yaw_diff)

            if yaw_diff > 5.0:
                wp_location = self.route_waypoints[next_idx].transform.location
                vehicle_location = self.vehicle.get_location()
                distance_to_curve = self.euclidean_distance(
                    (vehicle_location.x, vehicle_location.y),
                    (wp_location.x, wp_location.y)
                )
                break
            distance_to_curve += 1.0

        return distance_to_curve

    # 创建行人
    def create_pedestrians(self):
        try:
            walker_spawn_points = []
            for i in range(NUMBER_OF_PEDESTRIAN):
                spawn_point_ = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if loc is not None:
                    spawn_point_.location = loc
                    walker_spawn_points.append(spawn_point_)

            for spawn_point_ in walker_spawn_points:
                walker_bp = random.choice(self.blueprint_library.filter('walker.pedestrian.*'))
                walker_controller_bp = self.blueprint_library.find('controller.ai.walker')
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                if walker_bp.has_attribute('speed'):
                    walker_bp.set_attribute('speed', (walker_bp.get_attribute('speed').recommended_values[1]))
                else:
                    walker_bp.set_attribute('speed', 0.0)
                walker = self.world.try_spawn_actor(walker_bp, spawn_point_)
                if walker is not None:
                    walker_controller = self.world.spawn_actor(
                        walker_controller_bp, carla.Transform(), walker)
                    self.walker_list.append(walker_controller.id)
                    self.walker_list.append(walker.id)
            all_actors = self.world.get_actors(self.walker_list)

            for i in range(0, len(self.walker_list), 2):
                all_actors[i].start()
                all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            print("NPC walkers have been generated in autopilot mode.")

        except Exception as e:
            print(f"Error creating pedestrians: {e}")
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.walker_list])

    # 生成其他车辆
    def set_other_vehicles(self):
        try:
            for _ in range(0, NUMBER_OF_VEHICLES):
                spawn_point = random.choice(self.map.get_spawn_points())
                bp_vehicle = random.choice(self.blueprint_library.filter('vehicle'))
                other_vehicle = self.world.try_spawn_actor(bp_vehicle, spawn_point)
                if other_vehicle is not None:
                    other_vehicle.set_autopilot(True)
                    self.vehicle_list.append(other_vehicle)
        except Exception as e:
            print(f"Error creating other vehicles: {e}")
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicle_list])

    # 设置鬼探头场景
    def set_ghost_scenarios(self):
        try:
            other_vehicle = Ghost_probe_vehicle(self.blueprint_library, self.world)
            self.vehicle_list.append(other_vehicle)
            self.walker_probe = Ghost_probe_walk(self.blueprint_library, self.world)
            self.actor_list.append(self.walker_probe)
        except Exception as e:
            print(f"Error setting ghost scenario: {e}")
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicle_list])

    # 设置急刹场景
    def set_slam_scenarios(self):
        try:
            self.slam_vehicle, self.agent_vehicle = Slam_brake_vehicle(self.blueprint_library, self.world)
        except Exception as e:
            print(f"Error setting slam brake scenario: {e}")

    # 切换城镇
    def change_town(self, new_town):
        self.world = self.client.load_world(new_town)

    # 获取世界
    def get_world(self) -> object:
        return self.world

    # 获取蓝图库
    def get_blueprint_library(self) -> object:
        return self.world.get_blueprint_library()

    # 计算角度差
    def angle_diff(self, v0, v1):
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle <= -np.pi:
            angle += 2 * np.pi
        return angle

    # 计算点到线段的距离
    def distance_to_line(self, A, B, p):
        num = np.linalg.norm(np.cross(B - A, A - p))
        denom = np.linalg.norm(B - A)
        if np.isclose(denom, 0):
            return np.linalg.norm(p - A)
        return num / denom

    # 转换carla向量到numpy数组
    def vector(self, v):
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])

    # 获取车辆蓝图
    def get_vehicle(self, vehicle_name):
        blueprint = self.blueprint_library.filter(vehicle_name)[0]
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        return blueprint

    # 生成车辆
    def set_vehicle(self, vehicle_bp, spawn_points):
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

    # 清理传感器
    def remove_sensors(self):
        self.camera_obj = None
        self.collision_obj = None
        self.lane_invasion_obj = None
        self.env_camera_obj = None
        self.front_camera = None
        self.collision_history = None
        self.wrong_maneuver = None

    # 计算欧氏距离
    def euclidean_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    # 向量转标量
    def vector_to_scalar(self, vector):
        scalar = np.around(np.sqrt(vector.x ** 2 +
                                   vector.y ** 2 +
                                   vector.z ** 2), 2)
        return scalar