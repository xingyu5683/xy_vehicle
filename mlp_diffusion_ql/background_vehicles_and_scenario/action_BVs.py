import random
import time
import carla
import math
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.controller import VehiclePIDController, PIDLongitudinalController
from agents.tools.misc import distance_vehicle
import numpy as np

class StableWaypointTracker:
    """
    适用于大时间步（0.1s）的稳定waypoint跟踪控制器
    结合Pure Pursuit和速度控制，对大时间步更鲁棒
    """
    def __init__(self, vehicle, dt=0.1, lookahead_base=3.0, min_lookahead=3.0, max_lookahead=10.0, wheelbase=2.9):
        """
        :param vehicle: Carla车辆对象
        :param dt: 时间步长（秒）
        :param lookahead_base: 基础前瞻距离（米）
        :param min_lookahead: 最小前瞻距离（米）
        :param max_lookahead: 最大前瞻距离（米）
        :param wheelbase: 车辆轴距（米）
        """
        self.vehicle = vehicle
        self.dt = dt
        self.lookahead_base = lookahead_base
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self.wheelbase = wheelbase
        
        # 速度控制参数（PI控制器）
        self.speed_K_P = 0.6
        self.speed_K_I = 0.1
        self.speed_integral = 0.0
        self.speed_integral_max = 2.0
        
        # 转向平滑参数（防止大时间步下的突变）
        self.past_steering = 0.0
        self.max_steering_change = 0.15 * (dt / 0.02)  # 根据dt调整最大转向变化率
        
    def find_lookahead_waypoint(self, waypoints, vehicle_transform, lookahead_distance):
        """
        在waypoints列表中找到前瞻点
        :param waypoints: waypoint列表 [(waypoint, option), ...]
        :param vehicle_transform: 车辆当前变换
        :param lookahead_distance: 前瞻距离
        :return: 前瞻waypoint，如果未找到则返回None
        """
        vehicle_location = vehicle_transform.location
        vehicle_forward = vehicle_transform.get_forward_vector()
        vehicle_forward_vec = np.array([vehicle_forward.x, vehicle_forward.y, 0.0])
        
        best_waypoint = None
        best_distance = float('inf')
        
        # 从当前waypoint_index开始查找
        for i, (wp, _) in enumerate(waypoints):
            wp_location = wp.transform.location
            to_waypoint = carla.Location(
                wp_location.x - vehicle_location.x,
                wp_location.y - vehicle_location.y,
                wp_location.z - vehicle_location.z
            )
            
            distance = math.sqrt(to_waypoint.x**2 + to_waypoint.y**2 + to_waypoint.z**2)
            
            if distance < 0.1:
                continue
            
            # 检查是否在前方
            to_waypoint_vec = np.array([to_waypoint.x, to_waypoint.y, 0.0])
            dot_product = np.dot(vehicle_forward_vec, to_waypoint_vec) / distance
            
            if dot_product > 0:  # 在前方
                # 找到最接近前瞻距离的点
                if abs(distance - lookahead_distance) < abs(best_distance - lookahead_distance):
                    best_distance = distance
                    best_waypoint = (wp, i)
        
        if best_waypoint and self.min_lookahead <= best_distance <= self.max_lookahead * 2:
            return best_waypoint[0]
        
        # 如果没找到合适的点，返回路径上最远的点
        if waypoints:
            return waypoints[-1][0]
        
        return None
    
    def calculate_steering(self, vehicle_transform, target_waypoint):
        """
        使用Pure Pursuit算法计算转向角
        :param vehicle_transform: 车辆当前变换
        :param target_waypoint: 目标waypoint
        :return: 转向角（-1到1之间）
        """
        vehicle_location = vehicle_transform.location
        vehicle_forward = vehicle_transform.get_forward_vector()
        target_location = target_waypoint.transform.location
        
        # 计算到目标点的向量
        dx = target_location.x - vehicle_location.x
        dy = target_location.y - vehicle_location.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 0.1:
            return 0.0
        
        # 计算车辆航向角（yaw）
        yaw_rad = math.radians(vehicle_transform.rotation.yaw)
        
        # 计算目标点的航向角
        target_yaw = math.atan2(dy, dx)
        
        # 计算角度差（归一化到[-pi, pi]）
        angle_diff = target_yaw - yaw_rad
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Pure Pursuit公式：steer = atan2(2 * wheelbase * sin(alpha), lookahead_distance)
        steering_angle = math.atan2(2.0 * self.wheelbase * math.sin(angle_diff), distance)
        
        # 归一化到[-1, 1]，假设最大转向角为±30度
        max_steering_angle = math.radians(30.0)
        steering = steering_angle / max_steering_angle
        steering = np.clip(steering, -1.0, 1.0)
        
        return steering
    
    def calculate_speed_control(self, current_speed, target_speed):
        """
        计算速度控制（PI控制器）
        :param current_speed: 当前速度（m/s）
        :param target_speed: 目标速度（m/s）
        :return: (throttle, brake) 元组
        """
        speed_error = target_speed - current_speed
        
        # 更新积分项
        self.speed_integral += speed_error * self.dt
        self.speed_integral = np.clip(self.speed_integral, -self.speed_integral_max, self.speed_integral_max)
        
        # PI控制
        control_output = self.speed_K_P * speed_error + self.speed_K_I * self.speed_integral
        
        if control_output > 0:
            throttle = np.clip(control_output, 0.0, 0.75)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(-control_output, 0.0, 0.3)
        
        return throttle, brake
    
    def run_step(self, target_speed, waypoints, vehicle_transform=None):
        """
        执行一步控制
        :param target_speed: 目标速度（m/s）
        :param waypoints: waypoint列表
        :param vehicle_transform: 车辆变换（如果为None则从vehicle获取）
        :return: carla.VehicleControl对象
        """
        if vehicle_transform is None:
            vehicle_transform = self.vehicle.get_transform()
        
        # 获取当前速度
        velocity = self.vehicle.get_velocity()
        current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # m/s
        
        # 根据速度自适应调整前瞻距离
        lookahead_distance = self.lookahead_base * (1.0 + current_speed / 10.0)
        lookahead_distance = np.clip(lookahead_distance, self.min_lookahead, self.max_lookahead)
        
        # 找到前瞻点
        lookahead_waypoint = self.find_lookahead_waypoint(waypoints, vehicle_transform, lookahead_distance)
        
        if lookahead_waypoint is None:
            # 如果没有找到前瞻点，使用当前waypoint
            if waypoints:
                lookahead_waypoint = waypoints[0][0]
            else:
                # 如果没有waypoint，保持当前状态
                control = carla.VehicleControl()
                control.throttle = 0.0
                control.brake = 0.3
                control.steer = 0.0
                return control
        
        # 计算转向角
        steering = self.calculate_steering(vehicle_transform, lookahead_waypoint)
        
        # 转向平滑（防止大时间步下的突变）
        steering_change = steering - self.past_steering
        if abs(steering_change) > self.max_steering_change:
            steering = self.past_steering + np.sign(steering_change) * self.max_steering_change
        
        self.past_steering = steering
        
        # 计算速度控制
        throttle, brake = self.calculate_speed_control(current_speed, target_speed)
        
        # 创建控制命令
        control = carla.VehicleControl()
        control.throttle = throttle
        control.brake = brake
        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        
        return control


class ActionBVBase:
    """
    This is the base class for the background vehicles with specified actions (Action BV)

    """
    def __init__(self, world):
        """
        """
        self.world = world
        self.map = self.world.get_map()
        self.lat_K_P = 4
        self.lat_K_D = 1
        self.lat_K_I = 0.1
        self.dt = 1/10
        self.lon_K_P = 1
        self.lon_K_D = 0.05
        self.lon_K_I = 0

    def reset(self):
        pass

    def take_actions(self):
        pass

    # Define General Functions
    def cal_target_route(self, vehicle, lanechange, target_dis=20): # TODO: not only lane change
        # 实例化道路规划模块
        # sampling_resolution = random.uniform(2, 3)  # 在2-3之间随机选取sampling_resolution
        grp = GlobalRoutePlanner(self.map, 2)  # sampling_resolution表示规划路径的分辨率，单位为m，表示waypoint的间隔
        # 获取npc车辆当前所在的waypoint
        current_location = vehicle.get_transform().location
        current_waypoint = self.map.get_waypoint(current_location) #当前位置的最近路点
        # 选择变道方向
        if "left" in lanechange:
            target_org_waypoint = current_waypoint.get_left_lane()
        elif "right" in lanechange:
            target_org_waypoint = current_waypoint.get_right_lane()
   # 3. 目标点（终点）
        target_location = target_org_waypoint.next(target_dis)[0].transform.location

        # 第一段：当前点 -> 变道完成点
        route = grp.trace_route(current_location, target_location)


        return route

    def draw_target_line(self, waypoints):
        # 获取世界和调试助手
        debug = self.world.debug
        # 设置绘制参数
        life_time = 20.0  # 点和线将持续显示的时间（秒）
        color = carla.Color(255, 0, 0)
        thickness = 0.3  # 线的厚度
        for i in range(len(waypoints) - 1):
            debug.draw_line(waypoints[i][0].transform.location + carla.Location(z=0.5),
                            waypoints[i + 1][0].transform.location + carla.Location(z=0.5),
                            thickness=thickness,
                            color=color,
                            life_time=life_time)

    def draw_current_point(self, current_point):
        self.world.debug.draw_point(current_point, size=0.1, color=carla.Color(b=255), life_time=20)

    def speed_con_by_pid(self, vehicle=None, pid=None, target_speed=30):
        control_signal = pid.run_step(target_speed=target_speed, debug=False)
        throttle = max(min(control_signal, 1.0), 0.0)  # 确保油门值在0到1之间
        brake = 0.0  # 根据需要设置刹车值
        if control_signal < 0:
            throttle = 0.0
            brake = abs(control_signal)  # 假设控制器输出的负值可以用来刹车
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake))



class ActionBVNone(ActionBVBase):
    def reset(self, ego):
        pass

    def take_actions(self):
        pass

class ActionBVACC(ActionBVBase):
    def reset(self, ego):
       pass

    def take_actions(self):
       pass


class ActionBVUnprotLeft(ActionBVBase):
    def reset(self, ego):
        pass

    def take_actions(self):
        pass

class ActionBVUnprotRight(ActionBVBase):
    def reset(self, ego):
        pass

    def take_actions(self):
        pass

class ActionBVLaneChangeIn(ActionBVBase):
    def reset(self, ego):
        # BV 的 spawn 位置
        pass

    def take_actions(self):
        pass

class ActionBVBeingCutIn(ActionBVBase):
    def reset(self, ego):
        # 变道需要的参数(是否需要加条件判断，是否需要移动位置)
        self.waypoint_index = 0
        self.cut_in_flag = False
        self.need_cal_route = True
        self.target_distance_threshold = 2.0  # 切换waypoint的距离010
        self.arrive_target_point = False
        self.start_sim_time = time.time()
        self.ego = ego
        self.back_dist = 10
        self.over = 0

        self.lanechange = random.choice(["left", "right"])  # 随机选择左或右，概率为50%
        self.dis_to_cut = random.uniform(4.3, 7)  # 在5-8之间随机选取

        # bv_spawn_point = carla.Transform(
        #     carla.Location(x=138.5, y=248.25, z=1.0),  # 位置
        #     carla.Rotation(pitch=0.0, yaw=00.0, roll=0.0)  # 旋转
        # )
        ego_wp = self.map.get_waypoint(self.ego.get_location())
        adj_wp = ego_wp.get_right_lane() if self.lanechange=="left" else ego_wp.get_left_lane()
        if (adj_wp is None):
            if self.over == 1:
                return None
            if self.lanechange == "left":
                self.lanechange = "right"
                self.over = 1
            elif self.lanechange == "right":
                self.lanechange = "left"
                self.over = 1

        prevs = adj_wp.previous(self.back_dist)
        if not prevs:
            return None#TODO:if return None, random generate ego in carla_env

        spawn_wp = prevs[0]
        bv_spawn_point = carla.Transform(spawn_wp.transform.location, spawn_wp.transform.rotation)
        bv_spawn_point.location.z=1
        # 获取目标车辆蓝图(可以改为随机)
        vehicle_bp = self.world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
        vehicle_bp.set_attribute('color', '0,0,255')

        self.npc_vehicle = self.world.try_spawn_actor(vehicle_bp, bv_spawn_point)

        # 设定周车的pid控制器（用于初始速度控制）
        self.initspd_pid = PIDLongitudinalController(self.npc_vehicle, K_P=self.lon_K_P, K_I=self.lon_K_I, K_D=self.lon_K_D)

        # 使用稳定的waypoint跟踪控制器，适合0.1s时间步
        # 根据dt调整前瞻距离：dt越大，前瞻距离应该越大
        lookahead_base = 3.6  # 相对于0.02的比例调整
        # print("lookahead_base", lookahead_base)
        self.waypoint_tracker = StableWaypointTracker(
            self.npc_vehicle,
            dt=self.dt,
            lookahead_base=lookahead_base,
            min_lookahead=3.0,
            max_lookahead=10.0,
            wheelbase=2.9  # Tesla Model 3轴距
        )
        
        # 保留PID作为备选（如果需要可以切换）
        args_lateral_dict = {'K_P': self.lat_K_P, 'K_D': self.lat_K_D, 'K_I': self.lat_K_I, 'dt': self.dt}
        args_long_dict = {'K_P': self.lon_K_P, 'K_D': self.lon_K_D, 'K_I': self.lon_K_I, 'dt': self.dt}
        self.PID = VehiclePIDController(self.npc_vehicle, args_lateral_dict, args_long_dict)
        
        # 控制器选择：'stable_tracker' 或 'pid'
        self.controller_type = 'stable_tracker'

        return self.npc_vehicle

    def take_actions(self):
        ego_speed = (self.ego.get_velocity().x * 3.6)  # km/h 世界 X 方向速度 #######
        target_speed = ego_speed + 8  # 目标车的目标速度

        # 是否满足cut_in条件
        if self.cut_in_flag:
            if self.need_cal_route:
                self.waypoints = self.cal_target_route(self.npc_vehicle, self.lanechange, target_dis=150)
                self.draw_target_line(self.waypoints)
                self.need_cal_route = False

            # 如果已经计算了路线
            if self.waypoints is not None and self.waypoint_index < len(self.waypoints):
                # 获取当前目标路点
                target_waypoint = self.waypoints[self.waypoint_index][0]
                # 获取车辆当前位置
                transform = self.npc_vehicle.get_transform()
                # 绘制当前运行的点
                self.draw_current_point(transform.location)
                # 计算车辆与当前目标路点的距离
                distance_to_waypoint = distance_vehicle(target_waypoint, transform)
                if distance_to_waypoint < self.target_distance_threshold:
                    self.waypoint_index += 1  # 移动到下一个路点
                    if self.waypoint_index >= len(self.waypoints):
                        self.arrive_target_point = True
                        print("npc_vehicle had arrive target point.")

                else:
                    # 计算控制命令
                    if self.controller_type == 'stable_tracker':
                        # 使用稳定的waypoint跟踪控制器
                        # 将目标速度从km/h转换为m/s
                        target_speed_ms = target_speed / 3.6
                        # 获取从当前waypoint_index开始的剩余waypoints
                        remaining_waypoints = self.waypoints[self.waypoint_index:]
                        vehicle_transform = self.npc_vehicle.get_transform()
                        control = self.waypoint_tracker.run_step(target_speed_ms, remaining_waypoints, vehicle_transform)
                    else:
                        # 使用传统PID控制器
                        control = self.PID.run_step(target_speed, target_waypoint)
                    
                    # 应用控制命令
                    self.npc_vehicle.apply_control(control)
        else:
            # 设置NPC的初始速度
            self.speed_con_by_pid(self.npc_vehicle, self.initspd_pid, target_speed)
            # 判断是否可以cut in（使用reset时生成的随机值）
            # print("dis_to_cut", self.dis_to_cut)
            self.cut_in_flag = self._should_cut_in(self.npc_vehicle, self.ego, dis_to_cut=self.dis_to_cut)

            # 判断是否达到模拟时长
            if time.time() - self.start_sim_time > 60:
                print("[Being Cut In] Simulation ended due to time limit.")
                self.arrive_target_point = True #其实是到达时长

        return self.arrive_target_point

    def _should_cut_in(self, npc_vehicle, ego_vehicle, dis_to_cut):
        location1 = npc_vehicle.get_transform().location
        location2 = ego_vehicle.get_transform().location
        rel_x = location1.x - location2.x
        rel_y = location1.y - location2.y
        distance = math.sqrt(rel_x * rel_x + rel_y * rel_y)
        # print("relative dis", distance)
        if rel_x >= 0:
            distance = distance
        else:
            distance = -distance
        if distance >= dis_to_cut:
            print("The conditions for changing lanes are met.")
            cut_in_flag = True
        else:
            cut_in_flag = False
        return cut_in_flag
