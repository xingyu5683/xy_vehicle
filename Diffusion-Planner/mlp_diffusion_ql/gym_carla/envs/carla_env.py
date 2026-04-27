import gymnasium as gym
import carla
import numpy as np
import random
import time
import math
from background_vehicles_and_scenario.BVs_manager import BV_manager
from background_vehicles_and_scenario.scenario_manager import ScenarioManager
from agents.navigation.global_route_planner import GlobalRoutePlanner



class CarlaEnv(gym.Env):
    def __init__(self, env_params):
        self.collision_sensor = None
        self.dt = self.dt = env_params['dt']
        self.max_ego_spawn_times = env_params['max_ego_spawn_times']
        self.surrounding_vehicle_spawned_randomly = env_params['surrounding_vehicle_spawned_randomly']
        self.number_of_vehicles = env_params['number_of_vehicles']
        self.perception_range = env_params['perception_range']
        self.max_nearby_vehicles = env_params['max_nearby_vehicles']
        self.max_waypoints = env_params['max_waypoints']
        self.visualize_waypoints = env_params['visualize_waypoints']
        self.view_mode = env_params['view_mode']
        self.max_time_episode = env_params['max_time_episode']
        self.desired_speed = env_params['desired_speed']
        self.case_id = env_params['case_id']
        self.traffic = 'off'
        self.render_mode = None

        # info初始化
        self._ego_collision = False
        self._ego_off_road = False
        self._ego_min_dis = None

        obs_low = np.array([  # 合理上界
            150.0,  # ego_speed km/h
            150.0,  # dist_front m
            100.0,  # rel_speed km/h
            5.0,  # lane_offset m
            180.0,  # yaw_error deg
        ], dtype=np.float32)

        obs_high = np.array([  # 合理上界
            150.0,   # ego_speed km/h
            150.0,   # dist_front m
            100.0,   # rel_speed km/h
            5.0,     # lane_offset m
            180.0,   # yaw_error deg
        ], dtype=np.float32)

        self.observation_space = gym.spaces.Dict({
            'ego_state': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32),
            'nearby_vehicles': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32),
            'waypoints': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32),
            'lane_info': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        })

        #obs_space初始化
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32)
        )

        print('Connecting to Carla server...')
        client = carla.Client('localhost', env_params['port'])
        client.set_timeout(10.0)
        self.world = client.load_world(env_params['town'])
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        print('Connection established!')

        self.world_map = self.world.get_map()

        self.grp = GlobalRoutePlanner(self.world_map, 0.2)

        # 获取车辆重生点
        # self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())

        # 获取蓝图：车辆及传感器
        self.ego_bp = self._create_vehicle_bluepprint(env_params['ego_vehicle_filter'], color='255,0,0')

        self.collision_hist = []  # 碰撞事件列表
        self.collision_hist_l = 1  # 碰撞事件长度阈值
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # 获取世界设置、设置仿真步长
        self.settings = self.world.get_settings()  # Get the current world settings
        self.settings.fixed_delta_seconds = self.dt  # Set the physics simulation step size (in seconds)
                                                     # This ensures consistent time intervals for simulation updates

        #BV_controller实例化
        self.bv_manager = BV_manager(case_id=self.case_id, world=self.world, number_of_vehicles=self.number_of_vehicles)
        self.scenario_manager = ScenarioManager(total_episodes=env_params['total_episodes'])

        self.reset_step = 0
        self.total_step = 0
        self.current_episode = 0
        

    def reset(self, seed=None, options=None):
        # select
        self.current_episode += 1
        self.scenario_type, self.scenario_style = self.scenario_manager.sample_scene(self.current_episode)
        # scenario_type, scenario_style 在这里采样

        # Stop and destroy the collision sensor if it exists
        if self.collision_sensor is not None:
            try:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()
            except:
                pass
            self.collision_sensor = None

        self._clear_all_actors([
            # 'sensor.other.collision',
            # 'sensor.lidar.ray_cast',
            'sensor.camera.rgb',
            'vehicle.*',
            # 'controller.ai.walker',
            # 'walker.*'
        ])  # Remove all specified actors from the world

        # reset info
        self._ego_collision = False
        self._ego_off_road = False
        self._ego_min_dis = None

        self._set_synchronous_mode(False)  # Switch back to asynchronous mode 以便更改世界设置

        # 车辆边界框记录
        self.vehicle_polygons = []

# ==================================ego spawn============

        # 在重生点生成自车
        ego_spawn_times = 0
        while True:
            if ego_spawn_times > self.max_ego_spawn_times:
                self.reset()  # If failed too many times, reset the environment

            #town 6
            if self.scenario_type == "Normal":
                spawn_points = self.world.get_map().get_spawn_points()
                # 随机挑选一个 spawn 点
                ego_spawn_point = random.choice(spawn_points)
                # ego_spawn_point = carla.Transform(
                #     carla.Location(x=148.5, y=244.5, z=1.0),  # 位置
                #     carla.Rotation(pitch=0.0, yaw=00.0, roll=0.0)  # 旋转
                # )
            elif self.scenario_type == "ACC":
                ego_spawn_point = carla.Transform(
                    carla.Location(x=148.5, y=244.5, z=1.0),  # 位置
                    carla.Rotation(pitch=0.0, yaw=00.0, roll=0.0)  # 旋转
                )
                self.need_cal_route = True
                self.initial_lane_waypoint = None  # 存储ego初始位置所在车道的waypoint
            elif self.scenario_type == "UnprotLeft": #Town05
                ego_spawn_point = carla.Transform(
                    carla.Location(x=-80, y=2.75, z=1.0),  # 位置
                    carla.Rotation(pitch=0.0, yaw=0, roll=0.0)  # 旋转
                )
                #对向抢行，左侧抢行，右侧抢行
            elif self.scenario_type == "UnprotRight":
                ego_spawn_point = carla.Transform(
                    carla.Location(x=-80, y=6.25, z=1.0),  # 位置
                    carla.Rotation(pitch=0.0, yaw=0, roll=0.0)  # 旋转
                )
                #右侧抢行： Transform(Location(x=-121.022667, y=22.534832, z=0.000000), Rotation(pitch=360.000000, yaw=269.488617, roll=0.000000))
            elif self.scenario_type == "LaneChangeIn":
                ego_spawn_point = carla.Transform(
                    carla.Location(x=148.5, y=241, z=1.0),  # 位置
                    carla.Rotation(pitch=0.0, yaw=00.0, roll=0.0)  # 旋转
                )
                self.need_cal_route = True
                self.over = False
            elif self.scenario_type == "BeingCutIn":
                ego_spawn_point = carla.Transform(
                    carla.Location(x=148.5, y=244.5, z=1.0),  # 位置
                    carla.Rotation(pitch=0.0, yaw=00.0, roll=0.0)  # 旋转
                )
                self.need_cal_route = True
                self.initial_lane_waypoint = None  # 存储ego初始位置所在车道的waypoint

            # Try to spawn the ego vehicle at the selected location
            if self._try_spawn_ego_vehicle_at(ego_spawn_point): #不检测生成时是否有重叠
                break  # Successfully spawned the ego vehicle
            else:
                ego_spawn_times += 1  # Retry counter
                time.sleep(0.1)  # Small delay before retrying
        # respawn background vehicles
        self.ABV = self.bv_manager.reset(self.ego, self.scenario_type, self.scenario_style)
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')  # 返回actor id和四个边界点
        self.vehicle_polygons.append(vehicle_poly_dict)

        # 设置交通灯
        if self.traffic == 'off':
            # Set all traffic lights to green and freeze them
            for actor in self.world.get_actors().filter('traffic.traffic_light*'):
                actor.set_state(carla.TrafficLightState.Green)
                actor.freeze(True)
        elif self.traffic == 'on':
            # Allow traffic lights to work normally
            for actor in self.world.get_actors().filter('traffic.traffic_light*'):
                actor.freeze(False)

        # 生成碰撞传感器
        self.collision_sensor = self.world.spawn_actor(
            self.collision_bp,
            carla.Transform(),  # Attach at the center of the ego vehicle (no offset)
            attach_to=self.ego
        )

        # Start listening for collision events
        self.collision_sensor.listen(
            lambda event: get_collision_hist(event)
            # When a collision event happens, pass the event to get_collision_hist()
        )
        # 计算撞击冲量并加入到撞击信息序列中
        def get_collision_hist(event):
            impulse = event.normal_impulse  # Get the collision impulse (a 3D vector)
            intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)  # Calculate collision intensity (vector norm)
            self.collision_hist.append(intensity)  # Record the collision intensity
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)  # Keep only the latest collision records (FIFO)

        self.collision_hist = []

        # Update timesteps
        self.time_step = 1  # Indicates a new episode has started
        self.reset_step += 1  # Count how many resets have occurred

        self._set_synchronous_mode(True)  # Switch to synchronous mode for simulation
        self.world.tick()  # Advance the simulation by one tick
        
        # 记录ego初始位置所在车道的waypoint
        self.initial_lane_waypoint = self.world_map.get_waypoint(self.ego.get_location())

        info = {
            'ego_collision': self._ego_collision,
            'ego_off_road': self._ego_off_road,
            'ego_min_dis': self._ego_min_dis
        }

        return self._get_obs(), info  # Return the initial observation and info after reset


    def step(self, action):
        '''
        第一步，如何将动作值映射到Carla的控制中
        第二步，执行了一步动作，如何通过传感器得到下一步的状态
        第三步，如何根据当前的状态，计算出奖励值并且返回数值
        第四步，返回这个DONE的布尔值，比如如果相撞了，那么直接为True，或者完成了任务，也为True。

        :param action: longitudinal, steer
        :return: next_obs, reward, done, truncated, info
        '''
        longitudinal = float(np.clip(action[0], -1.0, 1.0))
        steer = float(np.clip(action[1], -1.0, 1.0))
        if longitudinal >= 0:
            throttle = longitudinal
            brake = 0.0
        else:
            throttle = 0.0
            brake = -longitudinal  # 转为正数

        # Apply control
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self.ego.apply_control(control)

        # 周车apply Controls
        self.bv_done = self.bv_manager.take_actions()

        self.world.tick()

        # 设定观察者视角
        spectator = self.world.get_spectator()
        transform = self.ego.get_transform()

        if self.view_mode == 'top':
            # Top-down view (bird's eye)
            spectator.set_transform(
                carla.Transform(
                    transform.location + carla.Location(z=40),
                    carla.Rotation(pitch=-90)
                )
            )
        elif self.view_mode == 'follow':
            # Follow view (behind and above the ego vehicle)
            cam_location = transform.transform(carla.Location(x=-6.0, z=3.0))  # 6 meters behind, 3 meters above
            cam_rotation = carla.Rotation(pitch=-10, yaw=transform.rotation.yaw, roll=0)
            spectator.set_transform(carla.Transform(cam_location, cam_rotation))

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        obs = self._get_obs()
        done = self._terminal(obs)
        reward = self._get_reward(obs, done)
        # cost = self._get_cost(obs)


        # return next_obs, reward, done, None
        info = {
            'ego_collision': self._ego_collision,
            'ego_off_road': self._ego_off_road,
            'ego_min_dis': self._ego_min_dis
        }

        truncated = bool(self.time_step >= self.max_time_episode)
        return (obs, reward, done, truncated, info)
        # return (obs, reward, done, info)


    def _vector_to_scalar(vector):
        scalar = np.around(np.sqrt(vector.x ** 2 +
                                   vector.y ** 2 +
                                   vector.z ** 2), 2)
        return scalar


    def _get_obs(self):
        obs = {}
    # ========================== Ego vehicle state extraction =======================================
        # 获取自车位置和航向角
        ego_transform = self.ego.get_transform()
        ego_x = ego_transform.location.x
        ego_y = ego_transform.location.y
        ego_yaw = np.deg2rad(ego_transform.rotation.yaw)

        # 自车状态提取
        velocity = self.ego.get_velocity()
        speed = np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        angular_velocity = self.ego.get_angular_velocity()
        acceleration = self.ego.get_acceleration()

        front_vehicle_distance = 0.0
        relative_speed = 0.0

        min_front_distance = 20.0  # Search range threshold
        vehicle_list = self.world.get_actors().filter('vehicle.*')

        # 在周围车辆中，找到前方距离最近的车辆，并获取与ego的距离和相对速度
        for vehicle in vehicle_list:
            if vehicle.id == self.ego.id:
                continue

            transform = vehicle.get_transform()
            rel_x = transform.location.x - ego_x
            rel_y = transform.location.y - ego_y  # 被减向量指向减向量，ego指向tran

            local_x = np.cos(-ego_yaw) * rel_x - np.sin(-ego_yaw) * rel_y
            local_y = np.sin(-ego_yaw) * rel_x + np.cos(-ego_yaw) * rel_y  # 换到自车坐标系

            if 0 < local_x < min_front_distance and abs(local_y) < 2:  # 正（侧）前方同车道很近的车辆
                d = np.sqrt(local_x ** 2 + local_y ** 2)
                if front_vehicle_distance == 0.0 or d < front_vehicle_distance:  # 获取当前时刻前方车辆与自车最近的距离
                    front_vehicle_distance = d
                    front_speed = vehicle.get_velocity()
                    front_speed_mag = np.sqrt(front_speed.x ** 2 + front_speed.y ** 2 + front_speed.z ** 2)
                    relative_speed = speed - front_speed_mag

            if front_vehicle_distance == 0.0:
                front_vehicle_distance = 20 #相当于是前方无车的标志位

        # 打包自车状态，9维
        ego_state = np.array([
            ego_x,
            ego_y,
            ego_yaw,
            speed,
            angular_velocity.z,
            acceleration.x,
            acceleration.y,
            front_vehicle_distance,
            relative_speed
        ], dtype=np.float32)

        obs['ego_state'] = ego_state

    # ================ Nearby vehicles state extraction (up to 5 vehicles, within perception range) ===============
        # 周围车辆状态提取
        max_vehicles = self.max_nearby_vehicles
        perception_range = self.perception_range
        vehicle_list = self.world.get_actors().filter('vehicle.*')

        # 返回周围车辆的[local_x, local_y, yaw - ego_yaw, speed]，维数为4*5展平，按与自车距离升序排列
        vehicle_data = []
        for vehicle in vehicle_list:
            if vehicle.id == self.ego.id:
                continue  # Skip the ego vehicle itself

            transform = vehicle.get_transform()
            x = transform.location.x
            y = transform.location.y
            yaw = np.deg2rad(transform.rotation.yaw)

            rel_x = x - ego_x
            rel_y = y - ego_y

            distance = np.sqrt(rel_x ** 2 + rel_y ** 2)
            if distance > perception_range:
                continue  # Ignore vehicles outside the perception range

            # Transform to 自车坐标系
            local_x = np.cos(-ego_yaw) * rel_x - np.sin(-ego_yaw) * rel_y
            local_y = np.sin(-ego_yaw) * rel_x + np.cos(-ego_yaw) * rel_y

            v = vehicle.get_velocity()
            speed = np.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

            vehicle_data.append((distance, [local_x, local_y, yaw - ego_yaw, speed]))

        # Sort vehicles by distance and select the nearest max_vehicles
        vehicle_data.sort(key=lambda x: x[0])  # 距离按升序排列
        nearby_vehicles = [data for _, data in vehicle_data[:max_vehicles]]

        # Pad with zeros if fewer than max_vehicles are detected
        while len(nearby_vehicles) < max_vehicles:
            nearby_vehicles.append([0.0, 0.0, 0.0, 0.0])

        obs['nearby_vehicles'] = np.array(nearby_vehicles,
                                          dtype=np.float32).flatten()  # [local_x, local_y, yaw - ego_yaw, speed]

    # ========================== Current reference waypoints (up to N waypoints) ==========================
        max_waypoints = self.max_waypoints

        if self.scenario_type == "Normal":
            waypoints_array = self._get_next_waypoint_random(max_waypoints, ego_yaw, ego_x, ego_y)
        if self.scenario_type == "UnprotLeft":
            waypoints_array = self._get_next_waypoint_left_turn(max_waypoints, ego_yaw, ego_x, ego_y)
        if self.scenario_type == "UnprotRight":
            waypoints_array = self._get_next_waypoint_right_turn(max_waypoints, ego_yaw, ego_x, ego_y)
        if self.scenario_type == "LaneChangeIn":
            waypoints_array = self._get_next_waypoint_lanechange(max_waypoints, ego_yaw, ego_x, ego_y)
        if self.scenario_type == "BeingCutIn" or self.scenario_type == "ACC":
            waypoints_array = self._get_next_waypoint_straight(max_waypoints, ego_yaw, ego_x, ego_y)

        obs['waypoints'] = waypoints_array.flatten()  # 返回沿车道的后续每间隔2m路点[local_x, local_y, yaw_relative]，维数为12*3展平

    # ============================= Lane boundary information =========================================
        waypoint_center = self.world_map.get_waypoint(
            self.ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
        )  # 将自车位置投影到最近的可行驶车道中心线上，并返回该处的waypoint

        if waypoint_center is None:
            # If no valid driving lane is found
            obs['lane_info'] = np.array([0.0, 0.0], dtype=np.float32)
        else:
            lane_width = waypoint_center.lane_width

            ego_location = self.ego.get_location()
            center_location = waypoint_center.transform.location

            # Calculate lateral offset between ego position and lane centerline
            # lateral_offset = np.linalg.norm(  # Carla路点是非常密集的,所以与最近点的距离就是横向偏离距离,这样也可以覆盖弯道情况
            #     np.array([
            #         ego_location.x - center_location.x,
            #         ego_location.y - center_location.y
            #     ])
            # )
            lateral_offset = abs(waypoints_array[0][1])
            # print(lateral_offset)
            # 返回车道宽度和横向偏离车道距离
            obs['lane_info'] = np.array([lane_width, lateral_offset], dtype=np.float32)

    # =============================== Visualize current reference waypoints ===============================
        if self.visualize_waypoints:
            for i in range(max_waypoints):
                wx, wy, _ = waypoints_array[i]

                # Transform from ego-centric local coordinates to global coordinates 转换回全局坐标系
                gx = np.cos(ego_yaw) * wx - np.sin(ego_yaw) * wy + ego_x
                gy = np.sin(ego_yaw) * wx + np.cos(ego_yaw) * wy + ego_y

                self.world.debug.draw_point(
                    carla.Location(x=gx, y=gy, z=ego_transform.location.z + 1.0),
                    size=0.1,
                    life_time=0.2,  # 0.5
                    color=carla.Color(0, 255, 0)  # Green points
                )

        return obs


    def _terminal(self, obs):
        ego_transform = self.ego.get_transform()
        ego_x = ego_transform.location.x
        ego_y = ego_transform.location.y

        # 1. Collision termination
        if len(self.collision_hist) > 0:
            self._ego_collision = True
            print('Collision occurred')
            return True

        # 2. Exceeding maximum allowed timesteps
        if self.time_step > self.max_time_episode:
            print('Exceeded maximum timesteps')
            return True

        # # 3. Goal reaching termination (optional)
        # if self.dests is not None:
        #     for dest in self.dests:
        #         if np.sqrt((ego_x - dest[0])**2 + (ego_y - dest[1])**2) < 4:
        #             return True

        # 4. Check if the current lane is a drivable lane
        waypoint = self.world.get_map().get_waypoint(
            self.ego.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if waypoint is None:
            self._ego_off_road = True
            print('Non-drivable lane detected')
            return True

        # 5. Check if the vehicle's heading deviates too much from lane direction (> ±90°)
        ego_yaw = self.ego.get_transform().rotation.yaw
        lane_yaw = waypoint.transform.rotation.yaw
        yaw_diff = np.deg2rad(ego_yaw - lane_yaw)
        yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))  # Normalize to [-π, π]
        if not waypoint.is_intersection:
            if abs(yaw_diff) > np.pi / 2:  # More than 90 degrees deviation (wrong-way driving)
                self._ego_off_road = True
                print('Wrong-way driving detected')
                return True

        # 6. Deviation too far from lane center
        lane_width, lateral_offset = obs['lane_info']
        if not waypoint.is_intersection:
            # if lateral_offset > lane_width + 1.0:
            if lateral_offset > lane_width / 2 + 1.0:
                self._ego_off_road = True
                print('Deviated from lane')
                return True

        # 7. 周车完成路径或到达仿真时长
        if self.bv_done:
            print('BV_done')
            return True

        return False

    def _get_reward(self, obs, done): 

        # r_collision = -1 if self._ego_collision else 0

        # # reward for steering:
        # r_steer = -self.ego.get_control().steer ** 2 #输入的是state和action，这里是action

        # # reward for out of lane
        # lane_width, lateral_offset = obs['lane_info']
        # # out_lane_thres = lane_width / 2 + 1.0
        # out_lane_thres = 2.0
        # r_out = -1 if abs(lateral_offset) > out_lane_thres else 0

        # # cost for too fast
        # speed = obs['ego_state'][3]
        # r_fast = -1 if speed > self.desired_speed else 0

        # # cost for lateral acceleration
        # a_lat = obs['ego_state'][6]
        # r_lat = -abs(a_lat) * speed ** 2

        # reward = 200 * r_collision + 1 * speed + 10 * r_fast + 1 * r_out + r_steer * 5 + 0.2 * r_lat - 0.1
    

        #Old reward
        reward = 0.0

        # 1. Forward driving reward (within speed limit and along lane direction)
        speed = obs['ego_state'][3]
        # print("speed", speed)
        if speed <= self.desired_speed:
            reward += 1.0 * speed
        else:
            reward += -5.0 * (speed - self.desired_speed)

        # 2. Lane deviation penalty (penalize offset from lane center)
        lane_width, lateral_offset = obs['lane_info']
        reward += -1.0 * lateral_offset

        # 3. Smooth driving penalty (lateral acceleration penalty)
        a_lat = obs['ego_state'][6]
        reward += -0.2 * abs(a_lat)* speed ** 2

        # 3*. Smooth driving penalty (steer penalty)
        r_steer = -self.ego.get_control().steer ** 2
        reward += 5 * r_steer

        # 4. Stationary penalty (if no vehicle ahead but ego is barely moving)
        # front_distance = obs['ego_state'][7]
        # if front_distance > 10.0 and speed < 0.1:
        #     reward += -2.0 #da yi dian
        #     # print("front_distance", front_distance)

        # 5. Collision penalty
        if self._ego_collision:
            reward += -200.0

        # 6. Off-road penalty
        if self._ego_off_road:
            reward += -100.0

        # 7. Sparse terminal reward (for safely reaching the goal)
        # if done:
        #     if not self._is_collision and not self._is_off_road:
        #         reward += 200.0

        return reward

    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        """Create a vehicle blueprint based on the given filter and wheel number.

        Args:
            actor_filter (str): Filter string to select vehicle types, e.g., 'vehicle.lincoln*'
                                ('*' matches a series of models).
            color (str, optional): Desired vehicle color. Randomly chosen if None.
            number_of_wheels (list): A list of acceptable wheel numbers (default is [4]).

        Returns:
            bp (carla.ActorBlueprint): A randomly selected blueprint matching the criteria.
        """
        # Get all blueprints matching the actor filter
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []

        # Further filter blueprints based on the number of wheels
        # Keeping number_of_wheels as a list makes it flexible to match multiple types (e.g., cars, trucks)
        for nw in number_of_wheels:
            blueprint_library += [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]

        # Randomly select one blueprint from the filtered list
        bp = random.choice(blueprint_library)

        # Set the vehicle color
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        return bp

    def _set_synchronous_mode(self, synchronous=True):

        """Enable or disable synchronous mode for the simulation.
        Args:
            synchronous (bool):
                True to enable synchronous mode (server waits for client each frame),
                False to disable and run in asynchronous mode (default is True).
        """
        self.settings.synchronous_mode = synchronous  # Set the synchronous mode
        self.world.apply_settings(self.settings)  # Apply the updated settings to the world

    def _clear_all_actors(self, actor_filters):
        """Clear (destroy) all actors matching the given filter patterns.

        Args:
            actor_filters (list): A list of filter strings, e.g., ['vehicle.*', 'walker.*', 'sensor.*'].
        """
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                try:
                    # If the actor is a sensor, stop it before destroying
                    if 'sensor' in actor.type_id:
                        actor.stop()
                    actor.destroy()
                except:
                    pass  # Ignore any errors during destruction

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        """Try to spawn a surrounding vehicle at a specific transform.

        Args:
            transform (carla.Transform): Location and orientation where the vehicle should be spawned.
            number_of_wheels (list): Acceptable number(s) of wheels for the vehicle blueprint.
            random_vehicle (bool):
                False to use Tesla Model 3 with a blue color,
                True to randomly select a vehicle model and color (default).

        Returns:
            carla.Actor or None: Spawned vehicle actor if successful, otherwise None.
        """
        if self.surrounding_vehicle_spawned_randomly:
            # Randomly choose any vehicle blueprint
            blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
        else:
            # Fixed: Tesla Model 3 with blue color
            blueprint = self._create_vehicle_bluepprint('vehicle.tesla.model3', color='0,0,255',
                                                        number_of_wheels=number_of_wheels)

        blueprint.set_attribute('role_name', 'autopilot')  # Set the vehicle to autopilot mode

        # Try to spawn the vehicle
        vehicle = self.world.try_spawn_actor(blueprint, transform)

        return vehicle if vehicle is not None else None

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at a specific transform.

        Args:
            transform (carla.Transform): Target location and orientation.

        Returns:
            Bool: True if spawn is successful, False otherwise.
        """
        vehicle = None
        overlap = False

        # Check if ego position overlaps with surrounding vehicles
        # for idx, poly in self.vehicle_polygons[-1].items():  # Use .items() to iterate over dict keys and values
        #     poly_center = np.mean(poly, axis=0)
        #     ego_center = np.array([transform.location.x, transform.location.y])
        #     dis = np.linalg.norm(poly_center - ego_center)  # 线性代数子模块中的求范数
        #
        #     if dis > 8:
        #         continue
        #     else:
        #         overlap = True
        #         break

        # If no overlap, try to spawn the ego vehicle
        # if not overlap:
        vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

        if vehicle is not None:
            time.sleep(0.1)
            self.ego = vehicle
            return True

        return False

    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.

        Args:
            filt: the filter indicating what type of actors we'll look at.

        Returns:
            actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
        """
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            # Get all actors in the current world that meet the filt condition, such as vehicle.* or walker.*
            # Note that self.world.get_actors() retrieves all objects in the current simulation environment (vehicles, pedestrians, traffic lights, etc.).

            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            # Get the actor's global position (location) and heading angle (rotation).

            x = trans.location.x
            # x, y are the actor's global coordinates.

            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            # yaw is the heading angle, whose unit is degrees, needs to be converted to radians (multiply by pi/180) to facilitate subsequent matrix calculations.

            # Get length and width
            bb = actor.bounding_box
            # Get the "half-length" boundary.

            l = bb.extent.x
            # "Half-length" in the x-direction (the distance from the center to the edge).

            w = bb.extent.y
            # "Half-width" in the y-direction (the distance from the center to the edge).

            # Get bounding box polygon in the actor's local coordinate
            # Take the vehicle center as the origin, build a local coordinate system, and list four corner points:
            # (l, w): front right corner, (l, -w): rear right corner, (-l, -w): rear left corner, (-l, w): front left corner
            poly_local = np.array([
                [l, w], [l, -w], [-l, -w], [-l, w]
            ]).transpose()
            # Transpose() here is to facilitate subsequent matrix operations,
            # changing the matrix from (4,2) to (2,4) format.

            # Get rotation matrix to transform to global coordinate
            # This is a standard 2D rotation matrix: used to transform points from the local coordinate system to the global coordinate system.
            # Rotation matrix R = [cosθ  -sinθ]
            #                     [sinθ   cosθ]
            R = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)]
            ])

            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)  # 获得四个边界点的全局位置
            # np.matmul(R, poly_local):
            # Transform the four corners (in the local coordinate system) into the global direction through the rotation matrix.
            # After .transpose(), it becomes (4,2) format (one point per row).
            # + np.repeat([[x,y]],4,axis=0):
            # Add the global position offset of the vehicle/pedestrian to each point
            # to obtain the final polygon coordinates in the global coordinate system.

            actor_poly_dict[actor.id] = poly
            # Store the calculated poly (a 4×2 array, four corner points in global coordinates)
            # with actor.id as the key into actor_poly_dict.
            # After returning, the entire dictionary structure:
            # {
            # actor_id_1: np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]),
            # actor_id_2: np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]),
            # ...
            # }

        return actor_poly_dict

    def _get_next_waypoint_random(self, max_waypoints, ego_yaw, ego_x, ego_y):
        waypoints_array = np.zeros((max_waypoints, 3), dtype=np.float32)
        waypoint = self.world_map.get_waypoint(self.ego.get_location())  # 距自车最近的路点

        for i in range(max_waypoints):
            if waypoint is None:
                break

            loc = waypoint.transform.location
            yaw = waypoint.transform.rotation.yaw

            # Transform waypoint location into ego-centric local coordinates
            local_x = np.cos(-ego_yaw) * (loc.x - ego_x) - np.sin(-ego_yaw) * (loc.y - ego_y)
            local_y = np.sin(-ego_yaw) * (loc.x - ego_x) + np.cos(-ego_yaw) * (loc.y - ego_y)
            yaw_relative = np.deg2rad(yaw) - ego_yaw  # Relative heading

            waypoints_array[i] = [local_x, local_y, yaw_relative]  # 路点和自车的相对坐标，在自车坐标中

            # Move to the next waypoint 2.0 meters ahead
            next_waypoints = waypoint.next(2.0)  # 当前路点沿车道2m后的后续路点

            waypoint = next_waypoints[0] if next_waypoints[0] else None  # 记得要next_waypoints[0]，从列表中取第一个元素

        return waypoints_array

    def _get_next_waypoint_left_turn(self, max_waypoints, ego_yaw, ego_x, ego_y):
        waypoints_array = np.zeros((max_waypoints, 3), dtype=np.float32)
        ego_loc = self.ego.get_location()
        ego_wp = self.world_map.get_waypoint(self.ego.get_location())  # 距自车最近的路点

        if not ego_wp.is_junction:
            self.in_junction = False
            self.junction_first = True
            waypoint = ego_wp
            for i in range(max_waypoints):
                if waypoint is None:
                    break

                loc = waypoint.transform.location
                yaw = waypoint.transform.rotation.yaw

                # world -> ego local
                dx = loc.x - ego_x
                dy = loc.y - ego_y
                local_x = np.cos(-ego_yaw) * dx - np.sin(-ego_yaw) * dy
                local_y = np.sin(-ego_yaw) * dx + np.cos(-ego_yaw) * dy
                yaw_relative = np.deg2rad(yaw) - ego_yaw

                waypoints_array[i] = [local_x, local_y, yaw_relative]

                # next 2m
                nxt = waypoint.next(2.0)
                waypoint = nxt[-1] if nxt else None

            return waypoints_array

        # ------------------ 情况 B：已经在路口内 -> 走“该车道的左转连接路径” ------------------
        junction = ego_wp.get_junction()

         #
        conn_pairs = junction.get_waypoints(carla.LaneType.Driving)

        left_candidates = []

        for entry_wp, exit_wp in conn_pairs:

            yaw_in = ego_wp.transform.rotation.yaw
            yaw_out = exit_wp.transform.rotation.yaw
            # print("yaw in", yaw_in)
            # print(yaw_out)

            # 规范到 [-180, 180]
            delta_yaw = ((yaw_out - yaw_in + 540) % 360) - 180

            # 左转：朝向变化大约 +90°（可以设个范围）
            if -160.0 < delta_yaw < -20:
                left_candidates.append((ego_wp, exit_wp))

        if not left_candidates:
            waypoint = ego_wp
            for i in range(max_waypoints):
                if waypoint is None:
                    break

                loc = waypoint.transform.location
                yaw = waypoint.transform.rotation.yaw

                # world -> ego local
                dx = loc.x - ego_x
                dy = loc.y - ego_y
                local_x = np.cos(-ego_yaw) * dx - np.sin(-ego_yaw) * dy
                local_y = np.sin(-ego_yaw) * dx + np.cos(-ego_yaw) * dy
                yaw_relative = np.deg2rad(yaw) - ego_yaw

                waypoints_array[i] = [local_x, local_y, yaw_relative]

                # next 2m
                nxt = waypoint.next(2.0)
                waypoint = nxt[0] if nxt else None

            return waypoints_array


        # 3) 如有多个左转连接，选入口点离当前 ego 最近的那个
        def sq_dist_entry(pair):
            entry_wp, _ = pair
            l = entry_wp.transform.location
            return (l.x - ego_loc.x) ** 2 + (l.y - ego_loc.y) ** 2

        ego_wp, exit_wp = min(left_candidates, key=sq_dist_entry)
        # print(exit_wp)

        if ego_wp.is_junction:
            self.in_junction = True

        if self.in_junction and self.junction_first:

            # 4) 用 GlobalRoutePlanner 从入口到出口生成完整左转路径
            route = self.grp.trace_route(ego_wp.transform.location, exit_wp.transform.location)
            self.route_wps = [wp for (wp, road_opt) in route]  # route 每个元素是 (Waypoint, RoadOption)
            self.junction_first = False

        def sq_dist_wp(wp):
            l = wp.transform.location
            return (l.x - ego_loc.x) ** 2 + (l.y - ego_loc.y) ** 2

        start_idx = min(range(len(self.route_wps)), key=lambda idx: sq_dist_wp(self.route_wps[idx]))
        start_wp = self.route_wps[start_idx]
        if ego_wp.is_junction:
            # 6) 从 start_idx 开始，往后每隔若干米取点，这里直接按顺序取 max_waypoints 个
            waypoint = start_wp
            for i in range(max_waypoints):
                if waypoint is None:
                    break

                loc = waypoint.transform.location
                yaw = waypoint.transform.rotation.yaw

                dx = loc.x - ego_x
                dy = loc.y - ego_y
                local_x = np.cos(-ego_yaw) * dx - np.sin(-ego_yaw) * dy
                local_y = np.sin(-ego_yaw) * dx + np.cos(-ego_yaw) * dy
                yaw_relative = np.deg2rad(yaw) - ego_yaw

                waypoints_array[i] = [local_x, local_y, yaw_relative]

                next_wps = waypoint.next(2.0)
                waypoint = next_wps[0] if next_wps else None

            return waypoints_array

    def _get_next_waypoint_right_turn(self, max_waypoints, ego_yaw, ego_x, ego_y):
        waypoints_array = np.zeros((max_waypoints, 3), dtype=np.float32)
        ego_loc = self.ego.get_location()
        ego_wp = self.world_map.get_waypoint(self.ego.get_location())  # 距自车最近的路点

        if not ego_wp.is_junction:
            self.in_junction = False
            self.junction_first = True
            waypoint = ego_wp
            for i in range(max_waypoints):
                if waypoint is None:
                    break

                loc = waypoint.transform.location
                yaw = waypoint.transform.rotation.yaw

                # world -> ego local
                dx = loc.x - ego_x
                dy = loc.y - ego_y
                local_x = np.cos(-ego_yaw) * dx - np.sin(-ego_yaw) * dy
                local_y = np.sin(-ego_yaw) * dx + np.cos(-ego_yaw) * dy
                yaw_relative = np.deg2rad(yaw) - ego_yaw

                waypoints_array[i] = [local_x, local_y, yaw_relative]

                # next 2m
                nxt = waypoint.next(2.0)
                waypoint = nxt[-1] if nxt else None

            return waypoints_array

        # ------------------ 情况 B：已经在路口内 -> 走“该车道的左转连接路径” ------------------
        junction = ego_wp.get_junction()

         #
        conn_pairs = junction.get_waypoints(carla.LaneType.Driving)

        left_candidates = []

        for entry_wp, exit_wp in conn_pairs:

            yaw_in = ego_wp.transform.rotation.yaw
            yaw_out = exit_wp.transform.rotation.yaw
            # print("yaw in", yaw_in)
            # print(yaw_out)

            # 规范到 [-180, 180]
            delta_yaw = ((yaw_out - yaw_in + 540) % 360) - 180

            # 左转：朝向变化大约 +90°（可以设个范围）
            if -160.0 < delta_yaw < -20:
                left_candidates.append((ego_wp, exit_wp))

        if not left_candidates:
            waypoint = ego_wp
            for i in range(max_waypoints):
                if waypoint is None:
                    break

                loc = waypoint.transform.location
                yaw = waypoint.transform.rotation.yaw

                # world -> ego local
                dx = loc.x - ego_x
                dy = loc.y - ego_y
                local_x = np.cos(-ego_yaw) * dx - np.sin(-ego_yaw) * dy
                local_y = np.sin(-ego_yaw) * dx + np.cos(-ego_yaw) * dy
                yaw_relative = np.deg2rad(yaw) - ego_yaw

                waypoints_array[i] = [local_x, local_y, yaw_relative]

                # next 2m
                nxt = waypoint.next(2.0)
                waypoint = nxt[0] if nxt else None

            return waypoints_array


        # 3) 如有多个左转连接，选入口点离当前 ego 最近的那个
        def sq_dist_entry(pair):
            entry_wp, _ = pair
            l = entry_wp.transform.location
            return (l.x - ego_loc.x) ** 2 + (l.y - ego_loc.y) ** 2

        ego_wp, exit_wp = min(left_candidates, key=sq_dist_entry)
        # print(exit_wp)

        if ego_wp.is_junction:
            self.in_junction = True

        if self.in_junction and self.junction_first:

            # 4) 用 GlobalRoutePlanner 从入口到出口生成完整左转路径
            route = self.grp.trace_route(ego_wp.transform.location, exit_wp.transform.location)
            print(ego_wp.transform.location,exit_wp.transform.location)
            self.route_wps = [wp for (wp, road_opt) in route]  # route 每个元素是 (Waypoint, RoadOption)
            self.junction_first = False

        def sq_dist_wp(wp):
            l = wp.transform.location
            return (l.x - ego_loc.x) ** 2 + (l.y - ego_loc.y) ** 2

        start_idx = min(range(len(self.route_wps)), key=lambda idx: sq_dist_wp(self.route_wps[idx]))
        start_wp = self.route_wps[start_idx]
        if ego_wp.is_junction:
            # 6) 从 start_idx 开始，往后每隔若干米取点，这里直接按顺序取 max_waypoints 个
            waypoint = start_wp
            for i in range(max_waypoints):
                if waypoint is None:
                    break

                loc = waypoint.transform.location
                yaw = waypoint.transform.rotation.yaw

                dx = loc.x - ego_x
                dy = loc.y - ego_y
                local_x = np.cos(-ego_yaw) * dx - np.sin(-ego_yaw) * dy
                local_y = np.sin(-ego_yaw) * dx + np.cos(-ego_yaw) * dy
                yaw_relative = np.deg2rad(yaw) - ego_yaw

                waypoints_array[i] = [local_x, local_y, yaw_relative]

                next_wps = waypoint.next(2.0)
                waypoint = next_wps[0] if next_wps else None

            return waypoints_array

    def _get_next_waypoint_lanechange(self, max_waypoints, ego_yaw, ego_x, ego_y):
        dis_to_cut = 7
        lanechange = "right"
        target_dis = 20

        waypoints_array = np.zeros((max_waypoints, 3), dtype=np.float32)

        ego_loc = self.ego.get_location()
        ego_wp = self.world_map.get_waypoint(ego_loc, project_to_road=True,
                                             lane_type=carla.LaneType.Driving)  # 获取自车当前的路点

        right_wp = ego_wp.get_right_lane()

        # 获取右侧车道上的下一路点，作为变道起点
        spawn_wp = right_wp.next(2.0)[0] # 获取距离当前路点2米的右侧车道路点

        npc_vehicle = self.ABV

        if npc_vehicle:
            npc_loc = npc_vehicle.get_location()
            lateral_distance = ego_loc.x - npc_loc.x  # 横向距离（假设车辆在同一条直线）

            # 判断是否满足变道条件
            if lateral_distance < dis_to_cut or self.over:
                waypoint = ego_wp
                for i in range(max_waypoints):
                    if waypoint is None:
                        break

                    loc = waypoint.transform.location
                    yaw = waypoint.transform.rotation.yaw

                    # world -> ego local
                    dx = loc.x - ego_x
                    dy = loc.y - ego_y
                    local_x = np.cos(-ego_yaw) * dx - np.sin(-ego_yaw) * dy
                    local_y = np.sin(-ego_yaw) * dx + np.cos(-ego_yaw) * dy
                    yaw_relative = np.deg2rad(yaw) - ego_yaw

                    waypoints_array[i] = [local_x, local_y, yaw_relative]

                    # next 2m
                    nxt = waypoint.next(2.0)
                    waypoint = nxt[-1] if nxt else None
                return waypoints_array

            if lateral_distance >= dis_to_cut and not self.over:
                # 生成变道轨迹：沿右侧车道生成后续的 waypoints
                if self.need_cal_route:
                    if "left" in lanechange:
                        target_org_waypoint = ego_wp.get_left_lane()
                    elif "right" in lanechange:
                        target_org_waypoint = ego_wp.get_right_lane()
                    # 获取变道终点的位置
                    target_location = target_org_waypoint.next(target_dis)[0].transform.location
                    route = self.grp.trace_route(ego_wp.transform.location, target_location)
                    # self.draw_target_line(route)
                    self.route_wps = [wp for (wp, road_opt) in route]

                    self.need_cal_route = False

                def sq_dist_wp(wp):
                    l = wp.transform.location
                    return (l.x - ego_loc.x) ** 2 + (l.y - ego_loc.y) ** 2

                start_idx = min(range(len(self.route_wps)), key=lambda idx: sq_dist_wp(self.route_wps[idx]))
                if start_idx > len(self.route_wps)*0.8:
                    self.over = True
                    
                start_wp = self.route_wps[start_idx]

                waypoint = start_wp
                for i in range(max_waypoints):
                    if waypoint is None:
                        break

                    loc = waypoint.transform.location
                    yaw = waypoint.transform.rotation.yaw

                    dx = loc.x - ego_x
                    dy = loc.y - ego_y
                    local_x = np.cos(-ego_yaw) * dx - np.sin(-ego_yaw) * dy
                    local_y = np.sin(-ego_yaw) * dx + np.cos(-ego_yaw) * dy
                    yaw_relative = np.deg2rad(yaw) - ego_yaw

                    waypoints_array[i] = [local_x, local_y, yaw_relative]

                    next_wps = waypoint.next(2.0)
                    waypoint = next_wps[0] if next_wps else None

                return waypoints_array

    def _get_next_waypoint_straight(self, max_waypoints, ego_yaw, ego_x, ego_y):
        waypoints_array = np.zeros((max_waypoints, 3), dtype=np.float32)
        
        # 如果还没有记录初始车道waypoint，则记录（备用）
        if self.initial_lane_waypoint is None:
            self.initial_lane_waypoint = self.world_map.get_waypoint(self.ego.get_location())
        
        # 获取初始车道的lane_id和road_id
        initial_lane_id = self.initial_lane_waypoint.lane_id
        initial_road_id = self.initial_lane_waypoint.road_id
        ego_location = self.ego.get_location()
        
        # 在初始车道上搜索距离当前ego位置最近的路点
        best_waypoint = self.initial_lane_waypoint
        min_dist = float('inf')
        
        # 向前搜索
        search_waypoint = self.initial_lane_waypoint
        for _ in range(3000):  # 最多向前搜索200米
            if search_waypoint is None:
                break
            if search_waypoint.lane_id != initial_lane_id or search_waypoint.road_id != initial_road_id:
                break
            dist = ego_location.distance(search_waypoint.transform.location)
            if dist < min_dist:
                min_dist = dist
                best_waypoint = search_waypoint
            next_wps = search_waypoint.next(0.2)
            if not next_wps:
                break
            search_waypoint = next_wps[0]
        
        # 向后搜索
        # search_waypoint = self.initial_lane_waypoint
        # for _ in range(100):  # 最多向后搜索200米
        #     if search_waypoint is None:
        #         break
        #     if search_waypoint.lane_id != initial_lane_id or search_waypoint.road_id != initial_road_id:
        #         break
        #     dist = ego_location.distance(search_waypoint.transform.location)
        #     if dist < min_dist:
        #         min_dist = dist
        #         best_waypoint = search_waypoint
        #     prev_wps = search_waypoint.previous(2.0)
        #     if not prev_wps:
        #         break
        #     search_waypoint = prev_wps[0]
        
        # 从找到的最佳waypoint开始，沿着初始车道生成后续waypoints
        waypoint = best_waypoint

        for i in range(max_waypoints):
            if waypoint is None:
                break

            loc = waypoint.transform.location
            yaw = waypoint.transform.rotation.yaw

            # Transform waypoint location into ego-centric local coordinates
            local_x = np.cos(-ego_yaw) * (loc.x - ego_x) - np.sin(-ego_yaw) * (loc.y - ego_y)
            local_y = np.sin(-ego_yaw) * (loc.x - ego_x) + np.cos(-ego_yaw) * (loc.y - ego_y)
            yaw_relative = np.deg2rad(yaw) - ego_yaw  # Relative heading

            waypoints_array[i] = [local_x, local_y, yaw_relative]  # 路点和自车的相对坐标，在自车坐标中

            # Move to the next waypoint 2.0 meters ahead
            next_waypoints = waypoint.next(2.0)  # 当前路点沿车道2m后的后续路点
            
            # 确保下一个waypoint仍在同一车道上
            if next_waypoints and len(next_waypoints) > 0:
                next_wp = next_waypoints[0]
                if next_wp.lane_id == initial_lane_id and next_wp.road_id == initial_road_id:
                    waypoint = next_wp
                else:
                    waypoint = None
            else:
                waypoint = None

        return waypoints_array
