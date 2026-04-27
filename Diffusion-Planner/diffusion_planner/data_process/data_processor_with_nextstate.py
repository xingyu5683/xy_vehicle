import numpy as np
import os
import tempfile
import shutil
import subprocess
from tqdm import tqdm

from nuplan.common.actor_state.state_representation import Point2D

from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction
from diffusion_planner.data_process.agent_process import (
agent_past_process, 
sampled_tracked_objects_to_array_list,
sampled_static_objects_to_array_list,
agent_future_process
)
from diffusion_planner.data_process.map_process import get_neighbor_vector_set_map, map_process
from diffusion_planner.data_process.ego_process import get_ego_past_array_from_scenario, get_ego_future_array_from_scenario, calculate_additional_ego_states
from diffusion_planner.data_process.utils import convert_to_model_inputs
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import EgoInternalIndex
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses


class DataProcessor(object):
    def __init__(self, config):

        self._save_dir = getattr(config, "save_path", None) 

        self.past_time_horizon = 2 # [seconds]
        self.num_past_poses = 10 * self.past_time_horizon 
        self.future_time_horizon = 8 # [seconds]
        self.num_future_poses = 10 * self.future_time_horizon

        self.num_agents = config.agent_num
        self.num_static = config.static_objects_num
        self.max_ped_bike = 10 # Limit the number of pedestrians and bicycles in the agent.
        self._radius = 100 # [m] query radius scope relative to the current pose.

        self._map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'ROUTE_LANES'] # name of map features to be extracted.
        self._max_elements = {'LANE': config.lane_num, 'LEFT_BOUNDARY': config.lane_num, 'RIGHT_BOUNDARY': config.lane_num, 'ROUTE_LANES': config.route_num} # maximum number of elements to extract per feature layer.
        self._max_points = {'LANE': config.lane_len, 'LEFT_BOUNDARY': config.lane_len, 'RIGHT_BOUNDARY': config.lane_len, 'ROUTE_LANES': config.route_len} # maximum number of points per feature to extract per feature layer.

    # Use for inference
    def observation_adapter(self, history_buffer, traffic_light_data, map_api, route_roadblock_ids, device='cpu'):

        '''
        ego
        '''
        ego_agent_past = None # inference no need ego_agent_past
        ego_state = history_buffer.current_state[0]
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        anchor_ego_state = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading], dtype=np.float64)

        '''
        neighbor
        '''
        observation_buffer = history_buffer.observation_buffer # Past observations including the current
        neighbor_agents_past, neighbor_agents_types = sampled_tracked_objects_to_array_list(observation_buffer)
        static_objects, static_objects_types = sampled_static_objects_to_array_list(observation_buffer[-1])
        _, neighbor_agents_past, _, static_objects = \
            agent_past_process(ego_agent_past, neighbor_agents_past, neighbor_agents_types, self.num_agents, static_objects, static_objects_types, self.num_static, self.max_ped_bike, anchor_ego_state)

        '''
        Map
        '''
        # Simply fixing disconnected routes without pre-searching for reference lines
        route_roadblock_ids = route_roadblock_correction(
            ego_state, map_api, route_roadblock_ids
        )
        coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
            map_api, self._map_features, ego_coords, self._radius, traffic_light_data
        )
        vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords, traffic_light_data, speed_limit, lane_route, self._map_features, 
                                    self._max_elements, self._max_points)

        
        data = {"neighbor_agents_past": neighbor_agents_past[:, -21:],
                "ego_current_state": np.array([0., 0., 1. ,0., 0., 0., 0., 0., 0., 0.], dtype=np.float32), # ego centric x, y, cos, sin, vx, vy, ax, ay, steering angle, yaw rate, we only use x, y, cos, sin during inference
                "static_objects": static_objects}
        data.update(vector_map)
        data = convert_to_model_inputs(data, device)

        return data
    
    def _process_single_iteration(self, scenario, iteration):
        """
        Process data for a single iteration (time step).
        This method extracts all data for a given iteration, following the same structure
        as the original work() method but for a specific iteration.
        
        :param scenario: NuPlan scenario object
        :param iteration: iteration index (0 for current, 1 for next)
        :return: dictionary containing processed data for this iteration
        """
        map_name = scenario._map_name
        token = scenario.token
        map_api = scenario.map_api

        '''
        ego & agents past
        '''
        # Get ego state at this iteration
        ego_state = scenario.get_ego_state_at_iteration(iteration)
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        anchor_ego_state = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading], dtype=np.float64)
        
        # Get ego past trajectory for this iteration
        # Unified approach: use the same logic for all iterations
        # Note: get_ego_past_array_from_scenario only supports iteration=0, so we use a unified approach
        past_ego_states = list(scenario.get_ego_past_trajectory(
            iteration=iteration, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
        ))
        current_ego_state = scenario.get_ego_state_at_iteration(iteration)
        sampled_past_ego_states = past_ego_states + [current_ego_state]
        
        # Convert to array format (same as sampled_past_ego_states_to_array)
        ego_agent_past = np.zeros((len(sampled_past_ego_states), 7), dtype=np.float64)
        for i, ego_state_item in enumerate(sampled_past_ego_states):
            ego_agent_past[i, EgoInternalIndex.x()] = ego_state_item.rear_axle.x
            ego_agent_past[i, EgoInternalIndex.y()] = ego_state_item.rear_axle.y
            ego_agent_past[i, EgoInternalIndex.heading()] = ego_state_item.rear_axle.heading
            ego_agent_past[i, EgoInternalIndex.vx()] = ego_state_item.dynamic_car_state.rear_axle_velocity_2d.x
            ego_agent_past[i, EgoInternalIndex.vy()] = ego_state_item.dynamic_car_state.rear_axle_velocity_2d.y
            ego_agent_past[i, EgoInternalIndex.ax()] = ego_state_item.dynamic_car_state.rear_axle_acceleration_2d.x
            ego_agent_past[i, EgoInternalIndex.ay()] = ego_state_item.dynamic_car_state.rear_axle_acceleration_2d.y
        
        # Get time stamps
        past_time_stamps = list(scenario.get_past_timestamps(
            iteration=iteration, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
        )) + [scenario.get_time_point(iteration)]
        time_stamps_past = np.array([t.time_us for t in past_time_stamps], dtype=np.int64)

        # Get tracked objects at this iteration
        tracked_objects_at_iteration = scenario.get_tracked_objects_at_iteration(iteration)
        present_tracked_objects = tracked_objects_at_iteration.tracked_objects
        
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_past_tracked_objects(
                iteration=iteration, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
            )
        ]
        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        neighbor_agents_past, neighbor_agents_types = \
            sampled_tracked_objects_to_array_list(sampled_past_observations)
        
        static_objects, static_objects_types = sampled_static_objects_to_array_list(present_tracked_objects)

        ego_agent_past, neighbor_agents_past, neighbor_indices, static_objects = \
            agent_past_process(ego_agent_past, neighbor_agents_past, neighbor_agents_types, self.num_agents, static_objects, static_objects_types, self.num_static, self.max_ped_bike, anchor_ego_state)
        
        '''
        Map
        '''
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(iteration))

        if route_roadblock_ids != ['']:
            route_roadblock_ids = route_roadblock_correction(
                ego_state, map_api, route_roadblock_ids
            )

        coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
            map_api, self._map_features, ego_coords, self._radius, traffic_light_data
        )

        vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords, traffic_light_data, speed_limit, lane_route, self._map_features, 
                                self._max_elements, self._max_points)

        '''
        ego & agents future
        '''
        # Get ego future trajectory for this iteration
        # Note: get_ego_future_array_from_scenario only supports iteration=0
        # For other iterations, we need to manually get future trajectory
        if iteration == 0:
            ego_agent_future = get_ego_future_array_from_scenario(scenario, ego_state, self.num_future_poses, self.future_time_horizon)
        else:
            # For iteration > 0, manually get future trajectory
            future_ego_states = list(scenario.get_ego_future_trajectory(
                iteration=iteration, num_samples=self.num_future_poses, time_horizon=self.future_time_horizon
            ))
            # Convert to relative coordinates (same as get_ego_future_array_from_scenario)
            future_trajectory_relative_poses = convert_absolute_to_relative_poses(
                ego_state.rear_axle, [state.rear_axle for state in future_ego_states]
            )
            ego_agent_future = future_trajectory_relative_poses

        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_future_tracked_objects(
                iteration=iteration, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses
            )
        ]

        sampled_future_observations = [present_tracked_objects] + future_tracked_objects
        future_tracked_objects_array_list, _ = sampled_tracked_objects_to_array_list(sampled_future_observations)
        neighbor_agents_future = agent_future_process(anchor_ego_state, future_tracked_objects_array_list, self.num_agents, neighbor_indices)

        '''
        ego current
        '''
        ego_current_state = calculate_additional_ego_states(ego_agent_past, time_stamps_past)

        # gather data
        data = {"map_name": map_name, "token": token, "ego_current_state": ego_current_state, "ego_agent_future": ego_agent_future,
                "neighbor_agents_past": neighbor_agents_past, "neighbor_agents_future": neighbor_agents_future, "static_objects": static_objects}
        data.update(vector_map)

        return data

    # Use for data preprocess
    def work(self, scenarios):
        """
        Process scenarios and save data with both current state and next state.
        Randomly selects two adjacent iterations from each scenario.
        The data structure and generation method are identical for both states, only the sampling time differs.
        """
        for scenario in tqdm(scenarios):
            num_iterations = scenario.get_number_of_iterations()
            
            # Check if scenario has at least 2 iterations
            if num_iterations < 2:
                print(f"Warning: Scenario {scenario.token} has only {num_iterations} iterations, need at least 2, skipping...")
                continue
            
            # Randomly select two adjacent iterations from the scenario
            # iteration_current can be any value from 0 to num_iterations-2
            # iteration_next will be iteration_current + 1
            iteration_current = np.random.randint(0, num_iterations - 1)
            iteration_next = iteration_current + 1
            
            # Process current state
            data_current = self._process_single_iteration(scenario, iteration_current)
            
            # Process next state
            data_next = self._process_single_iteration(scenario, iteration_next)
            
            # Combine data: current state uses original keys, next state uses _next suffix
            combined_data = {}
            
            # Add current state data (original keys)
            for key, value in data_current.items():
                if key not in ['map_name', 'token']:  # Don't duplicate metadata
                    combined_data[key] = value
            
            # Add next state data (with _next suffix)
            for key, value in data_next.items():
                if key not in ['map_name', 'token']:  # Don't duplicate metadata
                    combined_data[f"{key}_next"] = value
            
            # Add metadata (use from current state)
            combined_data['map_name'] = data_current['map_name']
            combined_data['token'] = data_current['token']
            
            # Save combined data
            self.save_to_disk(self._save_dir, combined_data)

    def save_to_disk(self, dir, data):
        # 清理文件名：替换路径分隔符和其他无效字符
        map_name = str(data['map_name']).replace('/', '_').replace('\\', '_')
        token = str(data['token']).replace('/', '_').replace('\\', '_')
        
        # 清理其他可能的无效字符
        invalid_chars = [':', '*', '?', '"', '<', '>', '|', '\n', '\r', '\t']
        for char in invalid_chars:
            map_name = map_name.replace(char, '_')
            token = token.replace(char, '_')
        
        # 限制文件名长度（Linux 文件名限制 255 字符）
        if len(token) > 100:
            token = token[:100]
        if len(map_name) > 100:
            map_name = map_name[:100]
        
        # 确保目录存在
        os.makedirs(dir, exist_ok=True)
        
        # 构建文件路径
        filepath = os.path.join(dir, f"{map_name}_{token}.npz")
        
        # 如果文件已存在且损坏，先删除
        if os.path.exists(filepath):
            try:
                # 尝试加载，如果失败说明文件损坏
                test_data = np.load(filepath)
                test_data.close()
            except:
                # 文件损坏，删除它
                try:
                    os.remove(filepath)
                except:
                    pass
        
        # 保存文件
        # 如果目标路径是 OSS 文件系统，先保存到临时目录再移动
        # 检查是否是 OSS 文件系统（简化检测：检查路径是否在 /mnt/data 下）
        abs_dir = os.path.abspath(dir)
        is_oss_fs = abs_dir.startswith('/mnt/data')
        
        # 也可以尝试检测 mount 信息（备用）
        if not is_oss_fs:
            try:
                result = subprocess.run(['mount'], capture_output=True, text=True, timeout=2)
                if 'ossfs' in result.stdout:
                    # 检查是否有 OSS 挂载在 /mnt 下
                    for line in result.stdout.split('\n'):
                        if 'ossfs' in line and '/mnt' in line:
                            is_oss_fs = True
                            break
            except:
                pass
        
        if is_oss_fs:
            # OSS 文件系统：先保存到临时目录，再移动
            temp_dir = tempfile.gettempdir()
            temp_filepath = os.path.join(temp_dir, f"{map_name}_{token}.npz")
            
            try:
                # 保存到临时目录
                np.savez(temp_filepath, **data)
                
                # 移动到目标位置
                shutil.move(temp_filepath, filepath)
            except Exception as e:
                # 清理临时文件
                if os.path.exists(temp_filepath):
                    try:
                        os.remove(temp_filepath)
                    except:
                        pass
                # 清理目标文件（如果存在）
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except:
                        pass
                raise RuntimeError(f"保存文件失败: {filepath}\n  错误: {e}\n  map_name: {data['map_name']}\n  token: {data['token']}") from e
        else:
            # 普通文件系统：直接保存
            try:
                np.savez(filepath, **data)
            except Exception as e:
                # 如果保存失败，删除可能损坏的文件
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except:
                        pass
                raise RuntimeError(f"保存文件失败: {filepath}\n  错误: {e}\n  map_name: {data['map_name']}\n  token: {data['token']}") from e