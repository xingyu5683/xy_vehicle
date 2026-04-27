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
    
    # Use for data preprocess
    def work(self, scenarios):

        for scenario in tqdm(scenarios):
            map_name = scenario._map_name
            token = scenario.token
            map_api = scenario.map_api        

            '''
            ego & agents past
            '''
            ego_state = scenario.initial_ego_state
            ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
            anchor_ego_state = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading], dtype=np.float64)
            ego_agent_past, time_stamps_past = get_ego_past_array_from_scenario(scenario, self.num_past_poses, self.past_time_horizon)

            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
            past_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_past_tracked_objects(
                    iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
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
            traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))

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
            ego_agent_future = get_ego_future_array_from_scenario(scenario, ego_state, self.num_future_poses, self.future_time_horizon)

            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
            future_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_future_tracked_objects(
                    iteration=0, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses
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

            self.save_to_disk(self._save_dir, data)

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