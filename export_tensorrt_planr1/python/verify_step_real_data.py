#!/usr/bin/env python3
"""
用真实数据验证 StepModel TensorRT

步骤:
1. 加载真实数据
2. 运行 MapEncoder (PyTorch) 得到 polygon_embs
3. 构建 StepModel 输入 (边索引等)
4. 对比 StepModel TensorRT vs PyTorch

用法:
    cd /workspace/planner/export_tensorrt_planr1
    /usr/local/bin/python3 python/verify_step_real_data.py --num-samples 5
"""

import sys
sys.path.insert(0, '/workspace/planner/Plan-R1')
sys.path.insert(0, '/workspace/planner')

import os
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from export_onnx_planr1.map_encoder_exportable import MapEncoderExportable
from export_onnx_planr1.step_exportable import StepModel
from model.PlanR1 import PlanR1


def prepare_map_encoder_inputs(data, max_polygons=145, max_polylines=1200):
    """准备 MapEncoder 输入"""
    SINK_POLYGON = max_polygons - 1
    SINK_POLYLINE = max_polylines - 1
    
    polygon = data['polygon']
    polyline = data['polyline']
    num_polygons = min(polygon['position'].shape[0], max_polygons - 1)
    num_polylines = min(polyline['position'].shape[0], max_polylines - 1)
    
    inputs = {}
    
    inputs['polyline_position'] = torch.zeros((max_polylines, 2), dtype=torch.float32)
    inputs['polyline_position'][:num_polylines] = polyline['position'][:num_polylines, :2].float()
    
    inputs['polyline_heading'] = torch.zeros(max_polylines, dtype=torch.float32)
    inputs['polyline_heading'][:num_polylines] = polyline['heading'][:num_polylines].float()
    
    inputs['polyline_length'] = torch.ones(max_polylines, dtype=torch.float32)
    inputs['polyline_length'][:num_polylines] = polyline['length'][:num_polylines].float()
    
    inputs['polygon_position'] = torch.zeros((max_polygons, 2), dtype=torch.float32)
    inputs['polygon_position'][:num_polygons] = polygon['position'][:num_polygons, :2].float()
    
    inputs['polygon_heading'] = torch.zeros(max_polygons, dtype=torch.float32)
    inputs['polygon_heading'][:num_polygons] = polygon['heading'][:num_polygons].float()
    
    inputs['polygon_speed_limit'] = torch.zeros(max_polygons, dtype=torch.float32)
    inputs['polygon_speed_limit'][:num_polygons] = polygon['speed_limit'][:num_polygons].float()
    
    inputs['polygon_speed_limit_valid'] = torch.zeros(max_polygons, dtype=torch.float32)
    inputs['polygon_speed_limit_valid'][:num_polygons] = polygon['speed_limit_valid_mask'][:num_polygons].float()
    
    inputs['polygon_type'] = torch.zeros(max_polygons, dtype=torch.long)
    inputs['polygon_type'][:num_polygons] = polygon['type'][:num_polygons].long()
    
    inputs['polygon_traffic_light'] = torch.zeros(max_polygons, dtype=torch.long)
    inputs['polygon_traffic_light'][:num_polygons] = polygon['traffic_light'][:num_polygons].long()
    
    inputs['polygon_on_route'] = torch.zeros(max_polygons, dtype=torch.long)
    inputs['polygon_on_route'][:num_polygons] = polygon['on_route_mask'][:num_polygons].long()
    
    edge_data = data[('polyline', 'polygon')]
    edge_idx = edge_data['polyline_to_polygon_edge_index']
    n_edges = min(edge_idx.shape[1], max_polylines)
    
    l2g = torch.full((2, max_polylines), SINK_POLYGON, dtype=torch.long)
    l2g[0] = SINK_POLYLINE
    l2g[:, :n_edges] = edge_idx[:, :n_edges].long()
    inputs['l2g_edge_index'] = l2g
    
    pg_data = data[('polygon', 'polygon')]
    for name in ['left_edge_index', 'right_edge_index', 'incoming_edge_index', 'outgoing_edge_index']:
        edge = torch.full((2, 80), SINK_POLYGON, dtype=torch.long)
        if name in pg_data:
            n = min(pg_data[name].shape[1], 80)
            edge[:, :n] = pg_data[name][:, :n].long()
        inputs[name] = edge
    
    return inputs, num_polygons, num_polylines


def prepare_step_inputs(data, polygon_embs, max_agents=21, max_polygons=145, num_intervals=4, interval=5):
    """从真实数据构建 StepModel 输入"""
    agent = data['agent']
    polygon = data['polygon']
    
    num_agents_raw = agent['num_nodes']
    num_agents = min(num_agents_raw, max_agents)
    num_polygons = min(polygon['position'].shape[0], max_polygons)
    
    # Agent token (从数据中获取或随机生成)
    agent_token = torch.zeros((max_agents, num_intervals), dtype=torch.long)
    
    # Agent type
    agent_type = torch.zeros(max_agents, dtype=torch.long)
    agent_type[:num_agents] = agent['type'][:num_agents].long()
    
    # Agent box
    agent_box = torch.ones(max_agents, 4, dtype=torch.float32)
    agent_box[:num_agents] = agent['box'][:num_agents].float()
    
    # Agent identity
    agent_identity = torch.ones(max_agents, dtype=torch.long)
    agent_identity[:num_agents] = agent['identity'][:num_agents].long()
    
    # 历史位置和朝向 (取 interval 帧: 0, 5, 10, 15, 20)
    hist_frames = [i * interval for i in range(num_intervals + 1)]
    
    hist_position = torch.zeros(max_agents, num_intervals + 1, 2, dtype=torch.float32)
    hist_heading = torch.zeros(max_agents, num_intervals + 1, dtype=torch.float32)
    hist_valid = torch.zeros(max_agents, num_intervals + 1, dtype=torch.bool)
    
    for t_idx, frame in enumerate(hist_frames):
        if frame < agent['position'].shape[1]:
            hist_position[:num_agents, t_idx] = agent['position'][:num_agents, frame, :2].float()
            hist_heading[:num_agents, t_idx] = agent['heading'][:num_agents, frame].float()
            hist_valid[:num_agents, t_idx] = agent['visible_mask'][:num_agents, frame]
    
    # 构建 k2k_t 边 (时间注意力)
    k2k_t_edges = []
    k2k_t_attrs = []
    n_tokens = max_agents * num_intervals
    
    for a in range(num_agents):
        for t1 in range(num_intervals):
            for t2 in range(max(0, t1-6), t1+1):
                if hist_valid[a, t1] and hist_valid[a, t2]:
                    src_idx = a * num_intervals + t2
                    dst_idx = a * num_intervals + t1
                    if src_idx < n_tokens and dst_idx < n_tokens:
                        k2k_t_edges.append([src_idx, dst_idx])
                        dx = hist_position[a, t2, 0] - hist_position[a, t1, 0]
                        dy = hist_position[a, t2, 1] - hist_position[a, t1, 1]
                        length = math.sqrt(dx**2 + dy**2 + 1e-8)
                        theta = math.atan2(dy.item(), dx.item())
                        dh = hist_heading[a, t2] - hist_heading[a, t1]
                        dt = (t2 - t1) * interval
                        k2k_t_attrs.append([length, math.cos(theta), math.sin(theta),
                                           math.cos(dh.item()), math.sin(dh.item()), dt])
    
    if len(k2k_t_edges) == 0:
        k2k_t_edges = [[0, 0]]
        k2k_t_attrs = [[1.0, 1.0, 0.0, 1.0, 0.0, 0.0]]
    
    k2k_t_edge_index = torch.tensor(k2k_t_edges, dtype=torch.long).T
    k2k_t_edge_attr = torch.tensor(k2k_t_attrs, dtype=torch.float32)
    
    # 构建 g2k 边 (地图到 agent)
    g2k_edges = []
    g2k_attrs = []
    polygon_pos = polygon['position'][:num_polygons, :2].float()
    polygon_heading = polygon['heading'][:num_polygons].float()
    
    for p in range(num_polygons):
        for a in range(num_agents):
            for t in range(num_intervals):
                if hist_valid[a, t]:
                    dx = polygon_pos[p, 0] - hist_position[a, t, 0]
                    dy = polygon_pos[p, 1] - hist_position[a, t, 1]
                    dist = math.sqrt(dx**2 + dy**2)
                    if dist < 30:  # polygon_radius
                        g2k_edges.append([p, a * num_intervals + t])
                        length = dist + 1e-8
                        theta = math.atan2(dy.item(), dx.item())
                        dh = polygon_heading[p] - hist_heading[a, t]
                        g2k_attrs.append([length, math.cos(theta), math.sin(theta),
                                         math.cos(dh.item()), math.sin(dh.item()), 1.0])
    
    if len(g2k_edges) == 0:
        g2k_edges = [[0, 0]]
        g2k_attrs = [[1.0, 1.0, 0.0, 1.0, 0.0, 1.0]]
    
    g2k_edge_index = torch.tensor(g2k_edges, dtype=torch.long).T
    g2k_edge_attr = torch.tensor(g2k_attrs, dtype=torch.float32)
    
    # 构建 k2k_a 边 (agent 间注意力, time-major 索引)
    k2k_a_edges = []
    k2k_a_attrs = []
    
    for t in range(num_intervals):
        for a1 in range(num_agents):
            for a2 in range(num_agents):
                if a1 != a2 and hist_valid[a1, t] and hist_valid[a2, t]:
                    dx = hist_position[a1, t, 0] - hist_position[a2, t, 0]
                    dy = hist_position[a1, t, 1] - hist_position[a2, t, 1]
                    dist = math.sqrt(dx**2 + dy**2)
                    if dist < 60:  # agent_radius
                        src_idx = t * max_agents + a1
                        dst_idx = t * max_agents + a2
                        k2k_a_edges.append([src_idx, dst_idx])
                        length = dist + 1e-8
                        theta = math.atan2(dy.item(), dx.item())
                        dh = hist_heading[a1, t] - hist_heading[a2, t]
                        k2k_a_attrs.append([length, math.cos(theta), math.sin(theta),
                                           math.cos(dh.item()), math.sin(dh.item())])
    
    if len(k2k_a_edges) == 0:
        k2k_a_edges = [[0, 0]]
        k2k_a_attrs = [[1.0, 1.0, 0.0, 1.0, 0.0]]
    
    k2k_a_edge_index = torch.tensor(k2k_a_edges, dtype=torch.long).T
    k2k_a_edge_attr = torch.tensor(k2k_a_attrs, dtype=torch.float32)
    
    return {
        'agent_token': agent_token,
        'agent_type': agent_type,
        'agent_box': agent_box,
        'agent_identity': agent_identity,
        'polygon_embs': polygon_embs,
        'k2k_t_edge_index': k2k_t_edge_index,
        'k2k_t_edge_attr': k2k_t_edge_attr,
        'g2k_edge_index': g2k_edge_index,
        'g2k_edge_attr': g2k_edge_attr,
        'k2k_a_edge_index': k2k_a_edge_index,
        'k2k_a_edge_attr': k2k_a_edge_attr,
    }, num_agents


def main():
    parser = argparse.ArgumentParser(description='用真实数据验证 StepModel')
    parser.add_argument('--data-dir', type=str, 
                        default='/workspace/volumes/ad-pnc-al-sh01/plan-r1/dataset/nuplan-v1.1/splits/train-processed-pred-train-PlanR1')
    parser.add_argument('--ckpt-path', type=str,
                        default='/workspace/planner/Plan-R1/ckpts/fine-tuning.ckpt')
    parser.add_argument('--num-samples', type=int, default=5)
    parser.add_argument('--output', type=str, default='/tmp/step_real_data.npz')
    args = parser.parse_args()
    
    print('=' * 60)
    print('StepModel 真实数据验证 - 步骤1: PyTorch 生成参考输出')
    print('=' * 60)
    print()
    
    # 加载模型
    print('加载 PyTorch 模型...')
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    hparams = ckpt.get('hyper_parameters', {})
    
    original_model = PlanR1(
        mode='pred',
        token_dict_path='/workspace/planner/Plan-R1/tokens/tokens_1024.pt',
        **{k: v for k, v in hparams.items() if k not in ['mode', 'token_dict_path']}
    )
    original_model.load_state_dict(ckpt['state_dict'], strict=False)
    
    # MapEncoder
    map_encoder = MapEncoderExportable(hidden_dim=128, num_hops=4, num_heads=8, dropout=0.1)
    map_encoder.load_weights_from(original_model.pred_map_encoder)
    map_encoder.eval()
    
    # StepModel
    step_model = StepModel(hidden_dim=128, num_tokens=1024, num_attn_layers=6, num_heads=8, dropout=0.1)
    step_model.load_weights_from(original_model.pred_backbone, original_model.pred_decoder_head)
    step_model.eval()
    
    print('✓ 模型加载完成')
    print()
    
    # 加载数据
    data_dir = Path(args.data_dir)
    data_files = sorted(data_dir.glob('*.pt'))
    print(f'找到 {len(data_files)} 个数据文件')
    print()
    
    results = {}
    count = 0
    
    for data_file in data_files[:args.num_samples * 3]:
        if count >= args.num_samples:
            break
        
        data = torch.load(data_file)
        
        # 检查数据大小
        num_polygons_raw = data['polygon']['position'].shape[0]
        num_polylines_raw = data['polyline']['position'].shape[0]
        num_agents_raw = data['agent']['num_nodes']
        
        if num_polylines_raw > 1200 or num_polygons_raw > 145 or num_agents_raw < 2:
            continue
        
        print(f'处理样本 {count}: {data_file.name}')
        
        # 1. 运行 MapEncoder
        map_inputs, num_polygons, num_polylines = prepare_map_encoder_inputs(data)
        
        with torch.no_grad():
            polygon_embs = map_encoder(
                map_inputs['polyline_position'],
                map_inputs['polyline_heading'],
                map_inputs['polyline_length'],
                map_inputs['polygon_position'],
                map_inputs['polygon_heading'],
                map_inputs['polygon_speed_limit'],
                map_inputs['polygon_speed_limit_valid'],
                map_inputs['polygon_type'],
                map_inputs['polygon_traffic_light'],
                map_inputs['polygon_on_route'],
                map_inputs['l2g_edge_index'],
                map_inputs['left_edge_index'],
                map_inputs['right_edge_index'],
                map_inputs['incoming_edge_index'],
                map_inputs['outgoing_edge_index'],
            )
        
        print(f'  ✓ MapEncoder: polygon_embs shape = {polygon_embs.shape}')
        
        # 2. 构建 StepModel 输入
        step_inputs, num_agents = prepare_step_inputs(data, polygon_embs)
        
        print(f'  ✓ 构建边: k2k_t={step_inputs["k2k_t_edge_index"].shape[1]}, '
              f'g2k={step_inputs["g2k_edge_index"].shape[1]}, '
              f'k2k_a={step_inputs["k2k_a_edge_index"].shape[1]}')
        
        # 3. 运行 StepModel
        with torch.no_grad():
            pt_out = step_model(
                num_agents=torch.tensor([num_agents]),
                num_intervals=torch.tensor([4]),
                **step_inputs
            )
        
        print(f'  ✓ StepModel: output shape = {pt_out.shape}')
        
        # 保存结果
        results[f'sample_{count}_output'] = pt_out.numpy()
        results[f'sample_{count}_num_agents'] = np.array([num_agents])
        
        # 保存输入用于 TensorRT
        for k, v in step_inputs.items():
            if isinstance(v, torch.Tensor):
                results[f'sample_{count}_{k}'] = v.numpy()
        
        count += 1
        print()
    
    np.savez(args.output, **results)
    print(f'✓ 保存到 {args.output}')
    print(f'✓ 共处理 {count} 个样本')


if __name__ == '__main__':
    main()
