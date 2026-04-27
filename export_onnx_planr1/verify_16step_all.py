#!/usr/bin/env python3
"""
验证所有129个样本的16步自回归推理累积误差
"""

import sys
import json
import time
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path

sys.path.insert(0, '/workspace/planner/Plan-R1')
sys.path.insert(0, '/workspace/planner')

MAX_AGENTS = 21
MAX_POLYGONS = 145
MAX_POLYLINES = 1200
NUM_FUTURE_STEPS = 16
INTERVAL = 5


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def transform_to_local(point, origin, heading):
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    delta = point - origin
    local_x = delta[..., 0] * cos_h + delta[..., 1] * sin_h
    local_y = -delta[..., 0] * sin_h + delta[..., 1] * cos_h
    return np.stack([local_x, local_y], axis=-1)


def compute_edges(position, heading, valid_mask, polygon_pos, polygon_heading, num_agents, num_polygons, T):
    """计算所有边"""
    # k2k_t edges
    k2k_t_edges, k2k_t_attrs = [], []
    for a in range(num_agents):
        for t1 in range(T):
            if not valid_mask[a, t1]: continue
            for t2 in range(max(0, t1-6), t1+1):
                if not valid_mask[a, t2]: continue
                src, dst = a * T + t2, a * T + t1
                ev = transform_to_local(position[a,t2:t2+1], position[a,t1:t1+1], np.array([heading[a,t1]])).squeeze()
                length = np.linalg.norm(ev)
                theta = np.arctan2(ev[1], ev[0])
                hd = wrap_angle(heading[a,t2] - heading[a,t1])
                k2k_t_edges.append([src, dst])
                k2k_t_attrs.append([length, np.cos(theta), np.sin(theta), np.cos(hd), np.sin(hd), (t2-t1)*INTERVAL])
    
    if not k2k_t_edges:
        k2k_t_edges, k2k_t_attrs = [[0,0]], [[0,1,0,1,0,0]]
    
    # g2k edges
    g2k_edges, g2k_attrs = [], []
    for a in range(num_agents):
        for t in range(T):
            if not valid_mask[a, t]: continue
            for p in range(num_polygons):
                d = np.linalg.norm(polygon_pos[p] - position[a, t])
                if d < 30:
                    ev = transform_to_local(polygon_pos[p:p+1], position[a,t:t+1], np.array([heading[a,t]])).squeeze()
                    length = np.linalg.norm(ev)
                    theta = np.arctan2(ev[1], ev[0])
                    hd = wrap_angle(polygon_heading[p] - heading[a,t])
                    g2k_edges.append([p, a*T+t])
                    g2k_attrs.append([length, np.cos(theta), np.sin(theta), np.cos(hd), np.sin(hd), 1.0])
    
    if not g2k_edges:
        g2k_edges, g2k_attrs = [[0,0]], [[1,1,0,1,0,1]]
    
    # k2k_a edges
    k2k_a_edges, k2k_a_attrs = [], []
    for t in range(T):
        for a1 in range(num_agents):
            for a2 in range(num_agents):
                if not (valid_mask[a1,t] and valid_mask[a2,t]): continue
                d = np.linalg.norm(position[a1,t] - position[a2,t])
                if d < 60:
                    ev = transform_to_local(position[a1,t:t+1], position[a2,t:t+1], np.array([heading[a2,t]])).squeeze()
                    length = np.linalg.norm(ev)
                    theta = np.arctan2(ev[1], ev[0])
                    hd = wrap_angle(heading[a1,t] - heading[a2,t])
                    k2k_a_edges.append([t*MAX_AGENTS+a1, t*MAX_AGENTS+a2])
                    k2k_a_attrs.append([length, np.cos(theta), np.sin(theta), np.cos(hd), np.sin(hd)])
    
    if not k2k_a_edges:
        k2k_a_edges, k2k_a_attrs = [[0,0]], [[1,1,0,1,0]]
    
    return (np.array(k2k_t_edges, dtype=np.int64).T, np.array(k2k_t_attrs, dtype=np.float32),
            np.array(g2k_edges, dtype=np.int64).T, np.array(g2k_attrs, dtype=np.float32),
            np.array(k2k_a_edges, dtype=np.int64).T, np.array(k2k_a_attrs, dtype=np.float32))


def decode_token(tokens, token_dict, agent_type, current_pos, current_heading):
    N = tokens.shape[0]
    delta = np.zeros((N, 3), dtype=np.float32)
    type_map = {0: 'Vehicle', 1: 'Pedestrian', 2: 'Bicycle'}
    for i in range(N):
        delta[i] = token_dict[type_map[int(agent_type[i])]][tokens[i]]
    
    cos_h, sin_h = np.cos(current_heading), np.sin(current_heading)
    dx = delta[:,0] * cos_h - delta[:,1] * sin_h
    dy = delta[:,0] * sin_h + delta[:,1] * cos_h
    
    return current_pos + np.stack([dx, dy], axis=-1), wrap_angle(current_heading + delta[:,2])


def prepare_map_inputs(sample):
    num_polygons = sample['num_polygons']
    num_polylines = sample['num_polylines']
    SINK_POLYGON, SINK_POLYLINE = MAX_POLYGONS - 1, MAX_POLYLINES - 1
    
    inputs = {
        'polyline_position': np.zeros((MAX_POLYLINES, 2), dtype=np.float32),
        'polyline_heading': np.zeros(MAX_POLYLINES, dtype=np.float32),
        'polyline_length': np.ones(MAX_POLYLINES, dtype=np.float32),
        'polygon_position': np.zeros((MAX_POLYGONS, 2), dtype=np.float32),
        'polygon_heading': np.zeros(MAX_POLYGONS, dtype=np.float32),
        'polygon_speed_limit': np.zeros(MAX_POLYGONS, dtype=np.float32),
        'polygon_speed_limit_valid': np.zeros(MAX_POLYGONS, dtype=np.float32),
        'polygon_type': np.zeros(MAX_POLYGONS, dtype=np.int64),
        'polygon_traffic_light': np.full(MAX_POLYGONS, 4, dtype=np.int64),
        'polygon_on_route': np.zeros(MAX_POLYGONS, dtype=np.int64),
    }
    
    inputs['polyline_position'][:num_polylines] = sample['polyline_position'][:num_polylines].numpy()
    inputs['polyline_heading'][:num_polylines] = sample['polyline_heading'][:num_polylines].numpy()
    inputs['polyline_length'][:num_polylines] = sample['polyline_length'][:num_polylines].numpy()
    inputs['polygon_position'][:num_polygons] = sample['polygon_position'][:num_polygons].numpy()
    inputs['polygon_heading'][:num_polygons] = sample['polygon_heading'][:num_polygons].numpy()
    inputs['polygon_speed_limit'][:num_polygons] = sample['polygon_speed_limit'][:num_polygons].numpy()
    inputs['polygon_speed_limit_valid'][:num_polygons] = 1.0
    inputs['polygon_type'][:num_polygons] = sample['polygon_type'][:num_polygons].numpy()
    inputs['polygon_traffic_light'][:num_polygons] = sample['polygon_traffic_light'][:num_polygons].numpy()
    inputs['polygon_on_route'][:num_polygons] = sample['polygon_on_route'][:num_polygons].numpy().astype(np.int64)
    
    edge_l2g = sample['edge_polyline_to_polygon']
    n_l2g = min(edge_l2g.shape[1], MAX_POLYLINES)
    l2g = np.full((2, MAX_POLYLINES), SINK_POLYGON, dtype=np.int64)
    l2g[0] = SINK_POLYLINE
    l2g[:, :n_l2g] = edge_l2g.numpy()[:, :n_l2g]
    inputs['l2g_edge_index'] = l2g
    
    for name, key in [('left_edge_index', 'edge_left'), ('right_edge_index', 'edge_right')]:
        edge = np.full((2, 80), SINK_POLYGON, dtype=np.int64)
        if key in sample:
            n = min(sample[key].shape[1], 80)
            edge[:, :n] = sample[key].numpy()[:, :n]
        inputs[name] = edge
    
    inputs['incoming_edge_index'] = np.full((2, 80), SINK_POLYGON, dtype=np.int64)
    inputs['outgoing_edge_index'] = np.full((2, 80), SINK_POLYGON, dtype=np.int64)
    
    return inputs


def run_16step(sample, map_session, step_session, step_model_pt, token_dict, onnx_input_names):
    """运行16步自回归，返回每步的误差"""
    num_agents = sample['num_agents']
    num_polygons = sample['num_polygons']
    
    # 获取 polygon_embs
    polygon_embs = map_session.run(None, prepare_map_inputs(sample))[0]
    
    # 初始化状态
    position = sample['agent_position'][:num_agents, :4].numpy().copy()
    heading = sample['agent_heading'][:num_agents, :4].numpy().copy()
    valid = sample['agent_visible_mask'][:num_agents, :4].numpy().copy()
    agent_type = sample['agent_type'][:num_agents].numpy()
    agent_box = sample['agent_box'][:num_agents].numpy()
    agent_identity = sample['agent_identity'][:num_agents].numpy()
    
    polygon_pos = sample['polygon_position'][:num_polygons].numpy()
    polygon_heading_arr = sample['polygon_heading'][:num_polygons].numpy()
    
    np.random.seed(42)
    token = np.random.randint(0, 1024, (num_agents, 4))
    
    # 扩展到 MAX_AGENTS
    pos_full = np.zeros((MAX_AGENTS, 4, 2), dtype=np.float32)
    head_full = np.zeros((MAX_AGENTS, 4), dtype=np.float32)
    valid_full = np.zeros((MAX_AGENTS, 4), dtype=bool)
    token_full = np.zeros((MAX_AGENTS, 4), dtype=np.int64)
    type_full = np.zeros(MAX_AGENTS, dtype=np.int64)
    box_full = np.ones((MAX_AGENTS, 4), dtype=np.float32)
    id_full = np.ones(MAX_AGENTS, dtype=np.int64)
    
    pos_full[:num_agents] = position
    head_full[:num_agents] = heading
    valid_full[:num_agents] = valid
    token_full[:num_agents] = token
    type_full[:num_agents] = agent_type
    box_full[:num_agents] = agent_box
    id_full[:num_agents] = agent_identity
    
    step_results = []
    
    for step in range(NUM_FUTURE_STEPS):
        T = 4
        
        # 计算边
        k2k_t_idx, k2k_t_attr, g2k_idx, g2k_attr, k2k_a_idx, k2k_a_attr = compute_edges(
            pos_full[:num_agents], head_full[:num_agents], valid_full[:num_agents],
            polygon_pos, polygon_heading_arr, num_agents, num_polygons, T
        )
        
        inputs = {
            'agent_token': token_full,
            'agent_type': type_full,
            'agent_box': box_full,
            'agent_identity': id_full,
            'polygon_embs': polygon_embs,
            'k2k_t_edge_index': k2k_t_idx,
            'k2k_t_edge_attr': k2k_t_attr,
            'g2k_edge_index': g2k_idx,
            'g2k_edge_attr': g2k_attr,
            'k2k_a_edge_index': k2k_a_idx,
            'k2k_a_edge_attr': k2k_a_attr,
            'num_agents': np.array([num_agents], dtype=np.int64),
            'num_intervals': np.array([T], dtype=np.int64),
        }
        
        # PyTorch
        pt_inputs = {k: torch.from_numpy(v) for k, v in inputs.items()}
        with torch.no_grad():
            pt_logits = step_model_pt(**pt_inputs).numpy()
        
        # ONNX
        onnx_inputs = {k: v for k, v in inputs.items() if k in onnx_input_names}
        onnx_logits = step_session.run(None, onnx_inputs)[0]
        
        # 比较
        diff = np.abs(pt_logits[:num_agents] - onnx_logits[:num_agents])
        pt_tokens = np.argmax(pt_logits[:num_agents], axis=-1)
        onnx_tokens = np.argmax(onnx_logits[:num_agents], axis=-1)
        
        step_results.append({
            'step': step,
            'max_diff': float(diff.max()),
            'mean_diff': float(diff.mean()),
            'token_match': bool(np.all(pt_tokens == onnx_tokens)),
        })
        
        # 解码并更新状态（滑动窗口）
        next_pos, next_head = decode_token(onnx_tokens, token_dict, agent_type,
                                           pos_full[:num_agents, -1], head_full[:num_agents, -1])
        
        pos_full[:num_agents, :3] = pos_full[:num_agents, 1:4]
        pos_full[:num_agents, 3] = next_pos
        head_full[:num_agents, :3] = head_full[:num_agents, 1:4]
        head_full[:num_agents, 3] = next_head
        valid_full[:num_agents, :3] = valid_full[:num_agents, 1:4]
        valid_full[:num_agents, 3] = True
        token_full[:num_agents, :3] = token_full[:num_agents, 1:4]
        token_full[:num_agents, 3] = onnx_tokens
    
    return step_results


def main():
    print("=" * 70)
    print("验证所有129个样本的16步自回归累积误差")
    print("=" * 70, flush=True)
    
    # 加载数据
    data_path = Path('/workspace/planner/export_onnx_planr1/planr1_merged_realtime21.pt')
    print(f"加载数据: {data_path}", flush=True)
    data_list = torch.load(data_path, map_location='cpu')
    print(f"样本数: {len(data_list)}", flush=True)
    
    # 加载 token 字典
    token_dict_path = '/workspace/planner/export_onnx_planr1/tokens_1024.json'
    with open(token_dict_path, 'r') as f:
        token_dict_raw = json.load(f)
    token_dict = {k: np.array(v, dtype=np.float32) for k, v in token_dict_raw.items()}
    print(f"Token字典加载完成", flush=True)
    
    # 加载模型
    onnx_dir = Path('/workspace/planner/export_onnx_planr1')
    map_session = ort.InferenceSession(str(onnx_dir / 'map_encoder.onnx'), providers=['CPUExecutionProvider'])
    step_session = ort.InferenceSession(str(onnx_dir / 'step.onnx'), providers=['CPUExecutionProvider'])
    onnx_input_names = [inp.name for inp in step_session.get_inputs()]
    print("✓ ONNX 模型加载完成", flush=True)
    
    from export_onnx_planr1.step_exportable import StepModel
    from model.PlanR1 import PlanR1
    
    ckpt = torch.load('/workspace/planner/Plan-R1/ckpts/fine-tuning.ckpt', map_location='cpu')
    hparams = ckpt.get('hyper_parameters', {})
    original_model = PlanR1(mode='pred', token_dict_path='/workspace/planner/Plan-R1/tokens/tokens_1024.pt',
                            **{k: v for k, v in hparams.items() if k not in ['mode', 'token_dict_path']})
    original_model.load_state_dict(ckpt['state_dict'], strict=False)
    
    step_model = StepModel(hidden_dim=128, num_tokens=1024, num_attn_layers=6, num_heads=8, dropout=0.1)
    step_model.load_weights_from(original_model.pred_backbone, original_model.pred_decoder_head)
    step_model.eval()
    print("✓ PyTorch 模型加载完成\n", flush=True)
    
    # 运行验证
    print("=" * 70)
    print("开始16步自回归验证...")
    print("=" * 70, flush=True)
    
    start_time = time.time()
    all_results = []
    failed = []
    
    for i, sample in enumerate(data_list):
        try:
            results = run_16step(sample, map_session, step_session, step_model, token_dict, onnx_input_names)
            all_results.append(results)
            
            mismatches = sum(1 for r in results if not r['token_match'])
            max_diff = max(r['max_diff'] for r in results)
            
            if mismatches > 0:
                print(f"✗ Sample {i:3d}: {mismatches} token mismatches", flush=True)
                failed.append(i)
            elif (i + 1) % 10 == 0 or i == len(data_list) - 1:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (len(data_list) - i - 1)
                print(f"✓ Progress: {i+1}/{len(data_list)} ({elapsed:.0f}s, ETA: {eta:.0f}s)", flush=True)
        except Exception as e:
            print(f"✗ Sample {i:3d}: ERROR - {str(e)[:40]}", flush=True)
            failed.append(i)
    
    total_time = time.time() - start_time
    print(f"\n总耗时: {total_time:.0f}s ({total_time/60:.1f}分钟)", flush=True)
    
    # 汇总统计
    print("\n" + "=" * 70)
    print("每步误差统计")
    print("=" * 70, flush=True)
    
    for step in range(NUM_FUTURE_STEPS):
        max_diffs = [r[step]['max_diff'] for r in all_results]
        matches = [r[step]['token_match'] for r in all_results]
        print(f"  Step {step:2d}: avg={np.mean(max_diffs):.2e}, max={np.max(max_diffs):.2e}, match={sum(matches)/len(matches)*100:.0f}%", flush=True)
    
    # 累积趋势
    print("\n" + "=" * 70)
    print("累积误差趋势")
    print("=" * 70, flush=True)
    step0 = np.mean([r[0]['max_diff'] for r in all_results])
    step15 = np.mean([r[15]['max_diff'] for r in all_results])
    print(f"  Step 0  平均最大差异: {step0:.2e}")
    print(f"  Step 15 平均最大差异: {step15:.2e}")
    print(f"  累积增长倍数: {step15/step0:.2f}x", flush=True)
    
    # 总体
    all_max = [r[s]['max_diff'] for r in all_results for s in range(NUM_FUTURE_STEPS)]
    all_match = [r[s]['token_match'] for r in all_results for s in range(NUM_FUTURE_STEPS)]
    
    print("\n" + "=" * 70)
    print("总体统计")
    print("=" * 70)
    print(f"  验证样本数: {len(all_results)}")
    print(f"  失败样本数: {len(failed)}")
    print(f"  总推理步数: {len(all_results) * NUM_FUTURE_STEPS}")
    print(f"  全局最大差异: {np.max(all_max):.2e}")
    print(f"  全局平均差异: {np.mean(all_max):.2e}")
    print(f"  Token总匹配率: {sum(all_match)/len(all_match)*100:.2f}%", flush=True)
    
    if failed:
        print(f"\n失败样本: {failed}", flush=True)
    else:
        print("\n✓ 所有样本验证通过!", flush=True)


if __name__ == '__main__':
    main()
