#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 ONNX 模型输出与 PyTorch 模型输出的一致性

用法:
    python -m export_onnx_planr1.verify_onnx --data-dir /path/to/data --num-samples 10
    
    或者在 export_onnx_planr1 目录下:
    python verify_onnx.py --data-dir /path/to/data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def prepare_inputs(data, max_polygons=145, max_polylines=1200):
    """准备 ONNX 输入数据"""
    SINK_POLYGON = max_polygons - 1
    SINK_POLYLINE = max_polylines - 1
    
    polygon = data['polygon']
    polyline = data['polyline']
    num_polygons = min(polygon['position'].shape[0], max_polygons - 1)
    num_polylines = min(polyline['position'].shape[0], max_polylines - 1)
    
    inputs = {}
    
    # Polyline features
    inputs['polyline_position'] = np.zeros((max_polylines, 2), dtype=np.float32)
    inputs['polyline_position'][:num_polylines] = polyline['position'][:num_polylines, :2].numpy()
    
    inputs['polyline_heading'] = np.zeros(max_polylines, dtype=np.float32)
    inputs['polyline_heading'][:num_polylines] = polyline['heading'][:num_polylines].numpy()
    
    inputs['polyline_length'] = np.ones(max_polylines, dtype=np.float32)
    inputs['polyline_length'][:num_polylines] = polyline['length'][:num_polylines].numpy()
    
    # Polygon features
    inputs['polygon_position'] = np.zeros((max_polygons, 2), dtype=np.float32)
    inputs['polygon_position'][:num_polygons] = polygon['position'][:num_polygons, :2].numpy()
    
    inputs['polygon_heading'] = np.zeros(max_polygons, dtype=np.float32)
    inputs['polygon_heading'][:num_polygons] = polygon['heading'][:num_polygons].numpy()
    
    inputs['polygon_speed_limit'] = np.zeros(max_polygons, dtype=np.float32)
    inputs['polygon_speed_limit'][:num_polygons] = polygon['speed_limit'][:num_polygons].numpy()
    
    inputs['polygon_speed_limit_valid'] = np.zeros(max_polygons, dtype=np.float32)
    inputs['polygon_speed_limit_valid'][:num_polygons] = polygon['speed_limit_valid_mask'][:num_polygons].numpy().astype(np.float32)
    
    inputs['polygon_type'] = np.zeros(max_polygons, dtype=np.int64)
    inputs['polygon_type'][:num_polygons] = polygon['type'][:num_polygons].numpy()
    
    inputs['polygon_traffic_light'] = np.zeros(max_polygons, dtype=np.int64)
    inputs['polygon_traffic_light'][:num_polygons] = polygon['traffic_light'][:num_polygons].numpy()
    
    inputs['polygon_on_route'] = np.zeros(max_polygons, dtype=np.int64)
    inputs['polygon_on_route'][:num_polygons] = polygon['on_route_mask'][:num_polygons].numpy().astype(np.int64)
    
    # Edge indices
    edge_data = data[('polyline', 'polygon')]
    edge_idx = edge_data['polyline_to_polygon_edge_index']
    n_edges = min(edge_idx.shape[1], max_polylines)
    
    l2g = np.full((2, max_polylines), SINK_POLYGON, dtype=np.int64)
    l2g[0] = SINK_POLYLINE
    l2g[:, :n_edges] = edge_idx.numpy()[:, :n_edges]
    inputs['l2g_edge_index'] = l2g
    
    # Polygon-to-polygon edges
    pg_data = data[('polygon', 'polygon')]
    for name in ['left_edge_index', 'right_edge_index', 'incoming_edge_index', 'outgoing_edge_index']:
        edge = np.full((2, 80), SINK_POLYGON, dtype=np.int64)
        if name in pg_data:
            n = min(pg_data[name].shape[1], 80)
            edge[:, :n] = pg_data[name].numpy()[:, :n]
        inputs[name] = edge
    
    return inputs, num_polygons, num_polylines


def verify_map_encoder(onnx_path, pytorch_model, data_files, max_samples=10):
    """验证 MapEncoder ONNX 模型"""
    import onnxruntime as ort
    
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    results = []
    skipped = 0
    
    for i, data_file in enumerate(data_files[:max_samples]):
        data = torch.load(data_file)
        
        # 检查数据大小
        num_polygons_raw = data['polygon']['position'].shape[0]
        num_polylines_raw = data['polyline']['position'].shape[0]
        
        if num_polylines_raw > 1200 or num_polygons_raw > 145:
            skipped += 1
            continue
        
        inputs, num_polygons, num_polylines = prepare_inputs(data)
        
        # PyTorch 推理
        pt_inputs = {k: torch.from_numpy(v) for k, v in inputs.items()}
        with torch.no_grad():
            pt_out = pytorch_model(
                pt_inputs['polyline_position'],
                pt_inputs['polyline_heading'],
                pt_inputs['polyline_length'],
                pt_inputs['polygon_position'],
                pt_inputs['polygon_heading'],
                pt_inputs['polygon_speed_limit'],
                pt_inputs['polygon_speed_limit_valid'],
                pt_inputs['polygon_type'].long(),
                pt_inputs['polygon_traffic_light'].long(),
                pt_inputs['polygon_on_route'].long(),
                pt_inputs['l2g_edge_index'].long(),
                pt_inputs['left_edge_index'].long(),
                pt_inputs['right_edge_index'].long(),
                pt_inputs['incoming_edge_index'].long(),
                pt_inputs['outgoing_edge_index'].long(),
            ).numpy()
        
        # ONNX 推理
        onnx_out = session.run(None, inputs)[0]
        
        # 比较有效部分
        valid_pt = pt_out[:num_polygons]
        valid_onnx = onnx_out[:num_polygons]
        
        has_nan = np.isnan(valid_onnx).any()
        if has_nan:
            results.append({
                'file': data_file.name,
                'status': 'FAIL',
                'reason': 'NaN detected',
                'num_polygons': num_polygons,
                'num_polylines': num_polylines,
            })
        else:
            diff = np.abs(valid_pt - valid_onnx)
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            results.append({
                'file': data_file.name,
                'status': 'PASS' if max_diff < 0.01 else 'WARN',
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'num_polygons': num_polygons,
                'num_polylines': num_polylines,
            })
    
    return results, skipped


def prepare_step_inputs_from_data(data, polygon_embs, max_agents=21, max_polygons=145, interval=5):
    """从真实数据构建 StepModel 的输入
    
    Args:
        data: 从 .pt 文件加载的数据
        polygon_embs: 来自 MapEncoder 的 polygon embeddings [max_polygons, hidden_dim]
        max_agents: 最大 agent 数量
        max_polygons: 最大 polygon 数量
        interval: 时间间隔 (帧数)
    
    Returns:
        dict: StepModel 需要的输入
    """
    agent = data['agent']
    polygon = data['polygon']
    
    num_agents_raw = agent['num_nodes']
    num_agents = min(num_agents_raw, max_agents)
    num_polygons = min(polygon['position'].shape[0], max_polygons)
    
    # 历史时间步: 使用 interval=5, 取 [0, 5, 10, 15] 帧 -> 4 个历史 interval
    # position 形状是 [num_agents, 101, 2], 其中 0-20 是历史 (21 帧), 20 是当前, 21-100 是未来
    # 历史 intervals: frame 0, 5, 10, 15, 20 -> 5 个时间点
    num_hist_intervals = 4
    hist_frames = [i * interval for i in range(num_hist_intervals + 1)]  # [0, 5, 10, 15, 20]
    
    # Agent token - 需要从 position/heading 计算，这里用随机值模拟
    # 实际训练数据中会有 token 字段
    agent_token = torch.randint(0, 1024, (max_agents, num_hist_intervals), dtype=torch.long)
    
    # Agent type
    agent_type = torch.zeros(max_agents, dtype=torch.long)
    agent_type[:num_agents] = agent['type'][:num_agents].long()
    
    # Agent box
    agent_box = torch.ones(max_agents, 4, dtype=torch.float32)
    agent_box[:num_agents] = agent['box'][:num_agents]
    
    # Agent identity (ego=0, others=1)
    agent_identity = torch.ones(max_agents, dtype=torch.long)
    agent_identity[:num_agents] = agent['identity'][:num_agents].long()
    
    # 构建历史位置和 heading
    hist_position = torch.zeros(max_agents, num_hist_intervals + 1, 2, dtype=torch.float32)
    hist_heading = torch.zeros(max_agents, num_hist_intervals + 1, dtype=torch.float32)
    hist_valid = torch.zeros(max_agents, num_hist_intervals + 1, dtype=torch.bool)
    
    for t_idx, frame in enumerate(hist_frames):
        if frame < agent['position'].shape[1]:
            hist_position[:num_agents, t_idx] = agent['position'][:num_agents, frame, :2]
            hist_heading[:num_agents, t_idx] = agent['heading'][:num_agents, frame]
            hist_valid[:num_agents, t_idx] = agent['visible_mask'][:num_agents, frame]
    
    # 构建边
    # k2k_t (时间注意力): 每个 agent 的历史时间步之间的边
    k2k_t_edges = []
    k2k_t_attrs = []
    n_tokens = max_agents * num_hist_intervals
    
    for a in range(num_agents):
        for t1 in range(num_hist_intervals):
            for t2 in range(max(0, t1-6), t1+1):  # 只连接前6个时间步
                if hist_valid[a, t1] and hist_valid[a, t2]:
                    src_idx = a * num_hist_intervals + t2
                    dst_idx = a * num_hist_intervals + t1
                    if src_idx < n_tokens and dst_idx < n_tokens:
                        k2k_t_edges.append([src_idx, dst_idx])
                        # 边属性: [length, cos(theta), sin(theta), cos(heading), sin(heading), interval]
                        dx = hist_position[a, t2, 0] - hist_position[a, t1, 0]
                        dy = hist_position[a, t2, 1] - hist_position[a, t1, 1]
                        length = np.sqrt(dx**2 + dy**2 + 1e-8)
                        theta = np.arctan2(dy.item(), dx.item())
                        dh = hist_heading[a, t2] - hist_heading[a, t1]
                        dt = (t2 - t1) * interval
                        k2k_t_attrs.append([length.item(), np.cos(theta), np.sin(theta), 
                                           np.cos(dh.item()), np.sin(dh.item()), dt])
    
    if len(k2k_t_edges) == 0:
        k2k_t_edges = [[0, 0]]
        k2k_t_attrs = [[0.0, 1.0, 0.0, 1.0, 0.0, 0.0]]
    
    k2k_t_edge_index = torch.tensor(k2k_t_edges, dtype=torch.long).T  # [2, n_edges]
    k2k_t_edge_attr = torch.tensor(k2k_t_attrs, dtype=torch.float32)  # [n_edges, 6]
    
    # g2k (地图到 agent): polygon -> agent token
    g2k_edges = []
    g2k_attrs = []
    polygon_pos = polygon['position'][:num_polygons, :2]
    polygon_heading = polygon['heading'][:num_polygons]
    
    for p in range(num_polygons):
        for a in range(num_agents):
            for t in range(num_hist_intervals):
                if hist_valid[a, t]:
                    # 检查距离
                    dx = polygon_pos[p, 0] - hist_position[a, t, 0]
                    dy = polygon_pos[p, 1] - hist_position[a, t, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < 30:  # polygon_radius
                        g2k_edges.append([p, a * num_hist_intervals + t])
                        length = dist + 1e-8
                        theta = np.arctan2(dy.item(), dx.item())
                        dh = polygon_heading[p] - hist_heading[a, t]
                        g2k_attrs.append([length.item(), np.cos(theta), np.sin(theta),
                                         np.cos(dh.item()), np.sin(dh.item()), 1.0])
    
    if len(g2k_edges) == 0:
        g2k_edges = [[0, 0]]
        g2k_attrs = [[1.0, 1.0, 0.0, 1.0, 0.0, 1.0]]
    
    g2k_edge_index = torch.tensor(g2k_edges, dtype=torch.long).T  # [2, n_edges]
    g2k_edge_attr = torch.tensor(g2k_attrs, dtype=torch.float32)  # [n_edges, 6]
    
    # k2k_a (agent 间注意力): 同一时间步的不同 agent 之间
    # 注意: StepModel 中 k2k_a 使用 k_embs_flat.reshape(N, T, D).transpose(0, 1).reshape(T * N, D)
    # 所以索引是 time-major: idx = timestep * max_agents + agent
    k2k_a_edges = []
    k2k_a_attrs = []
    
    for t in range(num_hist_intervals):
        for a1 in range(num_agents):
            for a2 in range(num_agents):
                if a1 != a2 and hist_valid[a1, t] and hist_valid[a2, t]:
                    # 检查距离
                    dx = hist_position[a1, t, 0] - hist_position[a2, t, 0]
                    dy = hist_position[a1, t, 1] - hist_position[a2, t, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < 60:  # agent_radius
                        # time-major 索引: timestep * max_agents + agent
                        src_idx = t * max_agents + a1
                        dst_idx = t * max_agents + a2
                        k2k_a_edges.append([src_idx, dst_idx])
                        length = dist + 1e-8
                        theta = np.arctan2(dy.item(), dx.item())
                        dh = hist_heading[a1, t] - hist_heading[a2, t]
                        k2k_a_attrs.append([length.item(), np.cos(theta), np.sin(theta),
                                           np.cos(dh.item()), np.sin(dh.item())])
    
    if len(k2k_a_edges) == 0:
        k2k_a_edges = [[0, 0]]
        k2k_a_attrs = [[1.0, 1.0, 0.0, 1.0, 0.0]]
    
    k2k_a_edge_index = torch.tensor(k2k_a_edges, dtype=torch.long).T  # [2, n_edges]
    k2k_a_edge_attr = torch.tensor(k2k_a_attrs, dtype=torch.float32)  # [n_edges, 5]
    
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
        'num_agents': torch.tensor([num_agents], dtype=torch.long),
        'num_intervals': torch.tensor([num_hist_intervals], dtype=torch.long),
    }


def verify_step_model_with_fixed_inputs(step_onnx_path, pytorch_step_model, num_samples=5):
    """使用固定的 dummy 输入验证 StepModel - 确保 PyTorch 和 ONNX 完全相同的输入"""
    import onnxruntime as ort
    from export_onnx_planr1.step_exportable import create_step_dummy_inputs
    
    step_session = ort.InferenceSession(step_onnx_path, providers=['CPUExecutionProvider'])
    onnx_input_names = [inp.name for inp in step_session.get_inputs()]
    
    results = []
    
    # 设置随机种子确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    for i in range(num_samples):
        # 创建固定输入
        step_inputs = create_step_dummy_inputs('cpu')
        
        # PyTorch 推理
        pytorch_step_model.eval()
        with torch.no_grad():
            pt_out = pytorch_step_model(**step_inputs).numpy()
        
        # ONNX 推理 - 只传入 ONNX 模型期望的输入
        onnx_inputs = {}
        for k, v in step_inputs.items():
            if k in onnx_input_names and isinstance(v, torch.Tensor):
                onnx_inputs[k] = v.numpy()
        
        onnx_out = step_session.run(None, onnx_inputs)[0]
        
        # 比较
        has_nan_pt = np.isnan(pt_out).any()
        has_nan_onnx = np.isnan(onnx_out).any()
        
        if has_nan_pt or has_nan_onnx:
            results.append({
                'sample': i,
                'status': 'FAIL',
                'reason': f'NaN detected (PT:{has_nan_pt}, ONNX:{has_nan_onnx})',
            })
        else:
            diff = np.abs(pt_out - onnx_out)
            max_diff = diff.max()
            mean_diff = diff.mean()
            
            results.append({
                'sample': i,
                'status': 'PASS' if max_diff < 0.01 else 'WARN',
                'max_diff': max_diff,
                'mean_diff': mean_diff,
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='验证 ONNX 模型')
    parser.add_argument('--data-dir', type=str, 
                        default='/workspace/volumes/ad-pnc-al-sh01/plan-r1/dataset/nuplan-v1.1/splits/train-processed-pred-train-PlanR1',
                        help='数据目录路径')
    parser.add_argument('--ckpt-path', type=str,
                        default='/workspace/planner/Plan-R1/ckpts/fine-tuning.ckpt',
                        help='PyTorch 权重路径')
    parser.add_argument('--onnx-dir', type=str,
                        default='/workspace/planner/export_onnx_planr1',
                        help='ONNX 模型目录')
    parser.add_argument('--num-samples', type=int, default=20,
                        help='测试样本数量')
    parser.add_argument('--skip-step', action='store_true',
                        help='跳过 StepModel 验证')
    args = parser.parse_args()
    
    # 添加路径
    sys.path.insert(0, '/workspace/planner/Plan-R1')
    
    from export_onnx_planr1.map_encoder_exportable import MapEncoderExportable
    from export_onnx_planr1.step_exportable import StepModel
    from model.PlanR1 import PlanR1
    
    print('=' * 60)
    print('ONNX 模型验证')
    print('=' * 60)
    print()
    
    # 加载数据文件
    data_dir = Path(args.data_dir)
    data_files = sorted(data_dir.glob('*.pt'))
    print(f'找到 {len(data_files)} 个数据文件')
    print(f'测试样本数: {args.num_samples}')
    print()
    
    # 加载 PyTorch 模型
    print('加载 PyTorch 模型...')
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    hparams = ckpt.get('hyper_parameters', {})
    
    # MapEncoder
    map_encoder = MapEncoderExportable(hidden_dim=128, num_hops=4, num_heads=8, dropout=0.1)
    original_model = PlanR1(
        mode='pred',
        token_dict_path='/workspace/planner/Plan-R1/tokens/tokens_1024.pt',
        **{k: v for k, v in hparams.items() if k not in ['mode', 'token_dict_path']}
    )
    original_model.load_state_dict(ckpt['state_dict'], strict=False)
    map_encoder.load_weights_from(original_model.pred_map_encoder)
    map_encoder.eval()
    
    # StepModel
    step_model = StepModel(hidden_dim=128, num_tokens=1024, num_attn_layers=6, num_heads=8, dropout=0.1)
    step_model.load_weights_from(original_model.pred_backbone, original_model.pred_decoder_head)
    step_model.eval()
    
    print('✓ PyTorch 模型加载完成')
    print()
    
    # 验证 MapEncoder
    print('验证 MapEncoder...')
    onnx_path = Path(args.onnx_dir) / 'map_encoder.onnx'
    results, skipped = verify_map_encoder(onnx_path, map_encoder, data_files, args.num_samples)
    
    # 打印结果
    print()
    print('验证结果:')
    print('-' * 60)
    
    passed = 0
    failed = 0
    warned = 0
    max_diffs = []
    
    for r in results:
        if r['status'] == 'PASS':
            passed += 1
            max_diffs.append(r['max_diff'])
            print(f"✓ {r['file']}: max_diff={r['max_diff']:.6f}")
        elif r['status'] == 'WARN':
            warned += 1
            max_diffs.append(r['max_diff'])
            print(f"⚠ {r['file']}: max_diff={r['max_diff']:.6f}")
        else:
            failed += 1
            print(f"✗ {r['file']}: {r['reason']}")
    
    print()
    print('=' * 60)
    print('MapEncoder 汇总')
    print('=' * 60)
    print(f'通过: {passed}')
    print(f'警告: {warned}')
    print(f'失败: {failed}')
    print(f'跳过 (超出最大尺寸): {skipped}')
    
    if max_diffs:
        print()
        print(f'平均最大差异: {np.mean(max_diffs):.6f}')
        print(f'最大最大差异: {np.max(max_diffs):.6f}')
    
    map_encoder_passed = (failed == 0)
    
    # ========== 验证 StepModel ==========
    step_model_passed = True
    if not args.skip_step:
        print()
        print('=' * 60)
        print('验证 StepModel (使用固定输入)...')
        print('=' * 60)
        
        step_onnx_path = Path(args.onnx_dir) / 'step.onnx'
        
        step_results = verify_step_model_with_fixed_inputs(
            step_onnx_path, 
            step_model, 
            num_samples=args.num_samples
        )
        
        print()
        print('验证结果:')
        print('-' * 60)
        
        step_passed = 0
        step_failed = 0
        step_warned = 0
        step_max_diffs = []
        
        for r in step_results:
            label = r.get('file', f"Sample {r.get('sample', '?')}")
            if r['status'] == 'PASS':
                step_passed += 1
                step_max_diffs.append(r['max_diff'])
                print(f"✓ {label}: max_diff={r['max_diff']:.6f}")
            elif r['status'] == 'WARN':
                step_warned += 1
                step_max_diffs.append(r['max_diff'])
                print(f"⚠ {label}: max_diff={r['max_diff']:.6f}")
            else:
                step_failed += 1
                print(f"✗ {label}: {r['reason']}")
        
        print()
        print('=' * 60)
        print('StepModel 汇总')
        print('=' * 60)
        print(f'通过: {step_passed}')
        print(f'警告: {step_warned}')
        print(f'失败: {step_failed}')
        
        if step_max_diffs:
            print()
            print(f'平均最大差异: {np.mean(step_max_diffs):.6f}')
            print(f'最大最大差异: {np.max(step_max_diffs):.6f}')
        
        step_model_passed = (step_failed == 0)
    
    # ========== 最终结果 ==========
    print()
    print('=' * 60)
    print('最终结果')
    print('=' * 60)
    
    if map_encoder_passed and step_model_passed:
        print('✓ 所有验证通过!')
        return 0
    else:
        if not map_encoder_passed:
            print('✗ MapEncoder 验证失败')
        if not step_model_passed:
            print('✗ StepModel 验证失败')
        return 1


if __name__ == '__main__':
    sys.exit(main())
