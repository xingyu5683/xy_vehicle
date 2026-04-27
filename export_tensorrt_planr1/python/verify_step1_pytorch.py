#!/usr/bin/env python3
"""步骤1: 用 PyTorch 生成参考输出"""

import sys
sys.path.insert(0, '/workspace/planner/Plan-R1')
sys.path.insert(0, '/workspace/planner')

from pathlib import Path
import numpy as np
import torch

from export_onnx_planr1.map_encoder_exportable import MapEncoderExportable
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
    
    inputs['polyline_position'] = np.zeros((max_polylines, 2), dtype=np.float32)
    inputs['polyline_position'][:num_polylines] = polyline['position'][:num_polylines, :2].numpy()
    
    inputs['polyline_heading'] = np.zeros(max_polylines, dtype=np.float32)
    inputs['polyline_heading'][:num_polylines] = polyline['heading'][:num_polylines].numpy()
    
    inputs['polyline_length'] = np.ones(max_polylines, dtype=np.float32)
    inputs['polyline_length'][:num_polylines] = polyline['length'][:num_polylines].numpy()
    
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
    
    edge_data = data[('polyline', 'polygon')]
    edge_idx = edge_data['polyline_to_polygon_edge_index']
    n_edges = min(edge_idx.shape[1], max_polylines)
    
    l2g = np.full((2, max_polylines), SINK_POLYGON, dtype=np.int64)
    l2g[0] = SINK_POLYLINE
    l2g[:, :n_edges] = edge_idx.numpy()[:, :n_edges]
    inputs['l2g_edge_index'] = l2g
    
    pg_data = data[('polygon', 'polygon')]
    for name in ['left_edge_index', 'right_edge_index', 'incoming_edge_index', 'outgoing_edge_index']:
        edge = np.full((2, 80), SINK_POLYGON, dtype=np.int64)
        if name in pg_data:
            n = min(pg_data[name].shape[1], 80)
            edge[:, :n] = pg_data[name].numpy()[:, :n]
        inputs[name] = edge
    
    return inputs, num_polygons, num_polylines


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/workspace/volumes/ad-pnc-al-sh01/plan-r1/dataset/nuplan-v1.1/splits/train-processed-pred-train-PlanR1')
    parser.add_argument('--ckpt', default='/workspace/planner/Plan-R1/ckpts/fine-tuning.ckpt')
    parser.add_argument('--output', default='/tmp/pytorch_outputs.npz')
    parser.add_argument('--num-samples', type=int, default=3)
    args = parser.parse_args()
    
    print('加载 PyTorch 模型...')
    ckpt = torch.load(args.ckpt, map_location='cpu')
    hparams = ckpt.get('hyper_parameters', {})
    
    original_model = PlanR1(
        mode='pred',
        token_dict_path='/workspace/planner/Plan-R1/tokens/tokens_1024.pt',
        **{k: v for k, v in hparams.items() if k not in ['mode', 'token_dict_path']}
    )
    original_model.load_state_dict(ckpt['state_dict'], strict=False)
    
    map_encoder = MapEncoderExportable(hidden_dim=128, num_hops=4, num_heads=8, dropout=0.1)
    map_encoder.load_weights_from(original_model.pred_map_encoder)
    map_encoder.eval()
    
    print('✓ 模型加载完成')
    
    # 加载数据
    data_dir = Path(args.data_dir)
    data_files = sorted(data_dir.glob('*.pt'))
    print(f'找到 {len(data_files)} 个数据文件')
    
    results = {}
    count = 0
    
    for data_file in data_files[:args.num_samples * 3]:
        if count >= args.num_samples:
            break
            
        data = torch.load(data_file)
        
        num_polygons_raw = data['polygon']['position'].shape[0]
        num_polylines_raw = data['polyline']['position'].shape[0]
        
        if num_polylines_raw > 1200 or num_polygons_raw > 145:
            continue
        
        inputs, num_polygons, num_polylines = prepare_map_encoder_inputs(data)
        
        # PyTorch 推理
        pt_inputs = {k: torch.from_numpy(v) for k, v in inputs.items()}
        with torch.no_grad():
            pt_out = map_encoder(
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
        
        results[f'sample_{count}_output'] = pt_out
        results[f'sample_{count}_num_polygons'] = np.array([num_polygons])
        
        # 保存输入用于 TensorRT 验证
        for k, v in inputs.items():
            results[f'sample_{count}_{k}'] = v
        
        print(f'✓ 样本 {count}: {data_file.name}, polygons={num_polygons}')
        count += 1
    
    np.savez(args.output, **results)
    print(f'\n✓ 保存到 {args.output}')


if __name__ == '__main__':
    main()
