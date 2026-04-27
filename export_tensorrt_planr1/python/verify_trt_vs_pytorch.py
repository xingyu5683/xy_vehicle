#!/usr/bin/env python3
"""
验证 TensorRT 引擎输出与 PyTorch 模型输出的一致性

用法:
    # 在 base 环境 (Python 3.10+) 运行:
    conda deactivate
    cd /workspace/planner/export_tensorrt_planr1
    python python/verify_trt_vs_pytorch.py --num-samples 5
"""

import argparse
import ctypes
import sys
from pathlib import Path

# 先导入 PyTorch (重要: 必须在 pycuda 之前)
import torch
import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError as e:
    print(f"错误: 缺少依赖包: {e}")
    print("安装: pip install tensorrt pycuda")
    sys.exit(1)


class TRTInference:
    """TensorRT 推理封装"""
    
    def __init__(self, engine_path: str, plugin_path: str = None):
        if plugin_path:
            ctypes.CDLL(plugin_path)
            trt.init_libnvinfer_plugins(trt.Logger(), "planr1")
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if not self.engine:
            raise RuntimeError(f"无法加载引擎: {engine_path}")
        
        self.context = self.engine.create_execution_context()
        
        self.input_names = []
        self.output_names = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
    
    def infer(self, inputs: dict) -> dict:
        device_inputs = {}
        device_outputs = {}
        
        for name in self.input_names:
            data = inputs[name].astype(np.float32) if inputs[name].dtype == np.float64 else inputs[name]
            data = np.ascontiguousarray(data)
            self.context.set_input_shape(name, data.shape)
            device_inputs[name] = cuda.mem_alloc(data.nbytes)
            cuda.memcpy_htod(device_inputs[name], data)
            self.context.set_tensor_address(name, int(device_inputs[name]))
        
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            output = np.empty(shape, dtype=dtype)
            device_outputs[name] = cuda.mem_alloc(output.nbytes)
            self.context.set_tensor_address(name, int(device_outputs[name]))
        
        self.context.execute_async_v3(cuda.Stream().handle)
        cuda.Context.synchronize()
        
        outputs = {}
        for name in self.output_names:
            shape = self.context.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            output = np.empty(shape, dtype=dtype)
            cuda.memcpy_dtoh(output, device_outputs[name])
            outputs[name] = output
        
        for mem in device_inputs.values():
            mem.free()
        for mem in device_outputs.values():
            mem.free()
        
        return outputs


def prepare_map_encoder_inputs(data, max_polygons=145, max_polylines=1200):
    """准备 MapEncoder 输入"""
    SINK_POLYGON = max_polygons - 1
    SINK_POLYLINE = max_polylines - 1
    
    polygon = data['polygon']
    polyline = data['polyline']
    num_polygons = min(polygon['position'].shape[0], max_polygons - 1)
    num_polylines = min(polyline['position'].shape[0], max_polylines - 1)
    
    inputs = {}
    
    # Polyline
    inputs['polyline_position'] = np.zeros((max_polylines, 2), dtype=np.float32)
    inputs['polyline_position'][:num_polylines] = polyline['position'][:num_polylines, :2].numpy()
    
    inputs['polyline_heading'] = np.zeros(max_polylines, dtype=np.float32)
    inputs['polyline_heading'][:num_polylines] = polyline['heading'][:num_polylines].numpy()
    
    inputs['polyline_length'] = np.ones(max_polylines, dtype=np.float32)
    inputs['polyline_length'][:num_polylines] = polyline['length'][:num_polylines].numpy()
    
    # Polygon
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
    
    # Edges
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


def compare_outputs(name, trt_out, pt_out, tolerance=0.01):
    """比较输出"""
    has_nan_trt = np.isnan(trt_out).any()
    has_nan_pt = np.isnan(pt_out).any()
    
    if has_nan_trt or has_nan_pt:
        return {
            'name': name,
            'status': 'FAIL',
            'reason': f'NaN (TRT:{has_nan_trt}, PT:{has_nan_pt})'
        }
    
    diff = np.abs(trt_out.astype(np.float32) - pt_out.astype(np.float32))
    max_diff = diff.max()
    mean_diff = diff.mean()
    
    scale = max(np.abs(pt_out).max(), 1e-8)
    rel_diff = max_diff / scale
    
    status = 'PASS' if rel_diff < tolerance else ('WARN' if rel_diff < 0.1 else 'FAIL')
    
    return {
        'name': name,
        'status': status,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'rel_diff': rel_diff * 100  # 百分比
    }


def verify_map_encoder(trt_engine, pytorch_model, data_files, max_samples=5):
    """验证 MapEncoder"""
    results = []
    skipped = 0
    
    for data_file in data_files[:max_samples * 3]:  # 多读一些，跳过太大的
        if len(results) >= max_samples:
            break
            
        data = torch.load(data_file)
        
        num_polygons_raw = data['polygon']['position'].shape[0]
        num_polylines_raw = data['polyline']['position'].shape[0]
        
        if num_polylines_raw > 1200 or num_polygons_raw > 145:
            skipped += 1
            continue
        
        inputs, num_polygons, num_polylines = prepare_map_encoder_inputs(data)
        
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
        
        # TensorRT 推理
        trt_out = trt_engine.infer(inputs)['polygon_embs']
        
        # 只比较有效部分
        result = compare_outputs(
            data_file.name, 
            trt_out[:num_polygons], 
            pt_out[:num_polygons]
        )
        result['num_polygons'] = num_polygons
        result['num_polylines'] = num_polylines
        results.append(result)
    
    return results, skipped


def verify_step_model(trt_engine, pytorch_model, num_samples=5):
    """验证 StepModel (使用 dummy 输入)"""
    from export_onnx_planr1.step_exportable import create_step_dummy_inputs
    
    results = []
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    for i in range(num_samples):
        # 创建 dummy 输入
        step_inputs = create_step_dummy_inputs('cpu')
        
        # PyTorch
        pytorch_model.eval()
        with torch.no_grad():
            pt_out = pytorch_model(**step_inputs).numpy()
        
        # TensorRT 输入
        trt_inputs = {k: v.numpy() if isinstance(v, torch.Tensor) else v 
                      for k, v in step_inputs.items() 
                      if k in trt_engine.input_names}
        
        trt_out = trt_engine.infer(trt_inputs)
        output_name = list(trt_out.keys())[0]
        
        result = compare_outputs(f'Sample_{i}', trt_out[output_name], pt_out)
        results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='验证 TensorRT vs PyTorch')
    parser.add_argument('--data-dir', type=str, 
                        default='/workspace/volumes/ad-pnc-al-sh01/plan-r1/dataset/nuplan-v1.1/splits/train-processed-pred-train-PlanR1')
    parser.add_argument('--ckpt-path', type=str,
                        default='/workspace/planner/Plan-R1/ckpts/fine-tuning.ckpt')
    parser.add_argument('--trt-dir', type=str, default='./engines')
    parser.add_argument('--plugin', type=str, default='./build/libscatter_add_plugin.so')
    parser.add_argument('--num-samples', type=int, default=5)
    parser.add_argument('--map-only', action='store_true', help='只验证 MapEncoder')
    parser.add_argument('--step-only', action='store_true', help='只验证 StepModel')
    args = parser.parse_args()
    
    # 添加路径
    sys.path.insert(0, '/workspace/planner/Plan-R1')
    sys.path.insert(0, '/workspace/planner')
    
    print('=' * 60)
    print('TensorRT vs PyTorch 验证')
    print('=' * 60)
    print()
    
    map_encoder_passed = True
    step_model_passed = True
    
    # ============ MapEncoder ============
    if not args.step_only:
        print('加载 MapEncoder...')
        
        from export_onnx_planr1.map_encoder_exportable import MapEncoderExportable
        from model.PlanR1 import PlanR1
        
        # PyTorch 模型
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        hparams = ckpt.get('hyper_parameters', {})
        
        original_model = PlanR1(
            mode='pred',
            token_dict_path='/workspace/planner/Plan-R1/tokens/tokens_1024.pt',
            **{k: v for k, v in hparams.items() if k not in ['mode', 'token_dict_path']}
        )
        original_model.load_state_dict(ckpt['state_dict'], strict=False)
        
        pt_map_encoder = MapEncoderExportable(hidden_dim=128, num_hops=4, num_heads=8, dropout=0.1)
        pt_map_encoder.load_weights_from(original_model.pred_map_encoder)
        pt_map_encoder.eval()
        
        # TensorRT
        trt_map_encoder = TRTInference(
            f'{args.trt_dir}/map_encoder_fp16.trt', 
            args.plugin
        )
        
        print('✓ MapEncoder 加载完成')
        print()
        
        # 加载数据
        data_dir = Path(args.data_dir)
        data_files = sorted(data_dir.glob('*.pt'))
        print(f'找到 {len(data_files)} 个数据文件')
        print()
        
        # 验证
        print('验证 MapEncoder...')
        results, skipped = verify_map_encoder(
            trt_map_encoder, pt_map_encoder, data_files, args.num_samples
        )
        
        print()
        print('结果:')
        print('-' * 60)
        
        passed = warned = failed = 0
        max_diffs = []
        
        for r in results:
            if r['status'] == 'PASS':
                passed += 1
                max_diffs.append(r['max_diff'])
                print(f"✓ {r['name']}: max_diff={r['max_diff']:.6f}, rel={r['rel_diff']:.2f}%")
            elif r['status'] == 'WARN':
                warned += 1
                max_diffs.append(r['max_diff'])
                print(f"⚠ {r['name']}: max_diff={r['max_diff']:.6f}, rel={r['rel_diff']:.2f}%")
            else:
                failed += 1
                print(f"✗ {r['name']}: {r.get('reason', 'FAIL')}")
        
        print()
        print(f'通过: {passed}, 警告: {warned}, 失败: {failed}, 跳过: {skipped}')
        if max_diffs:
            print(f'平均最大差异: {np.mean(max_diffs):.6f}')
        
        map_encoder_passed = (failed == 0)
    
    # ============ StepModel ============
    if not args.map_only:
        print()
        print('=' * 60)
        print('验证 StepModel...')
        print('=' * 60)
        print()
        
        from export_onnx_planr1.step_exportable import StepModel
        from model.PlanR1 import PlanR1
        
        # 如果还没加载 PyTorch 模型
        if args.step_only:
            ckpt = torch.load(args.ckpt_path, map_location='cpu')
            hparams = ckpt.get('hyper_parameters', {})
            original_model = PlanR1(
                mode='pred',
                token_dict_path='/workspace/planner/Plan-R1/tokens/tokens_1024.pt',
                **{k: v for k, v in hparams.items() if k not in ['mode', 'token_dict_path']}
            )
            original_model.load_state_dict(ckpt['state_dict'], strict=False)
        
        # PyTorch StepModel
        pt_step_model = StepModel(hidden_dim=128, num_tokens=1024, num_attn_layers=6, num_heads=8, dropout=0.1)
        pt_step_model.load_weights_from(original_model.pred_backbone, original_model.pred_decoder_head)
        pt_step_model.eval()
        
        # TensorRT
        trt_step_model = TRTInference(
            f'{args.trt_dir}/step_fp16.trt', 
            args.plugin
        )
        
        print('✓ StepModel 加载完成')
        print()
        
        # 验证
        results = verify_step_model(trt_step_model, pt_step_model, args.num_samples)
        
        print('结果:')
        print('-' * 60)
        
        passed = warned = failed = 0
        max_diffs = []
        
        for r in results:
            if r['status'] == 'PASS':
                passed += 1
                max_diffs.append(r['max_diff'])
                print(f"✓ {r['name']}: max_diff={r['max_diff']:.6f}, rel={r['rel_diff']:.2f}%")
            elif r['status'] == 'WARN':
                warned += 1
                max_diffs.append(r['max_diff'])
                print(f"⚠ {r['name']}: max_diff={r['max_diff']:.6f}, rel={r['rel_diff']:.2f}%")
            else:
                failed += 1
                print(f"✗ {r['name']}: {r.get('reason', 'FAIL')}")
        
        print()
        print(f'通过: {passed}, 警告: {warned}, 失败: {failed}')
        if max_diffs:
            print(f'平均最大差异: {np.mean(max_diffs):.6f}')
        
        step_model_passed = (failed == 0)
    
    # ============ 总结 ============
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
