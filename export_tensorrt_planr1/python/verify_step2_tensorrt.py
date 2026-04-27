#!/usr/bin/env python3
"""步骤2: 用 TensorRT 推理并与 PyTorch 比较"""

import argparse
import ctypes
import sys

import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError as e:
    print(f"错误: {e}")
    print("安装: pip install tensorrt pycuda")
    sys.exit(1)


class TRTInference:
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
            data = inputs[name]
            if data.dtype == np.float64:
                data = data.astype(np.float32)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/tmp/pytorch_outputs.npz')
    parser.add_argument('--trt', default='./engines/map_encoder_fp16.trt')
    parser.add_argument('--plugin', default='./build/libscatter_add_plugin.so')
    args = parser.parse_args()
    
    print('加载 TensorRT 引擎...')
    trt_engine = TRTInference(args.trt, args.plugin)
    print(f'✓ 引擎加载成功')
    print(f'  输入: {trt_engine.input_names}')
    print(f'  输出: {trt_engine.output_names}')
    
    print('\n加载 PyTorch 参考输出...')
    data = np.load(args.input)
    
    # 计算样本数量
    sample_count = 0
    while f'sample_{sample_count}_output' in data:
        sample_count += 1
    print(f'✓ 找到 {sample_count} 个样本')
    
    print('\n' + '=' * 60)
    print('验证结果')
    print('=' * 60)
    
    passed = 0
    failed = 0
    max_diffs = []
    
    for i in range(sample_count):
        # 准备输入
        inputs = {}
        for name in trt_engine.input_names:
            key = f'sample_{i}_{name}'
            if key in data:
                inputs[name] = data[key]
            else:
                print(f'警告: 缺少输入 {name}')
        
        # TensorRT 推理
        trt_out = trt_engine.infer(inputs)
        output_name = list(trt_out.keys())[0]
        trt_output = trt_out[output_name]
        
        # PyTorch 参考输出
        pt_output = data[f'sample_{i}_output']
        num_polygons = int(data[f'sample_{i}_num_polygons'][0])
        
        # 只比较有效部分
        trt_valid = trt_output[:num_polygons]
        pt_valid = pt_output[:num_polygons]
        
        # 检查 NaN
        has_nan_trt = np.isnan(trt_valid).any()
        has_nan_pt = np.isnan(pt_valid).any()
        
        if has_nan_trt or has_nan_pt:
            print(f'✗ 样本 {i}: NaN (TRT={has_nan_trt}, PT={has_nan_pt})')
            failed += 1
            continue
        
        # 计算差异
        diff = np.abs(trt_valid.astype(np.float32) - pt_valid.astype(np.float32))
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        # 相对误差
        scale = max(np.abs(pt_valid).max(), 1e-8)
        rel_diff = max_diff / scale * 100  # 百分比
        
        max_diffs.append(max_diff)
        
        if rel_diff < 1.0:  # 1% 以内为通过
            print(f'✓ 样本 {i}: max_diff={max_diff:.6f}, rel={rel_diff:.2f}%')
            passed += 1
        elif rel_diff < 5.0:  # 5% 以内为警告
            print(f'⚠ 样本 {i}: max_diff={max_diff:.6f}, rel={rel_diff:.2f}% (FP16 精度损失)')
            passed += 1
        else:
            print(f'✗ 样本 {i}: max_diff={max_diff:.6f}, rel={rel_diff:.2f}%')
            failed += 1
    
    print()
    print('=' * 60)
    print('总结')
    print('=' * 60)
    print(f'通过: {passed}, 失败: {failed}')
    
    if max_diffs:
        print(f'平均最大差异: {np.mean(max_diffs):.6f}')
        print(f'最大最大差异: {np.max(max_diffs):.6f}')
    
    if failed == 0:
        print('\n✓ 验证通过!')
        return 0
    else:
        print('\n✗ 验证失败')
        return 1


if __name__ == '__main__':
    sys.exit(main())
