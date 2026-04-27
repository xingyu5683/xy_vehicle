#!/usr/bin/env python3
"""快速验证 TensorRT vs PyTorch"""

import os
import sys
import numpy as np
import ctypes

# 强制刷新输出
sys.stdout.reconfigure(line_buffering=True)

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def main():
    # 加载数据
    data = np.load('/tmp/pytorch_outputs.npz')
    print('✓ 数据加载完成')

    # 加载 TensorRT
    ctypes.CDLL('./build/libscatter_add_plugin.so')
    trt.init_libnvinfer_plugins(trt.Logger(), 'planr1')

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open('./engines/map_encoder_fp16.trt', 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    print('✓ TensorRT 引擎加载完成')

    # 获取 IO 名称
    input_names = []
    output_names = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)

    # 计算样本数
    sample_count = 0
    while f'sample_{sample_count}_output' in data:
        sample_count += 1
    print(f'✓ 找到 {sample_count} 个样本')

    print()
    print('=' * 60)
    print('验证结果')
    print('=' * 60)

    passed = 0
    failed = 0
    max_diffs = []

    for sample_idx in range(sample_count):
        # 准备输入
        inputs = {}
        for name in input_names:
            key = f'sample_{sample_idx}_{name}'
            if key in data:
                inputs[name] = data[key]

        # 分配设备内存并拷贝输入
        device_inputs = {}
        for name in input_names:
            arr = inputs[name]
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            arr = np.ascontiguousarray(arr)
            context.set_input_shape(name, arr.shape)
            device_inputs[name] = cuda.mem_alloc(arr.nbytes)
            cuda.memcpy_htod(device_inputs[name], arr)
            context.set_tensor_address(name, int(device_inputs[name]))

        # 分配输出
        device_outputs = {}
        for name in output_names:
            shape = context.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            device_outputs[name] = cuda.mem_alloc(np.empty(shape, dtype=dtype).nbytes)
            context.set_tensor_address(name, int(device_outputs[name]))

        # 执行
        stream = cuda.Stream()
        context.execute_async_v3(stream.handle)
        stream.synchronize()

        # 获取输出
        trt_output = None
        for name in output_names:
            shape = context.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            trt_output = np.empty(shape, dtype=dtype)
            cuda.memcpy_dtoh(trt_output, device_outputs[name])

        # 释放设备内存
        for mem in device_inputs.values():
            mem.free()
        for mem in device_outputs.values():
            mem.free()

        # 获取 PyTorch 参考输出
        pt_output = data[f'sample_{sample_idx}_output']
        num_polygons = int(data[f'sample_{sample_idx}_num_polygons'][0])

        # 只比较有效部分
        trt_valid = trt_output[:num_polygons]
        pt_valid = pt_output[:num_polygons]

        # 检查 NaN
        has_nan_trt = np.isnan(trt_valid).any()
        has_nan_pt = np.isnan(pt_valid).any()

        if has_nan_trt or has_nan_pt:
            print(f'✗ 样本 {sample_idx}: NaN (TRT={has_nan_trt}, PT={has_nan_pt})')
            failed += 1
            continue

        # 计算差异
        diff = np.abs(trt_valid.astype(np.float32) - pt_valid.astype(np.float32))
        max_diff = diff.max()
        mean_diff = diff.mean()

        scale = max(np.abs(pt_valid).max(), 1e-8)
        rel_diff = max_diff / scale * 100

        max_diffs.append(max_diff)

        if rel_diff < 1.0:
            print(f'✓ 样本 {sample_idx}: max_diff={max_diff:.6f}, rel={rel_diff:.2f}%')
            passed += 1
        elif rel_diff < 5.0:
            print(f'⚠ 样本 {sample_idx}: max_diff={max_diff:.6f}, rel={rel_diff:.2f}% (FP16精度)')
            passed += 1
        else:
            print(f'✗ 样本 {sample_idx}: max_diff={max_diff:.6f}, rel={rel_diff:.2f}%')
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
        print()
        print('✓ TensorRT vs PyTorch 验证通过!')
    else:
        print()
        print('✗ 验证失败')

    # 使用 os._exit 避免清理时的段错误
    os._exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
