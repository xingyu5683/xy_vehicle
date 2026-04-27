#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import ctypes
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os

print("Loading plugin...")
ctypes.CDLL('./build/libscatter_add_plugin.so')
trt.init_libnvinfer_plugins(trt.Logger(), 'planr1')

print("Loading engine...")
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

with open('./engines/step_fp16.trt', 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())

print(f'num_optimization_profiles: {engine.num_optimization_profiles}')

print("\nInputs:")
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    if mode == trt.TensorIOMode.INPUT:
        shape = engine.get_tensor_shape(name)
        print(f'  {name}: {shape}')

# 测试样本 0 和样本 1
data = np.load('/tmp/step_real_data.npz')

for sample_idx in [0, 1]:
    print(f"\n=== 测试样本 {sample_idx} ===")
    
    context = engine.create_execution_context()
    
    input_names = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
    
    # 准备输入
    device_inputs = {}
    for name in input_names:
        key = f'sample_{sample_idx}_{name}'
        arr = data[key]
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        arr = np.ascontiguousarray(arr)
        
        print(f'  设置 {name}: shape={arr.shape}')
        context.set_input_shape(name, arr.shape)
        
        device_inputs[name] = cuda.mem_alloc(arr.nbytes)
        cuda.memcpy_htod(device_inputs[name], arr)
        context.set_tensor_address(name, int(device_inputs[name]))
    
    # 检查所有输入是否正确设置
    all_ok = True
    for name in input_names:
        if not context.all_binding_shapes_specified:
            print(f"  警告: 形状未完全指定")
            all_ok = False
            break
    
    # 分配输出
    output_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.OUTPUT:
            output_name = name
            shape = context.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            print(f'  输出 {name}: shape={shape}')
            device_out = cuda.mem_alloc(np.empty(shape, dtype=dtype).nbytes)
            context.set_tensor_address(name, int(device_out))
    
    # 执行
    stream = cuda.Stream()
    success = context.execute_async_v3(stream.handle)
    stream.synchronize()
    print(f'  执行结果: {success}')
    
    # 获取输出
    shape = context.get_tensor_shape(output_name)
    dtype = trt.nptype(engine.get_tensor_dtype(output_name))
    output = np.empty(shape, dtype=dtype)
    cuda.memcpy_dtoh(output, device_out)
    
    print(f'  输出: nan={np.isnan(output).any()}, inf={np.isinf(output).any()}, min={output.min():.3f}, max={output.max():.3f}')
    
    # 释放
    for mem in device_inputs.values():
        mem.free()
    device_out.free()

os._exit(0)
