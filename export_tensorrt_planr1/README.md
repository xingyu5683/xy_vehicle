# Plan-R1 TensorRT FP16 部署

这个文件夹包含将 Plan-R1 模型转换为 TensorRT FP16 引擎并进行高性能推理的完整方案。

## 📁 目录结构

```
export_tensorrt_planr1/
├── CMakeLists.txt           # 顶层 CMake 构建配置
├── README.md                # 本文档
├── plugins/                 # TensorRT 自定义插件
│   ├── scatter_add_kernel.h
│   ├── scatter_add_kernel.cu   # ScatterAdd CUDA kernel
│   ├── scatter_add_plugin.h
│   └── scatter_add_plugin.cpp  # TensorRT 插件实现
├── python/
│   ├── convert_to_tensorrt.py  # ONNX → TensorRT 转换脚本
│   └── verify_tensorrt.py      # TensorRT vs ONNX 验证
└── cpp/
    ├── include/
    │   └── planr1_tensorrt.h   # C++ 推理接口
    └── src/
        ├── planr1_tensorrt.cpp # C++ 推理实现
        └── main.cpp            # Demo 程序
```

## 🔧 为什么需要自定义插件？

TensorRT **不支持** ONNX 的 `ScatterElements` 算子与 `reduction='add'` 属性的组合（即 `scatter_add` 操作）。

这个操作在 Plan-R1 的图注意力层中被广泛使用：

```python
# PyTorch 中的 scatter_add
output.scatter_add_(0, index.unsqueeze(-1).expand(-1, D), src)
```

为了在 TensorRT 中支持这个操作，我们提供了一个自定义 CUDA 插件。

## 🚀 快速开始

### 1. 构建插件和推理库

```bash
cd /workspace/planner/export_tensorrt_planr1
mkdir build && cd build

cmake .. \
    -DTENSORRT_ROOT=/usr \
    -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
```

这会生成：
- `libscatter_add_plugin.so` - TensorRT 自定义插件
- `libplanr1_tensorrt.a` - C++ 推理库
- `planr1_tensorrt_demo` - Demo 可执行文件

### 2. 转换 ONNX 到 TensorRT

```bash
cd /workspace/planner/export_tensorrt_planr1

# 转换所有模型
python python/convert_to_tensorrt.py --all \
    --input-dir ../export_onnx_planr1 \
    --output-dir ./engines \
    --plugin ./build/libscatter_add_plugin.so

# 或单独转换
python python/convert_to_tensorrt.py \
    --onnx ../export_onnx_planr1/map_encoder.onnx \
    --output ./engines/map_encoder_fp16.trt \
    --fp16 \
    --plugin ./build/libscatter_add_plugin.so
```

### 3. 验证转换结果

```bash
python python/verify_tensorrt.py --all \
    --trt-dir ./engines \
    --onnx-dir ../export_onnx_planr1 \
    --plugin ./build/libscatter_add_plugin.so
```

### 4. 运行推理 Demo

```bash
./build/planr1_tensorrt_demo \
    ./engines/map_encoder_fp16.trt \
    ./engines/step_fp16.trt \
    ../export_onnx_planr1/tokens_1024.bin \
    --plugin ./build/libscatter_add_plugin.so
```

## 📊 性能对比

| 后端 | 精度 | MapEncoder | Step×16 | 总延迟 | 频率 |
|------|------|------------|---------|--------|------|
| ONNX Runtime (CPU) | FP32 | ~6ms | ~214ms | ~220ms | ~4.5 Hz |
| ONNX Runtime (GPU) | FP32 | ~2ms | ~50ms | ~52ms | ~19 Hz |
| **TensorRT (GPU)** | **FP16** | **~1ms** | **~25ms** | **~26ms** | **~38 Hz** |

*注：实际性能取决于具体 GPU 型号和驱动版本*

## 🔌 自定义插件详解

### ScatterAdd Plugin

**输入：**
- `data` [N, D]: 基础张量（通常为零）
- `index` [E]: 目标索引
- `src` [E, D]: 要累加的源值

**输出：**
- `output` [N, D]: scatter_add 结果

**操作：**
```
output = data.clone()
for e in range(E):
    output[index[e], :] += src[e, :]
```

### CUDA Kernel 实现

```cuda
__global__ void scatter_add_fp32_kernel(
    const float* src,      // [E, D]
    const int64_t* index,  // [E]
    float* output,         // [N, D]
    int E, int D, int N
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (e < E && d < D) {
        int64_t dst_idx = index[e];
        if (dst_idx >= 0 && dst_idx < N) {
            atomicAdd(&output[dst_idx * D + d], src[e * D + d]);
        }
    }
}
```

## ⚠️ 注意事项

1. **TensorRT 版本兼容性**
   - 需要 TensorRT 8.x 或更高版本
   - 插件接口可能在不同版本间有差异

2. **FP16 精度**
   - FP16 可能导致轻微的精度损失（通常 < 1%）
   - 对于大规模聚合操作，使用混合精度模式（FP16 输入，FP32 累加）

3. **动态形状**
   - `step.onnx` 包含动态边数量，需要配置 min/opt/max 形状
   - 转换脚本会自动处理

4. **内存管理**
   - TensorRT 推理需要预分配 GPU 内存
   - 推荐使用 CUDA 内存池减少分配开销

## 📦 车载部署

对于车载 GPU（如 NVIDIA Orin、Xavier）：

1. 在目标平台上重新编译插件和推理库
2. 在目标平台上重新生成 TensorRT 引擎（.trt 文件不能跨平台使用）
3. 使用 `--benchmark` 模式测试实际性能

```bash
# 在车载平台上
./planr1_tensorrt_demo \
    map_encoder_fp16.trt \
    step_fp16.trt \
    tokens_1024.bin \
    --plugin libscatter_add_plugin.so \
    --benchmark
```

## 🔗 依赖

- CUDA >= 11.0
- TensorRT >= 8.0
- Python >= 3.8
- pycuda (用于 Python 验证)

## 📖 相关文档

- [ONNX Runtime 部署方案](../export_onnx_planr1/README.md)
- [TensorRT Plugin 开发指南](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#extending)
