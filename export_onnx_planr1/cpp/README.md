# Plan-R1 C++ ONNX Runtime 推理

使用 ONNX Runtime 的 C++ 推理实现，用于车载部署。

## 📁 目录结构

```
cpp/
├── CMakeLists.txt
├── README.md
├── include/
│   └── planr1_inference.h      # 头文件
└── src/
    ├── planr1_inference.cpp    # 推理实现
    └── main.cpp                # 示例程序
```

## 🔧 构建

```bash
mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT=/opt/onnxruntime -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## 🚀 运行

```bash
./planr1_demo <map_encoder.onnx> <step.onnx> <tokens.bin> [--cpu]

# 示例
./planr1_demo ../map_encoder.onnx ../step.onnx ../tokens_1024.bin

# 使用 CPU
./planr1_demo ../map_encoder.onnx ../step.onnx ../tokens_1024.bin --cpu
```

## 📦 车载部署文件

```
deploy/
├── map_encoder.onnx    # ~2.1 MB
├── step.onnx           # ~17.1 MB
├── tokens_1024.bin     # ~36 KB
└── planr1_demo         # 可执行文件
```

## 📝 API 使用

```cpp
#include "planr1_inference.h"

// 加载 token 字典
auto token_dict = planr1::loadTokenDictionary("tokens_1024.bin");

// 初始化推理引擎
planr1::PlanR1Inference inference(
    "map_encoder.onnx",
    "step.onnx",
    token_dict,
    true  // use_gpu
);

// 1. 运行 MapEncoder (每场景一次)
auto polygon_embs = inference.runMapEncoder(map_input);

// 2. 准备 StepModel 输入
step_input.polygon_embs = polygon_embs;

// 3. 运行自回归推理 (16步)
auto output = inference.runInference(step_input);

// 4. 获取结果
// output.positions: [num_agents * 16 * 2]
// output.headings:  [num_agents * 16]
```

## ⚙️ 依赖

- ONNX Runtime >= 1.15
- CMake >= 3.14
- C++17 编译器
- CUDA (可选，用于 GPU 加速)
