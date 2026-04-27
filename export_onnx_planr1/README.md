# Plan-R1 ONNX Export

将 Plan-R1 模型导出为 ONNX 格式，采用 **"单步 ONNX + 外部循环"** 策略。

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    外部 (Python/C++)                         │
├─────────────────────────────────────────────────────────────┤
│  1. 数据预处理                                               │
│     - 筛选距离内的 agents 和 polygons                        │
│     - 计算边索引和边属性 (固定，16步共用)                      │
│                                                             │
│  2. 运行 map_encoder.onnx (一次)                            │
│     输入: polyline/polygon 特征                              │
│     输出: polygon_embs [145, 128]                           │
│                                                             │
│  3. 自回归循环 (16次)                                        │
│     for step in range(16):                                  │
│       ├─ 运行 step.onnx                                     │
│       │   输入: agent_token, polygon_embs, edges            │
│       │   输出: logits [21, 1024]                           │
│       ├─ 采样下一个 token (argmax)                          │
│       ├─ 解码 token -> 轨迹 (position, heading)             │
│       └─ 更新状态                                           │
└─────────────────────────────────────────────────────────────┘
```

## 目录结构

```
export_onnx_planr1/
├── __init__.py                 # 模块入口
├── export_split_onnx.py        # 导出脚本
├── verify_split_onnx.py        # 验证脚本
├── inference_loop.py           # 推理循环参考实现
├── map_encoder_exportable.py   # MapEncoder 可导出版本
├── decoder_head_exportable.py  # DecoderHead 可导出版本
├── step_exportable.py          # 单步推理模型
├── layers_exportable.py        # 可导出的注意力层（无 PyG 依赖）
├── map_encoder.onnx            # 导出的地图编码器 (2.1 MB)
├── step.onnx                   # 导出的单步模型 (17.1 MB)
└── README.md                   # 本文件
```

## 使用方法

### 1. 导出 ONNX 模型

```bash
cd /workspace/planner

python -m export_onnx_planr1.export_split_onnx \
    --ckpt /workspace/planner/Plan-R1/ckpts/fine-tuning.ckpt \
    --tokens /workspace/planner/Plan-R1/tokens/tokens_1024.pt \
    --output-dir ./export_onnx_planr1/
```

### 2. 验证导出的模型

```bash
python -m export_onnx_planr1.verify_split_onnx \
    --dir ./export_onnx_planr1/
```

### 3. 运行推理

```bash
# 使用 Python 推理循环
python export_onnx_planr1/inference_loop.py
```

## 导出的 ONNX 模型

### map_encoder.onnx

计算地图嵌入向量，每个场景只需运行一次。

#### 输入

| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `polyline_position` | (1200, 2) | float32 | polyline 点位置 |
| `polyline_heading` | (1200,) | float32 | polyline 点朝向 |
| `polyline_length` | (1200,) | float32 | 到下一点的距离 |
| `polygon_position` | (145, 2) | float32 | polygon 中心位置 |
| `polygon_heading` | (145,) | float32 | polygon 朝向 |
| `polygon_heading_valid` | (145,) | bool | 朝向是否有效 |
| `polygon_type` | (145,) | int64 | polygon 类型 |
| `polygon_traffic_light` | (145,) | int64 | 信号灯状态 |
| `polygon_speed_limit` | (145,) | float32 | 速度限制 |
| `polygon_speed_limit_valid` | (145,) | bool | 速度限制是否有效 |
| `polygon_on_route` | (145,) | bool | 是否在规划路线上 |
| `left_edge_index` | (2, 40) | int64 | 左邻居边 |
| `right_edge_index` | (2, 40) | int64 | 右邻居边 |
| `incoming_edge_index` | (2, 80) | int64 | 入边 |
| `outgoing_edge_index` | (2, 80) | int64 | 出边 |
| `polyline_to_polygon_edge_index` | (2, 1200) | int64 | polyline-polygon 边 |

#### 输出

| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `polygon_embs` | (145, 128) | float32 | polygon 嵌入向量 |

### step.onnx

单步推理模型，自回归循环中运行 16 次。

#### 输入

| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `agent_token` | (21,) | int64 | 当前运动令牌 |
| `agent_type` | (21,) | int64 | agent 类型 |
| `agent_box` | (21, 4) | float32 | 边界框尺寸 |
| `agent_identity` | (21,) | int64 | 身份标识 |
| `agent_position_hist` | (21, T, 2) | float32 | 历史位置 |
| `agent_heading_hist` | (21, T) | float32 | 历史朝向 |
| `agent_valid_mask_hist` | (21, T) | bool | 历史有效掩码 |
| `agent_embs` | (21, 128) | float32 | agent 静态嵌入 |
| `polygon_embs` | (145, 128) | float32 | polygon 嵌入 (来自 map_encoder) |
| `polygon_position` | (145, 2) | float32 | polygon 位置 |
| `polygon_heading` | (145,) | float32 | polygon 朝向 |
| `polygon_heading_valid` | (145,) | bool | 朝向是否有效 |
| `k2k_t_edge_index` | (2, E1) | int64 | 时序自注意力边 |
| `k2k_t_edge_attr` | (E1, 6) | float32 | 时序边属性 |
| `g2k_edge_index` | (2, E2) | int64 | 地图-agent 交叉注意力边 |
| `g2k_edge_attr` | (E2, 6) | float32 | 交叉注意力边属性 |
| `k2k_a_edge_index` | (2, E3) | int64 | agent 间自注意力边 |
| `k2k_a_edge_attr` | (E3, 5) | float32 | agent 间边属性 |
| `num_k2k_t_edges` | (1,) | int64 | 实际时序边数量 |
| `num_g2k_edges` | (1,) | int64 | 实际交叉注意力边数量 |
| `num_k2k_a_edges` | (1,) | int64 | 实际 agent 间边数量 |

#### 输出

| 名称 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `logits` | (21, 1024) | float32 | 下一个运动令牌的 logits |

## 推理流程

```python
import onnxruntime as ort
import numpy as np

# 1. 加载模型
map_encoder = ort.InferenceSession("map_encoder.onnx")
step_model = ort.InferenceSession("step.onnx")

# 2. 预处理数据（一次性完成）
# - 筛选距离内的 agents 和 polygons
# - 计算边索引和边属性

# 3. 计算地图嵌入（运行一次）
polygon_embs = map_encoder.run(None, map_inputs)[0]

# 4. 自回归循环
for step in range(16):
    # 准备输入
    step_inputs = {
        "agent_token": current_token,
        "polygon_embs": polygon_embs,
        # ... 其他输入
    }
    
    # 运行单步推理
    logits = step_model.run(None, step_inputs)[0]
    
    # 采样下一个 token
    next_token = np.argmax(logits, axis=-1)
    
    # 解码 token 到轨迹
    delta_pos, delta_heading = decode_token(next_token, token_dict)
    
    # 更新状态
    position = position + delta_pos
    heading = heading + delta_heading
    current_token = next_token
```

## 技术说明

### 为什么采用单步 ONNX 策略？

Plan-R1 的完整推理包含以下 ONNX 不兼容的操作：

1. **自回归循环**: Python `for` 循环无法直接转换为 ONNX `Loop` 操作符
2. **PyTorch Geometric**: 动态图操作（`MessagePassing`, `radius_graph`）不受支持
3. **条件分支**: 数据依赖的 `if` 语句在追踪时被固化

### 解决方案

将模型拆分为两个静态 ONNX 图：

1. **map_encoder.onnx**: 纯前馈网络，无动态操作
2. **step.onnx**: 单步 backbone + decoder，边索引作为输入

自回归循环和边计算在 ONNX 外部完成（Python/C++）。

### 边计算

边索引和边属性在数据预处理阶段计算，16 个推理步骤共用同一套边：

- **k2k_t**: 时序自注意力边（同一 agent 不同时间步之间）
- **g2k**: 地图-agent 交叉注意力边（polygon 到 agent）
- **k2k_a**: agent 间自注意力边（不同 agent 之间）

## 默认维度

| 维度 | 值 | 说明 |
|------|-----|------|
| `max_agents` | 21 | 20 邻居 + 1 自车 |
| `max_polygons` | 145 | 最多 145 个 polygon |
| `max_polylines` | 1200 | 最多 1200 个 polyline 点 |
| `hidden_dim` | 128 | 隐藏层维度 |
| `num_tokens` | 1024 | 运动令牌数量 |
| `num_future_steps` | 16 | 预测步数 (16 × 0.5s = 8s) |
| `interval` | 5 | 每个令牌对应 5 帧 |

## 注意事项

1. **边数量可变**: step.onnx 支持动态边数量，通过 `num_*_edges` 输入指定
2. **内存**: 固定维度会导致一定的内存浪费，但保证了 ONNX 兼容性
3. **精度**: 导出后可能有微小的数值差异（< 1e-5）
4. **版本要求**: PyTorch >= 2.0, ONNX opset >= 17

## 转换为 TensorRT

```bash
# 转换 map_encoder
trtexec --onnx=map_encoder.onnx --saveEngine=map_encoder.trt --fp16

# 转换 step 模型（需指定动态形状）
trtexec --onnx=step.onnx --saveEngine=step.trt --fp16 \
    --minShapes=k2k_t_edge_index:2x1,g2k_edge_index:2x1,k2k_a_edge_index:2x1 \
    --optShapes=k2k_t_edge_index:2x100,g2k_edge_index:2x500,k2k_a_edge_index:2x200 \
    --maxShapes=k2k_t_edge_index:2x500,g2k_edge_index:2x2000,k2k_a_edge_index:2x500
```
