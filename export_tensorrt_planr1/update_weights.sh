#!/bin/bash
# ============================================================================
# Plan-R1 权重更新脚本
# 
# 用法:
#   ./update_weights.sh /path/to/new_weights.ckpt
#   ./update_weights.sh /path/to/new_weights.ckpt --verify
#   ./update_weights.sh /path/to/new_weights.ckpt --onnx-only
#   ./update_weights.sh /path/to/new_weights.ckpt --tensorrt-only
#   ./update_weights.sh /path/to/new_weights.ckpt --fp16
#
# 选项:
#   --verify        转换后运行验证 (使用真实数据)
#   --onnx-only     只导出 ONNX，不转换 TensorRT
#   --tensorrt-only 只转换 TensorRT（使用现有 ONNX）
#   --fp16          使用 FP16 精度 (默认 FP32)
#   --num-samples N 验证时使用的样本数 (默认 10)
#   --help          显示帮助
# ============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 目录定义
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_DIR="$SCRIPT_DIR/export_onnx_planr1"
TRT_DIR="$SCRIPT_DIR/export_tensorrt_planr1"
PLUGIN_PATH="$TRT_DIR/build/libscatter_add_plugin.so"

# 打印带颜色的消息
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# 显示帮助
show_help() {
    echo "Plan-R1 权重更新脚本"
    echo ""
    echo "用法: $0 <checkpoint_path> [options]"
    echo ""
    echo "参数:"
    echo "  checkpoint_path    PyTorch 权重文件路径 (.ckpt)"
    echo ""
    echo "选项:"
    echo "  --verify           转换后运行真实数据验证"
    echo "  --onnx-only        只导出 ONNX，不转换 TensorRT"
    echo "  --tensorrt-only    只转换 TensorRT（使用现有 ONNX）"
    echo "  --fp16             使用 FP16 精度 (默认 FP32，更稳定)"
    echo "  --num-samples N    验证时使用的样本数 (默认 10)"
    echo "  --help             显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 ./Plan-R1/ckpts/fine-tuning.ckpt"
    echo "  $0 ./new_model.ckpt --verify"
    echo "  $0 ./new_model.ckpt --verify --num-samples 50"
    echo "  $0 --tensorrt-only"
    echo "  $0 ./new_model.ckpt --fp16  # 使用 FP16 (可能有精度问题)"
    echo ""
    echo "输出文件:"
    echo "  ONNX:"
    echo "    - export_onnx_planr1/map_encoder.onnx"
    echo "    - export_onnx_planr1/step.onnx"
    echo ""
    echo "  TensorRT:"
    echo "    - export_tensorrt_planr1/engines/map_encoder_fp16.trt (或 fp32)"
    echo "    - export_tensorrt_planr1/engines/step_fp32.trt (或 fp16)"
    echo "    - export_tensorrt_planr1/engines/step_modified.onnx"
    echo ""
    echo "部署时需要复制到目标机器:"
    echo "  1. engines/ 目录下的 .trt 文件"
    echo "  2. engines/step_modified.onnx (用于在目标机器重新构建 TRT)"
    echo "  3. build/libscatter_add_plugin.so (需要在目标机器重新编译)"
    echo "  4. plugins/ 目录 (ScatterAdd 插件源码，用于目标机器编译)"
}

# 解析参数
CKPT_PATH=""
DO_VERIFY=false
ONNX_ONLY=false
TRT_ONLY=false
USE_FP16=false
NUM_SAMPLES=10

while [[ $# -gt 0 ]]; do
    case $1 in
        --verify)
            DO_VERIFY=true
            shift
            ;;
        --onnx-only)
            ONNX_ONLY=true
            shift
            ;;
        --tensorrt-only)
            TRT_ONLY=true
            shift
            ;;
        --fp16)
            USE_FP16=true
            shift
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            if [[ -z "$CKPT_PATH" ]]; then
                CKPT_PATH="$1"
            else
                error "未知参数: $1"
            fi
            shift
            ;;
    esac
done

# 检查参数
if [[ "$TRT_ONLY" == false && -z "$CKPT_PATH" ]]; then
    error "请提供权重文件路径。使用 --help 查看帮助。"
fi

if [[ -n "$CKPT_PATH" && ! -f "$CKPT_PATH" ]]; then
    error "权重文件不存在: $CKPT_PATH"
fi

# 设置精度
if [[ "$USE_FP16" == true ]]; then
    PRECISION="fp16"
    PRECISION_FLAG="--fp16"
else
    PRECISION="fp32"
    PRECISION_FLAG="--fp32"
fi

# 检查插件
if [[ "$ONNX_ONLY" == false && ! -f "$PLUGIN_PATH" ]]; then
    warn "TensorRT 插件不存在，将尝试编译..."
    cd "$TRT_DIR"
    if [[ ! -d "build" ]]; then
        mkdir build
    fi
    cd build
    cmake .. -DTENSORRT_ROOT=/usr -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc)
    cd "$SCRIPT_DIR"
    
    if [[ ! -f "$PLUGIN_PATH" ]]; then
        error "插件编译失败"
    fi
    success "插件编译完成"
fi

echo ""
echo "============================================================"
echo "         Plan-R1 权重更新"
echo "============================================================"
if [[ -n "$CKPT_PATH" ]]; then
    echo "权重文件: $CKPT_PATH"
fi
echo "ONNX 目录: $ONNX_DIR"
echo "TensorRT 目录: $TRT_DIR"
echo "精度: $PRECISION"
echo "============================================================"
echo ""

# ============================================================================
# 步骤 1: 导出 ONNX
# ============================================================================
if [[ "$TRT_ONLY" == false ]]; then
    info "步骤 1/3: 导出 ONNX 模型..."
    
    cd "$SCRIPT_DIR"
    python -m export_onnx_planr1.export_split_onnx \
        --ckpt "$CKPT_PATH" \
        --output-dir "$ONNX_DIR/"
    
    if [[ -f "$ONNX_DIR/map_encoder.onnx" && -f "$ONNX_DIR/step.onnx" ]]; then
        success "ONNX 导出完成"
        ls -lh "$ONNX_DIR"/*.onnx
    else
        error "ONNX 导出失败"
    fi
    echo ""
fi

# ============================================================================
# 步骤 2: 转换 TensorRT
# ============================================================================
if [[ "$ONNX_ONLY" == false ]]; then
    info "步骤 2/3: 转换 TensorRT $PRECISION 引擎..."
    
    cd "$TRT_DIR"
    
    # 创建 engines 目录
    mkdir -p engines
    
    # 转换 MapEncoder (FP16 对 MapEncoder 通常没问题)
    info "转换 MapEncoder..."
    python python/convert_to_tensorrt.py \
        --onnx "$ONNX_DIR/map_encoder.onnx" \
        --output "./engines/map_encoder_fp16.trt" \
        --fp16 \
        --plugin "$PLUGIN_PATH"
    
    # 转换 StepModel (使用指定精度)
    info "转换 StepModel ($PRECISION)..."
    python python/convert_to_tensorrt.py \
        --onnx "$ONNX_DIR/step.onnx" \
        --output "./engines/step_${PRECISION}.trt" \
        $PRECISION_FLAG \
        --dynamic \
        --plugin "$PLUGIN_PATH"
    
    # 检查输出
    if [[ -f "engines/map_encoder_fp16.trt" && -f "engines/step_${PRECISION}.trt" ]]; then
        success "TensorRT 转换完成"
        ls -lh engines/*.trt
        echo ""
        info "中间文件 (用于部署):"
        ls -lh engines/*.onnx 2>/dev/null || true
    else
        error "TensorRT 转换失败"
    fi
    echo ""
fi

# ============================================================================
# 步骤 3: 验证（可选）
# ============================================================================
if [[ "$DO_VERIFY" == true && "$ONNX_ONLY" == false ]]; then
    info "步骤 3/3: 运行真实数据验证 ($NUM_SAMPLES 个样本)..."
    
    cd "$TRT_DIR"
    
    # 生成参考数据
    info "生成 PyTorch 参考输出..."
    python python/verify_step_real_data.py \
        --num-samples "$NUM_SAMPLES" \
        --output /tmp/step_verify_data.npz
    
    # 修改验证脚本使用的数据路径
    sed -i "s|/tmp/step_real_data_50.npz|/tmp/step_verify_data.npz|g" python/verify_step_trt_real.py 2>/dev/null || true
    sed -i "s|/tmp/step_real_data.npz|/tmp/step_verify_data.npz|g" python/verify_step_trt_real.py 2>/dev/null || true
    
    # 运行 TensorRT 验证
    info "运行 TensorRT 验证..."
    python python/verify_step_trt_real.py
    
    if [[ $? -eq 0 ]]; then
        success "验证完成"
    else
        warn "验证失败，请检查模型"
    fi
    echo ""
fi

# ============================================================================
# 完成
# ============================================================================
echo ""
echo "============================================================"
success "更新完成！"
echo "============================================================"
echo ""
echo "生成的文件:"

if [[ "$TRT_ONLY" == false ]]; then
    echo "  ONNX (标准格式，可用于 ONNX Runtime):"
    echo "    - $ONNX_DIR/map_encoder.onnx"
    echo "    - $ONNX_DIR/step.onnx"
fi

if [[ "$ONNX_ONLY" == false ]]; then
    echo ""
    echo "  TensorRT 引擎 (当前机器专用):"
    echo "    - $TRT_DIR/engines/map_encoder_fp16.trt"
    echo "    - $TRT_DIR/engines/step_${PRECISION}.trt"
    echo ""
    echo "  TensorRT 中间文件 (用于其他机器重新构建):"
    echo "    - $TRT_DIR/engines/step_modified.onnx"
fi

echo ""
echo "============================================================"
echo "部署到其他机器时需要:"
echo "============================================================"
echo ""
echo "  方式 1: 复制 ONNX + 重新构建 TensorRT (推荐)"
echo "    复制: step_modified.onnx, map_encoder.onnx"
echo "    复制: plugins/ 目录 (ScatterAdd 插件源码)"
echo "    在目标机器: 编译插件 + 运行 convert_to_tensorrt.py"
echo ""
echo "  方式 2: 直接复制 TensorRT 引擎 (需要相同环境)"
echo "    要求: GPU 架构相同, TensorRT 版本相同"
echo "    复制: engines/*.trt + build/libscatter_add_plugin.so"
echo ""
