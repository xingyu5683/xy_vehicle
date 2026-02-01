#!/bin/bash

###############################################################################
# Carla 离线数据收集脚本
# 功能: 使用 autopilot 在 Carla 环境中收集离线训练数据
###############################################################################

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           🚗 Carla 离线数据收集                                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ============================================================================
# 1. 配置参数
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SCRIPT_DIR 是 offline 目录，PROJECT_ROOT 是其父目录
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFFLINE_DIR="${SCRIPT_DIR}"
CONFIG_FILE="${PROJECT_ROOT}/configs/base.yaml"

# 数据收集参数
NUM_EPISODES=1000             # 收集的 episode 数量
MAX_STEPS=250               # 每个 episode 最大步数
SEED=42                     # 随机种子

# 数据存储目录和文件名
DATASET_DIR="${OFFLINE_DIR}/dataset"
OUTPUT_FILENAME="carla_offline_dataset.hdf5"  # 输出文件名（含扩展名）
OUTPUT_FILE="${DATASET_DIR}/${OUTPUT_FILENAME}"  # 完整输出路径

# Python 环境
CONDA_ENV="carla_py37"
PYTHON="/root/miniconda3/envs/${CONDA_ENV}/bin/python"

# ============================================================================
# 2. 检查环境
# ============================================================================
echo -e "${YELLOW}>> 检查环境...${NC}"

# 检查 Conda 环境
if [ ! -f "${PYTHON}" ]; then
    echo -e "${RED}错误: Python 环境不存在: ${PYTHON}${NC}"
    echo "请先创建 carla_py37 环境"
    exit 1
fi

# 检查配置文件
if [ ! -f "${CONFIG_FILE}" ]; then
    echo -e "${RED}错误: 配置文件不存在: ${CONFIG_FILE}${NC}"
    exit 1
fi

# 检查数据收集脚本
COLLECT_SCRIPT="${OFFLINE_DIR}/collect_offline_data_carla.py"
if [ ! -f "${COLLECT_SCRIPT}" ]; then
    echo -e "${RED}错误: 数据收集脚本不存在: ${COLLECT_SCRIPT}${NC}"
    exit 1
fi

# 创建数据存储目录（如果不存在）
if [ ! -d "${DATASET_DIR}" ]; then
    echo -e "${YELLOW}>> 创建数据目录: ${DATASET_DIR}${NC}"
    mkdir -p "${DATASET_DIR}"
fi

echo -e "${GREEN}✓ 环境检查通过${NC}"

# ============================================================================
# 3. 显示配置信息
# ============================================================================
echo ""
echo -e "${YELLOW}>> 配置信息:${NC}"
echo "   项目根目录: ${PROJECT_ROOT}"
echo "   配置文件:   ${CONFIG_FILE}"
echo "   Python:     ${PYTHON}"
echo "   环境名称:   ${CONDA_ENV}"
echo ""
echo -e "${YELLOW}>> 数据收集参数:${NC}"
echo "   Episode 数量:    ${NUM_EPISODES}"
echo "   每 Episode 步数: ${MAX_STEPS}"
echo "   数据目录:        ${DATASET_DIR}"
echo "   输出文件:        ${OUTPUT_FILENAME}"
echo "   完整路径:        ${OUTPUT_FILE}"
echo "   随机种子:        ${SEED}"
echo ""

# ============================================================================
# 4. 激活环境并运行数据收集
# ============================================================================
echo -e "${GREEN}>> 激活 ${CONDA_ENV} 环境...${NC}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# 验证环境
echo -e "${YELLOW}>> Python 版本:${NC}"
${PYTHON} --version

echo ""
echo -e "${GREEN}>> 开始收集数据...${NC}"
echo "   请确保 Carla 服务器已启动 (端口 2000)"
echo "   时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 切换到 offline 目录（脚本需要在该目录下运行）
cd "${OFFLINE_DIR}"

# 运行数据收集脚本
${PYTHON} collect_offline_data_carla.py \
    --ROOT_DIR "${PROJECT_ROOT}" \
    --config "configs/base.yaml" \
    --num_episodes ${NUM_EPISODES} \
    --max_steps ${MAX_STEPS} \
    --output "${OUTPUT_FILE}" \
    --seed ${SEED}

# ============================================================================
# 5. 完成
# ============================================================================
echo ""
echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           ✅ 数据收集完成                                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# 显示生成的文件
echo -e "${YELLOW}>> 生成的文件:${NC}"
if [ -f "${OUTPUT_FILE}" ]; then
    FILE_SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
    echo "   数据文件: ${OUTPUT_FILE} (${FILE_SIZE})"
fi
# 统计文件名（移除 .hdf5 后缀，添加 _stats.yaml）
STATS_FILE="${OUTPUT_FILE%.hdf5}_stats.yaml"
if [ -f "${STATS_FILE}" ]; then
    echo "   统计文件: ${STATS_FILE}"
fi

echo ""
echo -e "${GREEN}✓ 完成! 时间: $(date '+%Y-%m-%d %H:%M:%S')${NC}"

