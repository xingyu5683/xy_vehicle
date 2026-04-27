#!/bin/bash

# 移动 nuplan_train_test.json 中的文件到 /mnt/datadownload

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认参数
JSON_FILE="nuplan_train_test.json"
SOURCE_DIRS="/mnt/data/dataset/nuplan-v1.1/splits"
TARGET_DIR="/mnt/datadownload"

# 解析命令行参数
DRY_RUN=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry_run"
            shift
            ;;
        --json-file)
            JSON_FILE="$2"
            shift 2
            ;;
        --target-dir)
            TARGET_DIR="$2"
            shift 2
            ;;
        --source-dirs)
            SOURCE_DIRS="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--dry-run] [--json-file FILE] [--target-dir DIR] [--source-dirs DIRS]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "移动 NuPlan 数据库文件"
echo "============================================================"
echo "JSON 文件: $JSON_FILE"
echo "源目录: $SOURCE_DIRS"
echo "目标目录: $TARGET_DIR"
if [ -n "$DRY_RUN" ]; then
    echo "模式: 模拟运行（不会实际移动文件）"
else
    echo "模式: 实际移动"
fi
echo "============================================================"
echo ""

# 运行 Python 脚本
python move_nuplan_files.py \
    --json_file "$JSON_FILE" \
    --source_dirs $SOURCE_DIRS \
    --target_dir "$TARGET_DIR" \
    $DRY_RUN

