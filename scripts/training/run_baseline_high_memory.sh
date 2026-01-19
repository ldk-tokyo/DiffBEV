#!/bin/bash
# Baseline高显存利用率训练启动脚本
# 配置：Batch Size=8 + FP16混合精度训练
# 预期显存使用：18-20GB（利用率60-65%）

set -e

PROJECT_ROOT="/media/ldk950413/data0/DiffBEV"
cd "$PROJECT_ROOT"

# 配置
CONFIG_FILE="configs/baseline/lss_swin_nuscenes_high_memory.py"
OUTPUT_DIR="runs/baseline_high_memory"
LOG_FILE="${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 环境变量
export DIFFBEV_ALLOW_PYTORCH2=1

echo "=========================================="
echo "Baseline高显存利用率训练"
echo "=========================================="
echo "配置: Batch Size=8 + FP16"
echo "输出目录: $OUTPUT_DIR"
echo "日志文件: $LOG_FILE"
echo "=========================================="
echo ""

# 检查是否有checkpoint可以恢复
LATEST_CKPT=$(find runs/baseline -name "best_mIoU.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

if [ -n "$LATEST_CKPT" ] && [ -f "$LATEST_CKPT" ]; then
    echo "找到checkpoint: $LATEST_CKPT"
    read -p "是否从此checkpoint恢复训练？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        RESUME_ARG="--resume-from $LATEST_CKPT"
        echo "将从checkpoint恢复训练"
    else
        RESUME_ARG=""
        echo "将从头开始训练"
    fi
else
    RESUME_ARG=""
    echo "未找到checkpoint，将从头开始训练"
fi

echo ""
echo "开始训练..."
echo ""

# 启动训练
python tools/train.py "$CONFIG_FILE" \
    --work-dir "$OUTPUT_DIR" \
    --gpu-ids 0 \
    $RESUME_ARG \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "训练完成！日志已保存到: $LOG_FILE"
