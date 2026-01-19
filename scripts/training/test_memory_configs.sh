#!/bin/bash
# 测试不同配置的显存使用情况

set -e

PROJECT_ROOT="/media/ldk950413/data0/DiffBEV"
cd "$PROJECT_ROOT"

export DIFFBEV_ALLOW_PYTORCH2=1

echo "=========================================="
echo "显存利用率测试工具"
echo "=========================================="
echo ""
echo "将测试以下配置："
echo "1. Batch Size=4 (当前配置)"
echo "2. Batch Size=8"
echo "3. Batch Size=8 + FP16"
echo ""
read -p "是否继续？(y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 1
fi

# 测试1: Batch Size=4
echo ""
echo "=========================================="
echo "测试1: Batch Size=4 (当前配置)"
echo "=========================================="
python tools/train.py configs/baseline/lss_swin_nuscenes.py \
    --work-dir runs/test_bs4 \
    --gpu-ids 0 \
    --options runner.max_iters=100 data.samples_per_gpu=4 2>&1 | \
    grep -E "(iter|loss|memory|显存)" | tail -5 || true

echo ""
echo "按Enter继续下一个测试..."
read

# 测试2: Batch Size=8
echo ""
echo "=========================================="
echo "测试2: Batch Size=8"
echo "=========================================="
python tools/train.py configs/baseline/lss_swin_nuscenes.py \
    --work-dir runs/test_bs8 \
    --gpu-ids 0 \
    --options runner.max_iters=100 data.samples_per_gpu=8 optimizer.lr=4e-4 2>&1 | \
    grep -E "(iter|loss|memory|显存)" | tail -5 || true

echo ""
echo "按Enter继续下一个测试..."
read

# 测试3: Batch Size=8 + FP16
echo ""
echo "=========================================="
echo "测试3: Batch Size=8 + FP16"
echo "=========================================="
python tools/train.py configs/baseline/lss_swin_nuscenes_high_memory.py \
    --work-dir runs/test_bs8_fp16 \
    --gpu-ids 0 \
    --options runner.max_iters=100 2>&1 | \
    grep -E "(iter|loss|memory|显存|FP16|Training|完成)" | tail -10 || true

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "请检查各配置的显存使用情况："
echo "nvidia-smi"
echo ""
