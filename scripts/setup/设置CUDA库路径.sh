#!/bin/bash
# 设置CUDA库路径以支持MMCV

CUDA_LIB_DIR="$CONDA_PREFIX/lib/python3.8/site-packages/nvidia/cuda_runtime/lib"
export LD_LIBRARY_PATH="$CUDA_LIB_DIR:$LD_LIBRARY_PATH"
export DIFFBEV_ALLOW_PYTORCH2=1

echo "✓ 已设置LD_LIBRARY_PATH: $CUDA_LIB_DIR"
echo "✓ 已设置DIFFBEV_ALLOW_PYTORCH2=1"

# 测试MMCV
python -c "from mmcv.ops import nms; print('✓ MMCV可用')" 2>&1
