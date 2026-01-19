#!/bin/bash
# GPU训练启动脚本（已配置CUDA 12.4和MMCV）

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}启动GPU训练（CUDA 12.4 + MMCV 2.1.0）${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""

# 配置CUDA 12.4环境
export CUDA_HOME=/usr/local/cuda-12.4
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 其他配置
export DISABLE_CUDA_VERSION_CHECK=1
export DIFFBEV_ALLOW_PYTORCH2=1

echo -e "${GREEN}✓ CUDA环境已配置${NC}"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  nvcc版本: $(nvcc --version | grep release | awk '{print $5}')"
echo ""

# 验证环境
python -c "
import torch
import mmcv
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
print(f'✓ MMCV: {mmcv.__version__}')
from mmcv.ops import nms
print('✓ MMCV CUDA扩展模块可用')
" || {
    echo -e "${RED}❌ 环境验证失败${NC}"
    exit 1
}

echo ""
echo -e "${GREEN}开始训练...${NC}"
echo ""

cd /media/ldk950413/data0/DiffBEV

# 运行训练
python tools/train.py configs/baseline/lss_swin_nuscenes.py \
    --work-dir runs/baseline \
    --gpu-ids 0
