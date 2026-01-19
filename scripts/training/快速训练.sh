#!/bin/bash
# DiffBEV 快速训练命令 - 直接使用环境中的Python

set -e

# 颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}DiffBEV 快速训练启动${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""

# 环境配置
ENV_NAME="diffbev_py310"
ENV_PATH="$HOME/.local/share/mamba/envs/$ENV_NAME"

# 检查环境是否存在
if [ ! -d "$ENV_PATH" ]; then
    echo -e "${RED}❌ 错误: 环境不存在: $ENV_PATH${NC}"
    echo "   请检查环境路径或手动激活环境后运行:"
    echo "   micromamba activate $ENV_NAME"
    exit 1
fi

# 直接使用环境中的Python
PYTHON_BIN="$ENV_PATH/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    echo -e "${RED}❌ 错误: Python不存在: $PYTHON_BIN${NC}"
    exit 1
fi

echo -e "${GREEN}使用Python: $PYTHON_BIN${NC}"
echo ""

# 设置环境变量
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export DIFFBEV_ALLOW_PYTORCH2=1
export DISABLE_CUDA_VERSION_CHECK=1

# 切换到项目目录
cd /media/ldk950413/data0/DiffBEV

# 配置文件
CONFIG_FILE="configs/baseline/lss_swin_nuscenes.py"
WORK_DIR="runs/baseline"

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ 错误: 配置文件不存在: $CONFIG_FILE${NC}"
    exit 1
fi

# 创建输出目录
mkdir -p "$WORK_DIR"

echo -e "${GREEN}训练配置:${NC}"
echo "  配置文件: $CONFIG_FILE"
echo "  工作目录: $WORK_DIR"
echo ""

# 显示GPU信息
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU信息:${NC}"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | head -1
    echo ""
fi

# 启动训练
echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}开始训练...${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""

"$PYTHON_BIN" tools/train.py "$CONFIG_FILE" \
    --work-dir "$WORK_DIR" \
    --gpu-ids 0
