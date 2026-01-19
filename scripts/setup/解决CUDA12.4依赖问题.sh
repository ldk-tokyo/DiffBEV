#!/bin/bash
# 解决CUDA 12.4安装依赖问题

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}解决CUDA 12.4安装依赖问题${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""

# 检测系统版本
OS_VERSION=$(lsb_release -rs 2>/dev/null || grep VERSION_ID /etc/os-release | cut -d'"' -f2 | cut -d'.' -f1)
echo -e "${BLUE}检测到系统版本: Ubuntu ${OS_VERSION}.04${NC}"
echo ""

# 方法1: 尝试安装缺失的依赖
echo -e "${BLUE}方法1: 尝试解决依赖问题...${NC}"
echo ""

# 尝试安装libtinfo5
if ! dpkg -l | grep -q libtinfo5; then
    echo "尝试安装libtinfo5..."
    sudo apt-get update
    sudo apt-get install -y libtinfo5 || echo "libtinfo5安装失败，尝试其他方法..."
fi

# 方法2: 使用--no-install-recommends跳过可选依赖
echo ""
echo -e "${BLUE}方法2: 仅安装核心CUDA toolkit（跳过可选工具）...${NC}"
echo ""

# 尝试安装核心包
sudo apt-get install -y --no-install-recommends \
    cuda-compiler-12-4 \
    cuda-cudart-dev-12-4 \
    cuda-libraries-dev-12-4 \
    cuda-nvcc-12-4 \
    libcurand-dev-12-4 \
    libcublas-dev-12-4 \
    libcusolver-dev-12-4 \
    libcusparse-dev-12-4 \
    libcufft-dev-12-4 \
    || echo "部分包安装失败"

# 检查是否安装了nvcc
if [ -f "/usr/local/cuda-12.4/bin/nvcc" ]; then
    echo -e "${GREEN}✓ CUDA 12.4 nvcc已安装${NC}"
    /usr/local/cuda-12.4/bin/nvcc --version
elif [ -f "/usr/bin/nvcc" ]; then
    echo -e "${GREEN}✓ nvcc已安装在/usr/bin${NC}"
    nvcc --version
    # 查找CUDA安装位置
    CUDA_BIN=$(which nvcc)
    CUDA_DIR=$(dirname $(dirname $CUDA_BIN))
    if [ -d "$CUDA_DIR" ]; then
        echo "CUDA目录: $CUDA_DIR"
        # 创建符号链接
        if [ ! -d "/usr/local/cuda-12.4" ]; then
            echo "创建符号链接: /usr/local/cuda-12.4 -> $CUDA_DIR"
            sudo ln -sfn "$CUDA_DIR" /usr/local/cuda-12.4
        fi
    fi
else
    echo -e "${YELLOW}⚠️  nvcc未找到，尝试方法3...${NC}"
fi

echo ""
echo -e "${BLUE}方法3: 使用runfile安装（推荐，不依赖包管理器）...${NC}"
echo ""
echo "如果上述方法失败，建议使用runfile方式安装："
echo ""
echo "1. 下载CUDA 12.4 runfile:"
echo "   wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run"
echo ""
echo "2. 运行安装程序:"
echo "   sudo sh cuda_12.4.0_550.54.14_linux.run"
echo ""
echo "3. 安装时选择:"
echo "   - [X] CUDA Toolkit 12.4"
echo "   - [ ] 取消选择驱动（如果已安装）"
echo "   - [ ] 取消选择nsight-systems等可选工具"
echo ""
