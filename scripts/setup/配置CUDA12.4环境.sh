#!/bin/bash
# 配置CUDA 12.4环境以匹配PyTorch 2.4.1

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}配置CUDA 12.4环境以匹配PyTorch${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""

PYTORCH_CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
echo -e "${BLUE}PyTorch CUDA版本: $PYTORCH_CUDA_VERSION${NC}"
echo ""

# 检查是否已有CUDA 12.4
CUDA_124_PATHS=(
    "/usr/local/cuda-12.4"
    "/usr/local/cuda12.4"
    "/opt/cuda-12.4"
    "/usr/cuda-12.4"
)

CUDA_124_FOUND=""
for path in "${CUDA_124_PATHS[@]}"; do
    if [ -d "$path" ]; then
        CUDA_124_FOUND="$path"
        echo -e "${GREEN}✓ 找到CUDA 12.4: $path${NC}"
        break
    fi
done

if [ -z "$CUDA_124_FOUND" ]; then
    echo -e "${YELLOW}未找到CUDA 12.4安装${NC}"
    echo ""
    echo "选项："
    echo "  1. 使用系统现有的CUDA 13.0（通过符号链接或环境变量）"
    echo "  2. 下载并安装CUDA 12.4"
    echo "  3. 使用PyTorch自带的CUDA库（推荐）"
    echo ""
    read -p "选择选项 (1/2/3，默认3): " choice
    choice=${choice:-3}
    
    case $choice in
        1)
            # 使用CUDA 13.0但配置环境让它兼容
            echo -e "${BLUE}配置使用CUDA 13.0...${NC}"
            CUDA_HOME="/usr/local/cuda-13.0"
            ;;
        2)
            echo -e "${YELLOW}需要手动下载并安装CUDA 12.4${NC}"
            echo "下载地址: https://developer.nvidia.com/cuda-12-4-0-download-archive"
            echo "安装后重新运行此脚本"
            exit 1
            ;;
        3)
            # 使用PyTorch自带的CUDA库
            echo -e "${BLUE}配置使用PyTorch自带的CUDA库...${NC}"
            PYTORCH_LIB=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)
            echo "PyTorch CUDA库: $PYTORCH_LIB"
            
            # 创建一个临时的CUDA目录结构
            TEMP_CUDA="/tmp/cuda_env_for_mmcv"
            mkdir -p "$TEMP_CUDA/bin"
            mkdir -p "$TEMP_CUDA/lib64"
            
            # 如果PyTorch有nvcc或CUDA工具，创建符号链接
            # 否则需要安装CUDA toolkit
            
            echo "注意：可能需要安装CUDA 12.4 toolkit来获得nvcc编译器"
            CUDA_HOME="$TEMP_CUDA"
            ;;
    esac
else
    CUDA_HOME="$CUDA_124_FOUND"
fi

echo ""
echo -e "${BLUE}配置环境变量...${NC}"

# 设置CUDA环境变量
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# 将环境变量保存到bashrc（可选）
cat >> ~/.bashrc << EOF

# CUDA环境配置（用于MMCV编译）
export CUDA_HOME="$CUDA_HOME"
export CUDA_PATH="\$CUDA_HOME"
export PATH="\$CUDA_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
EOF

echo "✓ 已设置环境变量:"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  CUDA_PATH: $CUDA_HOME"
echo "  PATH: 包含 $CUDA_HOME/bin"
echo "  LD_LIBRARY_PATH: 包含 $CUDA_HOME/lib64"
echo ""

# 验证nvcc
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | cut -c1-4)
    echo -e "${GREEN}✓ nvcc可用，版本: $NVCC_VERSION${NC}"
else
    echo -e "${YELLOW}⚠️  nvcc不可用，可能需要安装CUDA toolkit${NC}"
fi

echo ""
echo -e "${BLUE}尝试编译MMCV...${NC}"

# 设置其他必要的环境变量
export DISABLE_CUDA_VERSION_CHECK=1
export DIFFBEV_ALLOW_PYTORCH2=1

# 卸载旧版MMCV
pip uninstall -y mmcv-full mmcv 2>/dev/null || true

# 尝试安装MMCV
pip install mmcv==2.1.0 --no-cache-dir 2>&1 | tee /tmp/mmcv_build_cuda124.log | tail -50

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ MMCV编译成功！${NC}"
    
    # 验证
    python -c "
import mmcv
print(f'✓ MMCV版本: {mmcv.__version__}')
try:
    from mmcv.ops import nms
    print('✓ MMCV扩展模块可用')
except Exception as e:
    print(f'⚠️  扩展模块: {e}')
" 2>&1
else
    echo ""
    echo -e "${RED}✗ MMCV编译失败${NC}"
    echo "详细日志: /tmp/mmcv_build_cuda124.log"
    exit 1
fi

echo ""
echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}CUDA环境配置完成！${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""
echo "环境变量已添加到 ~/.bashrc"
echo "下次登录时会自动加载"
echo "或运行: source ~/.bashrc"
