#!/bin/bash
# 从源码编译MMCV - 选项3：使用系统CUDA版本编译MMCV

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}从源码编译MMCV - 选项3${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""

if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" != "diffbev" ]]; then
    echo -e "${YELLOW}⚠️  请先激活 diffbev 环境${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 当前环境: $CONDA_DEFAULT_ENV${NC}"
echo ""

# 检查PyTorch版本
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
PYTORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
echo -e "${BLUE}当前PyTorch版本: $PYTORCH_VERSION (CUDA $PYTORCH_CUDA)${NC}"
echo ""

# 方案1: 尝试安装预编译的MMCV 2.0+（如果可用）
echo -e "${BLUE}方案1: 尝试安装预编译的MMCV 2.0+ (CUDA $PYTORCH_CUDA)...${NC}"

# 卸载旧版本
pip uninstall -y mmcv-full mmcv 2>/dev/null || true

# 尝试安装MMCV 2.0.1（不带CUDA扩展，仅Python实现）
echo "尝试安装MMCV 2.0.1（Python实现，不依赖CUDA扩展）..."
pip install mmcv==2.0.1 --no-build-isolation --no-deps 2>&1 | tail -10 || {
    echo -e "${YELLOW}MMCV 2.0.1安装失败，尝试从源码编译...${NC}"
    
    # 方案2: 从源码编译MMCV
    echo ""
    echo -e "${BLUE}方案2: 从源码编译MMCV...${NC}"
    
    # 检查CUDA
    if ! command -v nvcc &> /dev/null; then
        echo -e "${YELLOW}⚠️  nvcc不可用，无法从源码编译${NC}"
        echo "  需要安装CUDA开发工具包"
        exit 1
    fi
    
    NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -c1-4)
    echo "检测到CUDA版本: $NVCC_VERSION"
    echo ""
    
    # 设置CUDA环境
    if [ -z "$CUDA_HOME" ]; then
        CUDA_HOME=$(dirname $(dirname $(which nvcc)))
        export CUDA_HOME
    fi
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    
    echo "CUDA_HOME: $CUDA_HOME"
    echo ""
    
    # 克隆MMCV（如果需要）
    MMCV_DIR="/tmp/mmcv"
    if [ ! -d "$MMCV_DIR" ]; then
        echo "克隆MMCV源码..."
        git clone https://github.com/open-mmlab/mmcv.git "$MMCV_DIR"
    else
        echo "使用已存在的MMCV源码: $MMCV_DIR"
        cd "$MMCV_DIR"
        git pull || true
    fi
    
    cd "$MMCV_DIR"
    
    # 检查out分支
    MMCV_VERSION="v1.3.15"  # 使用与项目兼容的版本
    git checkout "$MMCV_VERSION" 2>/dev/null || {
        echo -e "${YELLOW}未找到版本 $MMCV_VERSION，使用最新版本${NC}"
    }
    
    echo ""
    echo "开始编译MMCV（这可能需要10-30分钟）..."
    echo ""
    
    # 设置编译参数
    export MMCV_WITH_OPS=1
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"  # 支持的GPU架构
    
    # 编译安装
    pip install -e . -v 2>&1 | tee /tmp/mmcv_build.log
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ MMCV编译成功！${NC}"
    else
        echo ""
        echo -e "${RED}✗ MMCV编译失败，请查看日志: /tmp/mmcv_build.log${NC}"
        exit 1
    fi
}

echo ""
echo -e "${BLUE}验证MMCV安装...${NC}"
python -c "
import mmcv
print(f'✓ MMCV版本: {mmcv.__version__}')
try:
    from mmcv.ops import nms
    print('✓ MMCV扩展模块可用')
except Exception as e:
    print(f'⚠️  MMCV扩展模块不可用: {e}')
    print('   可能需要使用CPU模式或重新编译')
" 2>&1

echo ""
echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}MMCV编译/安装完成！${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""
