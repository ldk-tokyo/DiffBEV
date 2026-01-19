#!/bin/bash
# 使用CUDA 13.0编译MMCV（已禁用版本检查）

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}使用CUDA 13.0编译MMCV（已禁用版本检查）${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""

if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" != "diffbev" ]]; then
    echo -e "${YELLOW}⚠️  请先激活 diffbev 环境${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 当前环境: $CONDA_DEFAULT_ENV${NC}"
echo ""

# 配置CUDA 13.0环境
CUDA_HOME="/usr/local/cuda-13.0"
if [ ! -d "$CUDA_HOME" ]; then
    echo -e "${RED}❌ CUDA 13.0未找到: $CUDA_HOME${NC}"
    exit 1
fi

export CUDA_HOME
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

echo -e "${BLUE}CUDA环境配置:${NC}"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  PATH: 包含 $CUDA_HOME/bin"
echo "  LD_LIBRARY_PATH: 包含 $CUDA_HOME/lib64"
echo ""

# 验证nvcc
if [ -f "$CUDA_HOME/bin/nvcc" ]; then
    NVCC_VERSION=$("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | grep "release" | awk '{print $5}' || echo "未知")
    echo -e "${GREEN}✓ nvcc可用，版本: $NVCC_VERSION${NC}"
else
    echo -e "${RED}❌ nvcc未找到: $CUDA_HOME/bin/nvcc${NC}"
    exit 1
fi
echo ""

# 设置禁用CUDA版本检查
export DISABLE_CUDA_VERSION_CHECK=1
export DIFFBEV_ALLOW_PYTORCH2=1

echo -e "${BLUE}卸载旧版MMCV...${NC}"
pip uninstall -y mmcv-full mmcv 2>/dev/null || true

echo ""
echo -e "${BLUE}开始编译MMCV 2.1.0（使用CUDA 13.0，但PyTorch是12.4）...${NC}"
echo -e "${YELLOW}注意：虽然CUDA版本不匹配，但已禁用版本检查${NC}"
echo "这可能需要10-30分钟..."
echo ""

# 编译MMCV
pip install mmcv==2.1.0 --no-cache-dir --verbose 2>&1 | tee /tmp/mmcv_build_cuda13.log | tail -100

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ MMCV编译成功！${NC}"
    
    # 验证安装
    python -c "
import mmcv
print(f'✓ MMCV版本: {mmcv.__version__}')
try:
    from mmcv.ops import nms
    print('✓ MMCV扩展模块可用')
except Exception as e:
    print(f'⚠️  扩展模块: {e}')
" 2>&1
    
    echo ""
    echo -e "${GREEN}================================================================================${NC}"
    echo -e "${GREEN}MMCV编译完成！${NC}"
    echo -e "${GREEN}================================================================================${NC}"
    echo ""
    echo "现在可以尝试运行GPU训练："
    echo "  export DIFFBEV_ALLOW_PYTORCH2=1"
    echo "  bash run_baseline_nuscenes.sh"
else
    echo ""
    echo -e "${RED}✗ MMCV编译失败${NC}"
    echo "详细日志: /tmp/mmcv_build_cuda13.log"
    echo ""
    echo "可能的原因："
    echo "  - CUDA 13.0和12.4的API不兼容"
    echo "  - 需要其他编译依赖"
    echo ""
    echo "建议查看日志文件中的具体错误信息"
    exit 1
fi
