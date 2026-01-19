#!/bin/bash
# 修复MMCV CUDA库问题 - 选项2：配置多版本CUDA runtime

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}修复MMCV CUDA库问题 - 选项2${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""

echo -e "${BLUE}步骤1: 检查PyTorch自带的CUDA库...${NC}"
PYTORCH_LIBS=$(python -c "import torch; import os; site=os.path.dirname(torch.__file__); print(os.path.join(site, 'lib'))" 2>/dev/null || echo "")
if [ -n "$PYTORCH_LIBS" ] && [ -d "$PYTORCH_LIBS" ]; then
    echo "PyTorch库目录: $PYTORCH_LIBS"
    find "$PYTORCH_LIBS" -name "*cudart*" 2>/dev/null | head -5
fi
echo ""

echo -e "${BLUE}步骤2: 检查conda环境中的CUDA库...${NC}"
if [ -n "$CONDA_PREFIX" ]; then
    CONDALIB="$CONDA_PREFIX/lib"
    echo "Conda库目录: $CONDALIB"
    find "$CONDALIB" -name "*cudart*" 2>/dev/null | head -5
fi
echo ""

echo -e "${BLUE}步骤3: 创建符号链接（如果找到CUDA 11.0库）...${NC}"
# 尝试在conda环境中创建符号链接
if [ -n "$CONDA_PREFIX" ]; then
    CONDALIB="$CONDA_PREFIX/lib"
    
    # 查找CUDA 12.4的cudart库（PyTorch自带的）
    CUDA12_LIB=$(find "$CONDALIB" -name "libcudart.so.12*" 2>/dev/null | head -1)
    
    # 查找CUDA 11.x的cudart库
    CUDA11_LIB=$(find "$CONDALIB" -name "libcudart.so.11*" 2>/dev/null | head -1)
    
    if [ -z "$CUDA11_LIB" ] && [ -n "$CUDA12_LIB" ]; then
        echo -e "${YELLOW}未找到CUDA 11.0库，但找到CUDA 12.4库${NC}"
        echo "  这可能是问题所在 - MMCV 1.3.15需要CUDA 11.0"
        echo ""
        echo -e "${YELLOW}尝试方案: 设置LD_LIBRARY_PATH使用PyTorch的CUDA库${NC}"
        
        # 检查PyTorch安装目录
        PYTORCH_PATH=$(python -c "import torch; import os; print(os.path.dirname(os.path.dirname(torch.__file__)))" 2>/dev/null)
        if [ -n "$PYTORCH_PATH" ]; then
            TORCH_LIB="$PYTORCH_PATH/lib/python*/site-packages/torch/lib"
            TORCH_LIB_FOUND=$(find "$PYTORCH_PATH" -name "libcudart.so*" -path "*/torch/lib/*" 2>/dev/null | head -1)
            
            if [ -n "$TORCH_LIB_FOUND" ]; then
                TORCH_LIB_DIR=$(dirname "$TORCH_LIB_FOUND")
                echo "找到PyTorch CUDA库目录: $TORCH_LIB_DIR"
                echo ""
                echo -e "${GREEN}设置LD_LIBRARY_PATH...${NC}"
                export LD_LIBRARY_PATH="$TORCH_LIB_DIR:$LD_LIBRARY_PATH"
                echo "export LD_LIBRARY_PATH=\"$TORCH_LIB_DIR:\$LD_LIBRARY_PATH\"" >> ~/.bashrc
                echo -e "${GREEN}✓ 已设置LD_LIBRARY_PATH${NC}"
            fi
        fi
    elif [ -n "$CUDA11_LIB" ]; then
        echo -e "${GREEN}找到CUDA 11.0库: $CUDA11_LIB${NC}"
    fi
fi
echo ""

echo -e "${BLUE}步骤4: 测试MMCV导入...${NC}"
export DIFFBEV_ALLOW_PYTORCH2=1
python -c "
import sys
try:
    from mmcv.ops import nms
    print('✓ MMCV扩展模块导入成功')
except ImportError as e:
    print(f'✗ MMCV导入失败: {e}')
    sys.exit(1)
" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================================================================${NC}"
    echo -e "${GREEN}✓ MMCV CUDA库问题已修复！${NC}"
    echo -e "${GREEN}================================================================================${NC}"
    echo ""
    echo "现在可以运行GPU训练："
    echo "  export DIFFBEV_ALLOW_PYTORCH2=1"
    echo "  bash run_baseline_nuscenes.sh"
else
    echo ""
    echo -e "${YELLOW}选项2未能解决问题，将尝试选项3（从源码编译MMCV）${NC}"
    exit 1
fi
