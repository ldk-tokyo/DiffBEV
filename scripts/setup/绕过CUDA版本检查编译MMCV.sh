#!/bin/bash
# 绕过CUDA版本检查编译MMCV - 使用PyTorch的CUDA版本

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}绕过CUDA版本检查编译MMCV${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""

if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" != "diffbev" ]]; then
    echo -e "${YELLOW}⚠️  请先激活 diffbev 环境${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 当前环境: $CONDA_DEFAULT_ENV${NC}"
echo ""

# 获取PyTorch的CUDA库路径
PYTORCH_CUDA_LIB=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null)
echo -e "${BLUE}PyTorch CUDA库路径: $PYTORCH_CUDA_LIB${NC}"

# 卸载旧版MMCV
echo ""
echo -e "${BLUE}卸载旧版MMCV...${NC}"
pip uninstall -y mmcv-full mmcv 2>/dev/null || true

# 方法1: 设置环境变量，强制使用PyTorch的CUDA
echo ""
echo -e "${BLUE}方法1: 设置环境变量使用PyTorch的CUDA...${NC}"

# 隐藏系统的CUDA，让编译系统使用PyTorch自带的CUDA
export CUDA_HOME=""  # 清空CUDA_HOME，让PyTorch使用自己的CUDA
export CUDA_PATH=""   # 清空CUDA_PATH

# 设置PATH，确保不使用系统的nvcc
# 将PyTorch的库路径加入LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$PYTORCH_CUDA_LIB:$LD_LIBRARY_PATH"

# 禁用CUDA版本检查（通过环境变量）
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
export FORCE_CUDA="1"

echo "已设置环境变量:"
echo "  CUDA_HOME: (空，使用PyTorch的CUDA)"
echo "  LD_LIBRARY_PATH: 包含 $PYTORCH_CUDA_LIB"
echo ""

# 方法2: 临时修改PyTorch的CUDA版本检查（更激进的方法）
echo -e "${BLUE}方法2: 尝试patch PyTorch的CUDA版本检查...${NC}"

# 找到PyTorch的cpp_extension.py文件
PYTORCH_EXT_FILE=$(python -c "import torch.utils.cpp_extension; import os; print(os.path.abspath(torch.utils.cpp_extension.__file__))" 2>/dev/null)
echo "PyTorch扩展文件: $PYTORCH_EXT_FILE"

# 备份原文件
if [ -f "$PYTORCH_EXT_FILE" ]; then
    cp "$PYTORCH_EXT_FILE" "${PYTORCH_EXT_FILE}.backup"
    echo "✓ 已备份原文件"
    
    # 尝试修改CUDA版本检查（如果可能）
    # 注意：这需要谨慎操作
fi

# 尝试安装MMCV
echo ""
echo -e "${BLUE}开始安装MMCV 2.1.0...${NC}"
echo "这可能需要10-30分钟..."

pip install mmcv==2.1.0 --no-cache-dir --verbose 2>&1 | tee /tmp/mmcv_build.log | tail -50

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ MMCV编译成功！${NC}"
    
    # 恢复备份文件（如果需要）
    if [ -f "${PYTORCH_EXT_FILE}.backup" ]; then
        mv "${PYTORCH_EXT_FILE}.backup" "$PYTORCH_EXT_FILE"
        echo "✓ 已恢复PyTorch原文件"
    fi
    
    # 验证安装
    echo ""
    echo -e "${BLUE}验证MMCV安装...${NC}"
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
    echo "日志保存在: /tmp/mmcv_build.log"
    
    # 恢复备份文件
    if [ -f "${PYTORCH_EXT_FILE}.backup" ]; then
        mv "${PYTORCH_EXT_FILE}.backup" "$PYTORCH_EXT_FILE"
        echo "✓ 已恢复PyTorch原文件"
    fi
    
    exit 1
fi

echo ""
echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}MMCV编译完成！${NC}"
echo -e "${GREEN}================================================================================${NC}"
