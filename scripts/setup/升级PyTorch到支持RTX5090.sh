#!/bin/bash
# 升级PyTorch到支持RTX 5090 (sm_120)的版本
# 该脚本会升级PyTorch和相关依赖

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}升级PyTorch以支持RTX 5090 (sm_120)${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""

# 检查是否在diffbev环境中
if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" != "diffbev" ]]; then
    echo -e "${YELLOW}⚠️  警告: 请先激活 diffbev 环境${NC}"
    echo "   运行: micromamba activate diffbev"
    exit 1
fi

echo -e "${GREEN}✓ 当前环境: $CONDA_DEFAULT_ENV${NC}"
echo ""

# 检查当前PyTorch版本
echo -e "${BLUE}检查当前PyTorch版本...${NC}"
CURRENT_TORCH=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "未安装")
echo "  当前PyTorch: $CURRENT_TORCH"
echo ""

# 显示升级选项
echo -e "${YELLOW}可用选项:${NC}"
echo "  1. PyTorch 2.7.0+ nightly (CUDA 12.4) - 推荐，支持sm_120"
echo "  2. PyTorch 2.5.0+ (CUDA 12.4) - 稳定版本"
echo "  3. PyTorch 2.1.0+ (CUDA 12.1) - 较旧但稳定"
echo ""
read -p "请选择选项 (1/2/3，默认1): " choice
choice=${choice:-1}

# 根据选择设置PyTorch版本和CUDA版本
case $choice in
    1)
        TORCH_INDEX="https://download.pytorch.org/whl/nightly/cu124"
        TORCH_VERSION="nightly"
        CUDA_VERSION="12.4"
        echo -e "${GREEN}选择: PyTorch nightly (CUDA 12.4)${NC}"
        ;;
    2)
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        TORCH_VERSION="2.5.0"
        CUDA_VERSION="12.4"
        echo -e "${GREEN}选择: PyTorch 2.5.0 (CUDA 12.4)${NC}"
        ;;
    3)
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        TORCH_VERSION="2.1.0"
        CUDA_VERSION="12.1"
        echo -e "${GREEN}选择: PyTorch 2.1.0 (CUDA 12.1)${NC}"
        ;;
    *)
        echo -e "${RED}无效选项，使用默认选项1${NC}"
        TORCH_INDEX="https://download.pytorch.org/whl/nightly/cu124"
        TORCH_VERSION="nightly"
        CUDA_VERSION="12.4"
        ;;
esac

echo ""
echo -e "${YELLOW}升级步骤:${NC}"
echo "  1. 卸载当前PyTorch相关包"
echo "  2. 安装新的PyTorch (CUDA $CUDA_VERSION)"
echo "  3. 验证安装"
echo "  4. 检查GPU支持"
echo ""
read -p "是否继续？(Y/n): " confirm
if [[ $confirm =~ ^[Nn]$ ]]; then
    echo "取消升级"
    exit 0
fi

echo ""
echo -e "${BLUE}步骤1: 卸载当前PyTorch...${NC}"
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
echo -e "${GREEN}✓ PyTorch已卸载${NC}"
echo ""

echo -e "${BLUE}步骤2: 安装新的PyTorch (CUDA $CUDA_VERSION)...${NC}"
if [ "$TORCH_VERSION" = "nightly" ]; then
    pip install --pre torch torchvision torchaudio --index-url "$TORCH_INDEX"
else
    pip install torch==${TORCH_VERSION} torchvision torchaudio --index-url "$TORCH_INDEX"
fi
echo ""

echo -e "${BLUE}步骤3: 验证安装...${NC}"
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}')"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PyTorch安装成功${NC}"
else
    echo -e "${RED}❌ PyTorch安装验证失败${NC}"
    exit 1
fi
echo ""

echo -e "${BLUE}步骤4: 检查GPU支持...${NC}"
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    # 测试CUDA操作
    try:
        x = torch.randn(2, 3).cuda()
        print(f'✓ CUDA测试成功: {x.shape}')
        print(f'✓ GPU可用，可以开始训练')
    except RuntimeError as e:
        print(f'⚠️  CUDA测试失败: {e}')
        print(f'  可能需要检查CUDA驱动版本')
else:
    print('⚠️  CUDA不可用')
"

echo ""
echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}升级完成！${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""
echo "下一步："
echo "  1. 检查MMCV兼容性（可能需要升级）"
echo "  2. 运行测试训练验证GPU可用性"
echo ""
echo "测试GPU："
echo "  python -c \"import torch; x = torch.randn(2, 3).cuda(); print('GPU测试成功:', x.shape)\""
echo ""
echo "如果遇到MMCV相关错误，可能需要升级MMCV："
echo "  pip uninstall mmcv-full -y"
echo "  pip install openmim"
echo "  mim install mmcv>=2.0.0 mmengine"
echo ""
