#!/bin/bash
# 安装支持RTX 5090 (sm_120)的PyTorch nightly版本（CUDA 12.8）

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}安装PyTorch nightly (CUDA 12.8) 以支持RTX 5090${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""

if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" != "diffbev" ]]; then
    echo -e "${YELLOW}⚠️  请先激活 diffbev 环境${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 当前环境: $CONDA_DEFAULT_ENV${NC}"
echo ""

echo -e "${YELLOW}注意:${NC}"
echo "  - 将安装PyTorch nightly版本（预发行版）"
echo "  - 需要CUDA 12.8+驱动"
echo "  - 可能需要升级MMCV到2.x"
echo ""
read -p "是否继续？(Y/n): " confirm
if [[ $confirm =~ ^[Nn]$ ]]; then
    exit 0
fi

echo ""
echo -e "${BLUE}卸载当前PyTorch...${NC}"
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

echo ""
echo -e "${BLUE}安装PyTorch nightly (CUDA 12.8)...${NC}"
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

echo ""
echo -e "${BLUE}验证安装...${NC}"
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    
    # 测试GPU操作
    try:
        x = torch.randn(2, 3).cuda()
        print(f'✓ GPU测试成功: {x.shape}')
        print(f'✓ RTX 5090支持确认')
    except RuntimeError as e:
        print(f'⚠️  GPU测试失败: {e}')
        print(f'   可能需要检查CUDA驱动或PyTorch版本')
"

echo ""
echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}安装完成！${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""
