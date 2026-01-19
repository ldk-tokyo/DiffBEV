#!/bin/bash
# 使用runfile方式安装CUDA 12.4（推荐，避免依赖问题）

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0;m'

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}使用runfile方式安装CUDA 12.4 toolkit${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""

DOWNLOAD_URL="https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run"
INSTALLER_FILE="cuda_12.4.0_550.54.14_linux.run"
INSTALL_DIR="/usr/local/cuda-12.4"

echo -e "${BLUE}步骤1: 下载CUDA 12.4 runfile安装包...${NC}"
echo "下载地址: $DOWNLOAD_URL"
echo ""

if [ ! -f "$INSTALLER_FILE" ]; then
    echo "开始下载（约3.5GB，可能需要一些时间）..."
    wget -c "$DOWNLOAD_URL" -O "$INSTALLER_FILE"
    
    if [ ! -f "$INSTALLER_FILE" ]; then
        echo -e "${RED}❌ 下载失败${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ 下载完成${NC}"
else
    echo -e "${GREEN}✓ 安装包已存在: $INSTALLER_FILE${NC}"
fi

echo ""
echo -e "${BLUE}步骤2: 准备安装...${NC}"
echo ""

# 检查是否有权限
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}需要sudo权限来安装${NC}"
fi

echo ""
echo -e "${BLUE}步骤3: 运行安装程序${NC}"
echo ""
echo -e "${YELLOW}安装选项说明:${NC}"
echo "  - [X] CUDA Toolkit 12.4           (必需)"
echo "  - [ ] Driver                       (如果已安装驱动，取消选择)"
echo "  - [ ] Documentation                (可选)"
echo "  - [ ] Samples                      (可选)"
echo "  - [ ] nsight-systems               (可选，可能有依赖问题)"
echo ""
echo "安装路径: $INSTALL_DIR"
echo ""

read -p "按Enter继续安装，或Ctrl+C取消..."

# 运行安装程序（静默模式，只安装toolkit）
echo "开始安装（这可能需要几分钟）..."
sudo sh "$INSTALLER_FILE" \
    --toolkit \
    --silent \
    --toolkitpath="$INSTALL_DIR" \
    --no-opengl-libs \
    --no-man-page

if [ $? -eq 0 ] && [ -d "$INSTALL_DIR" ]; then
    echo ""
    echo -e "${GREEN}✓ CUDA 12.4安装成功！${NC}"
    echo ""
    
    # 验证安装
    if [ -f "$INSTALL_DIR/bin/nvcc" ]; then
        echo -e "${GREEN}验证安装:${NC}"
        "$INSTALL_DIR/bin/nvcc" --version
        echo ""
        
        echo -e "${BLUE}下一步:${NC}"
        echo "运行以下命令配置环境并编译MMCV:"
        echo ""
        echo "  cd /media/ldk950413/data0/DiffBEV"
        echo "  bash 安装系统CUDA12.4.sh"
        echo ""
    else
        echo -e "${YELLOW}⚠️  安装完成，但nvcc未找到${NC}"
    fi
else
    echo ""
    echo -e "${RED}❌ 安装失败${NC}"
    echo ""
    echo "可以尝试交互式安装:"
    echo "  sudo sh $INSTALLER_FILE"
    exit 1
fi
