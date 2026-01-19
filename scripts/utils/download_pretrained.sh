#!/bin/bash
# 下载Swin Transformer预训练权重脚本

set -e

PRETRAINED_DIR="/media/ldk950413/data0/DiffBEV/pretrained"
WEIGHT_FILE="swin_tiny_patch4_window7_224.pth"
WEIGHT_URL="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"

echo "================================================================================
下载Swin Transformer预训练权重
================================================================================"

# 创建目录
mkdir -p "$PRETRAINED_DIR"
cd "$PRETRAINED_DIR"

# 检查文件是否已存在
if [ -f "$WEIGHT_FILE" ]; then
    echo "✓ 预训练权重已存在: $PRETRAINED_DIR/$WEIGHT_FILE"
    echo "  文件大小: $(du -h $WEIGHT_FILE | cut -f1)"
    exit 0
fi

echo "开始下载预训练权重..."
echo "URL: $WEIGHT_URL"
echo "保存路径: $PRETRAINED_DIR/$WEIGHT_FILE"
echo ""

# 下载文件
if command -v wget &> /dev/null; then
    wget -O "$WEIGHT_FILE" "$WEIGHT_URL"
elif command -v curl &> /dev/null; then
    curl -L -o "$WEIGHT_FILE" "$WEIGHT_URL"
else
    echo "错误: 未找到 wget 或 curl，请手动下载:"
    echo "  URL: $WEIGHT_URL"
    echo "  保存到: $PRETRAINED_DIR/$WEIGHT_FILE"
    exit 1
fi

# 检查下载是否成功
if [ -f "$WEIGHT_FILE" ]; then
    echo ""
    echo "✓ 下载成功！"
    echo "  文件路径: $PRETRAINED_DIR/$WEIGHT_FILE"
    echo "  文件大小: $(du -h $WEIGHT_FILE | cut -f1)"
    echo ""
    echo "现在可以在配置文件中使用此路径:"
    echo "  pretrained=\"$PRETRAINED_DIR/$WEIGHT_FILE\""
else
    echo ""
    echo "✗ 下载失败，请手动下载:"
    echo "  URL: $WEIGHT_URL"
    echo "  保存到: $PRETRAINED_DIR/$WEIGHT_FILE"
    exit 1
fi
