#!/bin/bash
# 清理 .efficient_test 临时目录的脚本

EFFICIENT_TEST_DIR="/media/ldk950413/data0/DiffBEV/.efficient_test"

echo "正在清理 .efficient_test 临时目录..."
echo "目录: $EFFICIENT_TEST_DIR"

if [ -d "$EFFICIENT_TEST_DIR" ]; then
    SIZE=$(du -sh "$EFFICIENT_TEST_DIR" 2>/dev/null | cut -f1)
    echo "当前大小: $SIZE"
    read -p "确认删除? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$EFFICIENT_TEST_DIR"
        echo "已删除临时目录"
    else
        echo "取消操作"
    fi
else
    echo "目录不存在"
fi
