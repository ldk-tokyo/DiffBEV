#!/bin/bash
# 清理Git历史中的大文件
# 警告：这会重写Git历史，需要强制推送

set -e

PROJECT_ROOT="/media/ldk950413/data0/DiffBEV"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "清理Git历史中的大文件"
echo "=========================================="
echo ""
echo "警告：此操作会重写Git历史！"
echo "如果已经推送到远程，需要使用 --force 推送"
echo ""
read -p "是否继续？(y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 1
fi

# 检查是否安装了git-filter-repo（推荐）或git filter-branch
if command -v git-filter-repo &> /dev/null; then
    echo ""
    echo "使用 git-filter-repo 清理历史..."
    
    # 移除大文件
    git-filter-repo --path cuda_12.4.0_550.54.14_linux.run --invert-paths
    git-filter-repo --path cuda-keyring_1.1-1_all.deb --invert-paths
    git-filter-repo --path runs/baseline/best_mIoU.pth --invert-paths
    git-filter-repo --path runs/ --invert-paths
    
    echo "✓ 完成"
else
    echo ""
    echo "使用 git filter-branch 清理历史（较慢）..."
    
    # 移除大文件
    git filter-branch --force --index-filter \
        "git rm --cached --ignore-unmatch cuda_12.4.0_550.54.14_linux.run cuda-keyring_1.1-1_all.deb runs/baseline/best_mIoU.pth" \
        --prune-empty --tag-name-filter cat -- --all
    
    # 清理runs目录
    git filter-branch --force --index-filter \
        "git rm -rf --cached --ignore-unmatch runs/" \
        --prune-empty --tag-name-filter cat -- --all
    
    echo "✓ 完成"
    
    # 清理引用
    echo ""
    echo "清理引用..."
    rm -rf .git/refs/original/
    git reflog expire --expire=now --all
    git gc --prune=now --aggressive
    echo "✓ 完成"
fi

echo ""
echo "=========================================="
echo "清理完成！"
echo ""
echo "下一步："
echo "1. 检查历史大小: git count-objects -vH"
echo "2. 强制推送: git push origin inmage-test --force"
echo "=========================================="
