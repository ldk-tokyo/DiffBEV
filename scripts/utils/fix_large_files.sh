#!/bin/bash
# 修复Git历史中的大文件问题
# 方案1：使用git filter-branch清理历史（推荐）

set -e

PROJECT_ROOT="/media/ldk950413/data0/DiffBEV"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "修复Git历史中的大文件问题"
echo "=========================================="
echo ""
echo "当前仓库大小:"
git count-objects -vH | grep "size-pack"
echo ""
echo "需要移除的大文件："
echo "  - cuda_12.4.0_550.54.14_linux.run (4.4GB)"
echo "  - cuda-keyring_1.1-1_all.deb (4.3MB)"
echo "  - runs/baseline/best_mIoU.pth (355MB)"
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

# 备份当前分支
echo ""
echo "备份当前分支..."
BACKUP_BRANCH="backup-$(date +%Y%m%d-%H%M%S)"
git branch "$BACKUP_BRANCH"
echo "✓ 已创建备份分支: $BACKUP_BRANCH"

# 确保工作区干净（忽略子模块）
if ! git diff-index --quiet HEAD -- ':!mono-semantic-maps'; then
    echo "警告：工作区有未提交的更改（子模块除外）"
    echo "正在stash..."
    git stash push -m "临时stash before filter-branch"
    STASHED=1
else
    STASHED=0
fi

# 使用git filter-branch移除大文件
echo ""
echo "移除大文件..."
export FILTER_BRANCH_SQUELCH_WARNING=1
git filter-branch --force --index-filter \
    "git rm --cached --ignore-unmatch \
        cuda_12.4.0_550.54.14_linux.run \
        cuda-keyring_1.1-1_all.deb \
        runs/baseline/best_mIoU.pth" \
    --prune-empty --tag-name-filter cat -- --all

# 清理runs目录中的所有文件（如果存在）
echo ""
echo "清理runs目录..."
git filter-branch --force --index-filter \
    "git rm -rf --cached --ignore-unmatch runs/" \
    --prune-empty --tag-name-filter cat -- --all 2>/dev/null || true

# 清理引用和垃圾回收
echo ""
echo "清理引用和垃圾回收..."
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 恢复stash（如果有）
if [ "$STASHED" -eq 1 ]; then
    echo ""
    echo "恢复stash..."
    git stash pop || true
fi

echo ""
echo "=========================================="
echo "清理完成！"
echo ""
echo "新的仓库大小:"
git count-objects -vH | grep "size-pack"
echo ""
echo "下一步："
echo "1. 检查历史: git log --oneline -5"
echo "2. 强制推送: git push origin inmage-test --force"
echo "3. 如果出现问题，可以恢复: git checkout $BACKUP_BRANCH"
echo "=========================================="
