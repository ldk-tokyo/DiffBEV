#!/bin/bash
# 清理已合并的分支和备份分支

set -e

PROJECT_ROOT="/media/ldk950413/data0/DiffBEV"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "分支清理工具"
echo "=========================================="
echo ""
echo "当前分支状态："
git branch -a
echo ""

# 检查已合并到main的分支
MERGED_BRANCHES=$(git branch --merged main | grep -v "main" | grep -v "backup" | sed 's/^[ *]*//')
BACKUP_BRANCHES=$(git branch | grep "backup" | sed 's/^[ *]*//')

echo "已合并到main的分支："
if [ -z "$MERGED_BRANCHES" ]; then
    echo "  无"
else
    echo "$MERGED_BRANCHES" | sed 's/^/  - /'
fi
echo ""

echo "备份分支："
if [ -z "$BACKUP_BRANCHES" ]; then
    echo "  无"
else
    echo "$BACKUP_BRANCHES" | sed 's/^/  - /'
fi
echo ""

# 选项菜单
echo "请选择操作："
echo "1) 删除已合并的本地分支（inmage-test）"
echo "2) 删除远程已合并的分支（origin/inmage-test）"
echo "3) 删除所有备份分支"
echo "4) 执行所有清理操作"
echo "5) 取消"
echo ""
read -p "请输入选项 (1-5): " choice

case $choice in
    1)
        if [ -n "$MERGED_BRANCHES" ]; then
            echo ""
            echo "删除已合并的本地分支..."
            echo "$MERGED_BRANCHES" | xargs -I {} git branch -d {}
            echo "✓ 完成"
        else
            echo "没有需要删除的已合并分支"
        fi
        ;;
    2)
        echo ""
        echo "删除远程已合并的分支..."
        if git ls-remote --heads origin inmage-test | grep -q inmage-test; then
            git push origin --delete inmage-test
            echo "✓ 已删除远程分支 origin/inmage-test"
        else
            echo "远程分支 origin/inmage-test 不存在或已删除"
        fi
        ;;
    3)
        if [ -n "$BACKUP_BRANCHES" ]; then
            echo ""
            echo "删除备份分支..."
            echo "$BACKUP_BRANCHES" | xargs -I {} git branch -D {}
            echo "✓ 完成"
        else
            echo "没有备份分支需要删除"
        fi
        ;;
    4)
        echo ""
        echo "执行所有清理操作..."
        
        # 删除已合并的本地分支
        if [ -n "$MERGED_BRANCHES" ]; then
            echo "删除已合并的本地分支..."
            echo "$MERGED_BRANCHES" | xargs -I {} git branch -d {}
        fi
        
        # 删除远程已合并的分支
        echo "删除远程已合并的分支..."
        if git ls-remote --heads origin inmage-test | grep -q inmage-test; then
            git push origin --delete inmage-test
            echo "✓ 已删除远程分支 origin/inmage-test"
        fi
        
        # 删除备份分支
        if [ -n "$BACKUP_BRANCHES" ]; then
            echo "删除备份分支..."
            echo "$BACKUP_BRANCHES" | xargs -I {} git branch -D {}
        fi
        
        echo ""
        echo "=========================================="
        echo "清理完成！"
        echo "=========================================="
        ;;
    5)
        echo "已取消"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "当前分支状态："
git branch -a
