#!/bin/bash
# 准备提交到GitHub的脚本
# 用途：添加源代码文件，排除__pycache__、日志、checkpoint等

set -e

PROJECT_ROOT="/media/ldk950413/data0/DiffBEV"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "准备提交代码到GitHub"
echo "=========================================="

# Step 1: 移除已跟踪的__pycache__文件
echo ""
echo "Step 1: 移除已跟踪的__pycache__文件..."
git rm -r --cached mmseg/**/__pycache__/ 2>/dev/null || true
git rm -r --cached tools/__pycache__/ 2>/dev/null || true
echo "✓ 完成"

# Step 2: 添加.gitignore
echo ""
echo "Step 2: 添加.gitignore..."
git add .gitignore
echo "✓ 完成"

# Step 3: 添加源代码文件
echo ""
echo "Step 3: 添加源代码文件..."

# 核心代码
git add mmseg/apis/train.py
git add mmseg/core/evaluation/eval_hooks.py
git add mmseg/core/hooks/
git add mmseg/datasets/nuscenes.py
git add mmseg/utils/runner_compat.py
git add mmseg/utils/metrics_logger.py
git add mmseg/utils/__init__.py

# 配置文件
git add configs/baseline/
git add configs/diffbev/

# 脚本
git add scripts/training/
git add scripts/monitoring/

# 工具
git add tools/plot_metrics.py
git add tools/vis_nuscenes_bev.py

# 文档
git add docs_zh-CN/

# 删除的文件（CUDA安装文件）- 如果存在则添加
[ -f cuda-keyring_1.1-1_all.deb ] && git add cuda-keyring_1.1-1_all.deb || true
[ -f cuda_12.4.0_550.54.14_linux.run ] && git add cuda_12.4.0_550.54.14_linux.run || true

echo "✓ 完成"

# Step 4: 显示暂存的文件
echo ""
echo "=========================================="
echo "暂存的文件列表："
echo "=========================================="
git status --short | grep -E "^A|^M|^D" | head -30

echo ""
echo "=========================================="
echo "准备完成！"
echo ""
echo "下一步："
echo "1. 检查上面的文件列表，确认无误"
echo "2. 运行: git commit -m '你的提交信息'"
echo "3. 运行: git push origin inmage-test"
echo "=========================================="
