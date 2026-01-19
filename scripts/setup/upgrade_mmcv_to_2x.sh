#!/bin/bash
# MMCV 升级到 2.x 安装脚本

set -e

echo "================================================================================="
echo "MMCV 升级到 2.x 安装脚本"
echo "================================================================================="
echo ""

# 检查是否在diffbev环境中
if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" != "diffbev" ]]; then
    echo "⚠️  警告: 请先激活 diffbev 环境"
    echo "   运行: micromamba activate diffbev"
    exit 1
fi

echo "✓ 当前环境: $CONDA_DEFAULT_ENV"
echo ""

# 检查PyTorch版本
echo "检查PyTorch版本..."
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "未安装")
echo "  PyTorch版本: $PYTORCH_VERSION"

# 解析主版本号
PYTORCH_MAJOR=$(echo $PYTORCH_VERSION | cut -d. -f1)

if [[ "$PYTORCH_MAJOR" -lt 2 ]]; then
    echo ""
    echo "⚠️  警告: 检测到 PyTorch $PYTORCH_VERSION"
    echo "   MMCV 2.x 需要 PyTorch 2.x"
    echo "   如果使用 RTX 5090，建议先升级 PyTorch"
    echo ""
    read -p "是否继续？(yes/no): " continue_install
    if [[ "$continue_install" != "yes" ]]; then
        echo "已取消。"
        exit 0
    fi
fi

echo ""
echo "================================================================================="
echo "步骤 1: 卸载旧版本 MMCV"
echo "================================================================================="
echo ""

pip uninstall -y mmcv mmcv-full

echo ""
echo "================================================================================="
echo "步骤 2: 安装 MMCV 2.x"
echo "================================================================================="
echo ""

# 检查是否有mim
if ! command -v mim &> /dev/null; then
    echo "安装 openmim..."
    pip install -U openmim
fi

echo "使用 mim 安装 MMCV 2.x..."
mim install mmcv>=2.1.0

echo ""
echo "================================================================================="
echo "步骤 3: 安装 MMEngine（必需）"
echo "================================================================================="
echo ""

pip install mmengine>=0.10.0

echo ""
echo "================================================================================="
echo "步骤 4: 验证安装"
echo "================================================================================="
echo ""

python << EOF
try:
    import mmcv
    import mmengine
    print(f"✓ MMCV版本: {mmcv.__version__}")
    print(f"✓ MMEngine版本: {mmengine.__version__}")
    
    # 检查基本导入
    from mmengine import Config
    from mmengine.runner import load_checkpoint
    from mmengine.model import BaseModule
    print("✓ 基本API导入成功")
    
    print("\n✓ MMCV 2.x 安装成功！")
except Exception as e:
    print(f"\n✗ 安装验证失败: {e}")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 验证失败，请检查错误信息"
    exit 1
fi

echo ""
echo "================================================================================="
echo "升级完成！"
echo "================================================================================="
echo ""
echo "下一步："
echo "1. 运行自动迁移脚本: python migrate_mmcv_to_2x.py"
echo "2. 或手动按照 'mmcv升级指南.md' 进行代码迁移"
echo "3. 测试: python tools/train.py configs/... --work-dir work_dirs/test"
echo ""
