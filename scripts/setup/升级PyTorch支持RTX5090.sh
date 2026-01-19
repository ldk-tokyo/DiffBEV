#!/bin/bash
# 升级PyTorch以支持RTX 5090 (sm_120)架构

set -e

echo "=========================================="
echo "升级PyTorch以支持RTX 5090 (sm_120)"
echo "=========================================="
echo ""

# 检查当前环境
echo "1. 检查当前PyTorch版本..."
python -c "import torch; print(f'当前PyTorch版本: {torch.__version__}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" || {
    echo "错误: 无法导入torch，请先安装PyTorch"
    exit 1
}

# 检查GPU
echo ""
echo "2. 检查GPU信息..."
python -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)} (计算能力: {torch.cuda.get_device_capability(i)})') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print('无GPU可用')" || {
    echo "警告: 无法检查GPU信息"
}

# 升级到PyTorch nightly（支持sm_120）
echo ""
echo "3. 升级PyTorch到nightly版本（支持RTX 5090）..."
echo "注意: nightly版本可能不稳定，建议在测试环境使用"

# 卸载旧版本
echo "卸载旧版本PyTorch..."
pip uninstall -y torch torchvision torchaudio || true

# 安装nightly版本（CUDA 12.4）
echo "安装PyTorch nightly (CUDA 12.4)..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# 验证安装
echo ""
echo "4. 验证安装..."
python -c "
import torch
print(f'✓ PyTorch版本: {torch.__version__}')
print(f'✓ CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA版本: {torch.version.cuda}')
    print(f'✓ GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        print(f'✓ GPU {i}: {name} (计算能力: {cap})')
    
    # 测试GPU操作
    try:
        x = torch.randn(2, 3).cuda()
        print(f'✓ GPU测试成功！Tensor shape: {x.shape}')
        print('✓ RTX 5090支持已启用！')
    except Exception as e:
        print(f'✗ GPU测试失败: {e}')
        print('✗ 可能仍不支持RTX 5090，请尝试其他方案')
        exit(1)
else:
    print('✗ CUDA不可用')
    exit(1)
"

echo ""
echo "=========================================="
echo "升级完成！"
echo "=========================================="
echo ""
echo "如果验证失败，可能需要："
echo "1. 检查CUDA驱动版本（需要 >= 550）"
echo "2. 检查CUDA toolkit版本"
echo "3. 尝试从源码编译PyTorch"
echo ""
echo "参考文档："
echo "- PyTorch安装指南: https://pytorch.org/get-started/locally/"
echo "- RTX 5090兼容性说明: RTX5090_CUDA错误解决方案.md"