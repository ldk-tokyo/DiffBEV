# MMCV CUDA问题最终解决方案

## 问题总结

**根本原因**: CUDA版本不匹配
- 系统CUDA: 13.0
- PyTorch CUDA: 12.4
- MMCV 1.3.15需要: CUDA 11.0

**尝试的解决方案**:
1. ❌ 选项2: 安装/配置CUDA 11.0 runtime库 - 系统中不存在
2. ❌ 选项3: 从源码编译MMCV - CUDA版本不匹配导致编译失败

## 最终可行的方案

### 方案A: 创建符号链接（风险较高，但可能有效）

尝试创建一个符号链接，让MMCV 1.3.15能找到CUDA库：

```bash
# 找到PyTorch的CUDA 12.4库
PYTORCH_LIB=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")

# 在conda环境中创建符号链接
CONDA_LIB="$CONDA_PREFIX/lib"

# 查找CUDA 12的cudart库
CUDA12_LIB=$(find "$CONDA_LIB" -name "libcudart.so.12*" 2>/dev/null | head -1)

if [ -n "$CUDA12_LIB" ]; then
    # 创建符号链接指向CUDA 12库（虽然不完美，但可能有效）
    cd "$CONDA_LIB"
    ln -sf "$(basename $CUDA12_LIB)" libcudart.so.11.0 || true
    echo "已创建符号链接: libcudart.so.11.0 -> $(basename $CUDA12_LIB)"
fi
```

**风险**: CUDA 11.0和12.4的API可能不完全兼容，可能导致运行时错误。

### 方案B: 使用CPU模式训练（已验证可用）

这是最可靠的临时方案：

```bash
bash run_baseline_cpu.sh
```

**优点**: 稳定可靠，已验证可用
**缺点**: 训练速度慢

### 方案C: 降级PyTorch到1.9.1（恢复原始环境）

恢复到原始的PyTorch 1.9.1 + MMCV 1.3.15组合，然后使用CPU模式：

```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.15 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
export FORCE_CPU=1
bash run_baseline_cpu.sh
```

## 推荐方案

**立即执行**: 方案B（CPU模式）- 已验证可用，可以立即开始训练

**长期方案**: 
1. 等待PyTorch官方发布完整支持sm_120的稳定版本
2. 或等待MMCV发布与PyTorch 2.4.1 + CUDA 12.4匹配的预编译版本
3. 或使用兼容的GPU（如RTX 3090/4090）运行PyTorch 1.9.1

## 执行命令

### 测试方案A（符号链接）:
```bash
cd /media/ldk950413/data0/DiffBEV
bash -c '
CONDA_LIB="$CONDA_PREFIX/lib"
CUDA12_LIB=$(find "$CONDA_LIB" -path "*/nvidia/cuda_runtime/lib/libcudart.so.12" 2>/dev/null | head -1)
if [ -n "$CUDA12_LIB" ]; then
    cd "$(dirname "$CUDA12_LIB")"
    ln -sf libcudart.so.12 libcudart.so.11.0 2>/dev/null || true
    echo "符号链接已创建"
    export DIFFBEV_ALLOW_PYTORCH2=1
    python -c "from mmcv.ops import nms; print(\"✓ MMCV可用\")" 2>&1
fi
'
```

### 使用方案B（CPU模式）:
```bash
cd /media/ldk950413/data0/DiffBEV
bash run_baseline_cpu.sh
```
