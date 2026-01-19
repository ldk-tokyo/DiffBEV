# RTX 5090 GPU训练问题总结

## 当前状态

### ✅ 已完成
1. **PyTorch升级**: 1.9.1 → 2.4.1+cu124 ✅
2. **GPU操作测试**: PyTorch GPU tensor和卷积操作成功 ✅
3. **环境变量配置**: `DIFFBEV_ALLOW_PYTORCH2=1` ✅

### ❌ 当前问题
1. **MMCV安装失败**: 
   - 系统CUDA版本（13.0）与PyTorch CUDA版本（12.4）不匹配
   - MMCV无法编译或安装预编译版本

## 根本原因

系统环境：
- **系统CUDA**: 13.0
- **PyTorch CUDA**: 12.4
- **不匹配**: 导致MMCV编译失败

## 解决方案

### 方案1：重新安装匹配的PyTorch（推荐）

安装与系统CUDA版本匹配的PyTorch：

```bash
# 卸载当前PyTorch
pip uninstall torch torchvision torchaudio -y

# 安装CUDA 13.0版本的PyTorch（如果有）
# 或者安装CUDA 12.1版本（通常兼容性更好）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 方案2：使用MMCV 1.3.15 + 降级PyTorch（临时方案）

恢复原来的环境，使用CPU模式或等待更好的解决方案：

```bash
# 恢复PyTorch 1.9.1
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# 恢复MMCV
pip install mmcv-full==1.3.15 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# 使用CPU模式训练
export FORCE_CPU=1
bash run_baseline_cpu.sh
```

### 方案3：设置CUDA版本环境变量

尝试强制使用CUDA 12.4（如果系统支持）：

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
```

## 建议

鉴于当前情况，建议：

1. **短期**: 使用CPU模式训练（已实现），验证代码正确性
2. **中期**: 解决CUDA版本匹配问题，升级完整的PyTorch+MMCV环境
3. **长期**: 等待PyTorch官方发布完整支持sm_120的稳定版本

## 测试命令

```bash
# 检查CUDA版本
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# 测试GPU
python -c "import torch; x = torch.randn(2,3).cuda(); print('GPU:', x.device)"

# 测试MMCV
python -c "import mmcv; print('MMCV:', mmcv.__version__)"
```

## 下一步

需要您决定：
1. 继续尝试修复CUDA版本匹配问题
2. 暂时使用CPU模式训练
3. 回退到原始环境（PyTorch 1.9.1 + MMCV 1.3.15）并使用CPU模式
