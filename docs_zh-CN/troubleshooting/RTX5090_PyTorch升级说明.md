# RTX 5090 PyTorch升级说明

## 当前问题

**错误信息**:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**原因**:
- RTX 5090 使用 **sm_120** 架构（Blackwell架构）
- PyTorch 2.4.1+cu124 只支持 sm_50 到 sm_90，**不支持 sm_120**
- 需要在支持sm_120的PyTorch版本上运行

## 解决方案

### ✅ 推荐方案：升级到PyTorch nightly

PyTorch nightly版本通常包含最新的GPU架构支持，包括RTX 5090的sm_120。

#### 方法1：使用升级脚本（推荐）

```bash
# 运行升级脚本
bash 升级PyTorch支持RTX5090.sh
```

#### 方法2：手动升级

```bash
# 1. 卸载旧版本
pip uninstall -y torch torchvision torchaudio

# 2. 安装nightly版本（CUDA 12.4）
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# 3. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); x = torch.randn(2, 3).cuda(); print('GPU测试成功！')"
```

### ⚠️ 注意事项

1. **Nightly版本稳定性**
   - Nightly版本可能包含未测试的功能
   - 可能不稳定或与某些依赖不兼容
   - 建议在测试环境先验证

2. **依赖兼容性**
   - 升级PyTorch后，可能需要更新MMCV
   - 当前MMCV 2.2.0应该兼容PyTorch nightly
   - 如有问题，参考`MMCV_2.2.0升级记录.md`

3. **CUDA要求**
   - 需要CUDA 12.4+
   - 需要驱动 >= 550
   - 检查命令：`nvidia-smi`

### 🔄 如果nightly版本仍不支持

如果PyTorch nightly版本仍然不支持sm_120，可能需要：

1. **等待PyTorch 2.5+正式版**
   - PyTorch 2.5+应该会正式支持RTX 5090
   - 预计发布时间：2025年上半年

2. **从源码编译PyTorch**
   - 添加sm_120支持到编译配置
   - 需要CUDA toolkit 12.4+
   - 编译时间较长（数小时）

3. **临时使用CPU训练**
   - 仅用于测试代码流程
   - 不适合完整训练（速度太慢）

## 验证步骤

升级后，运行以下命令验证：

```bash
# 1. 检查PyTorch版本
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"

# 2. 检查GPU信息
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'计算能力: {torch.cuda.get_device_capability(0)}')"

# 3. 测试GPU操作
python -c "import torch; x = torch.randn(2, 3).cuda(); print('GPU测试成功！')"

# 4. 运行训练脚本（小批量测试）
python tools/train.py configs/baseline/lss_swin_nuscenes.py --work-dir runs/baseline --gpu-ids 0 --cfg-options total_iters=10
```

## 当前状态

- ✅ PyTorch 2.4.1+cu124 已安装
- ✅ MMCV 2.2.0 已安装（兼容PyTorch 2.x）
- ✅ 代码已更新以支持PyTorch 2.x
- ❌ **PyTorch 2.4.1不支持RTX 5090（sm_120）**
- ⏳ **需要升级到PyTorch nightly或等待2.5+版本**

## 相关文档

- **PyTorch安装指南**: https://pytorch.org/get-started/locally/
- **RTX 5090兼容性说明**: `RTX5090_CUDA错误解决方案.md`
- **PyTorch 2.x支持说明**: `PyTorch_2.x正式支持说明.md`
- **MMCV升级记录**: `MMCV_2.2.0升级记录.md`