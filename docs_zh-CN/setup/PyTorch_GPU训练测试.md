# PyTorch GPU训练测试说明

## 当前状态

- ✅ **PyTorch已升级**: 2.4.1+cu124
- ⚠️ **sm_120支持**: PyTorch 2.4.1 显示不支持sm_120，但GPU测试成功
- ⚠️ **MMCV兼容性**: MMCV 1.3.15 可能与PyTorch 2.4.1不兼容

## 测试方案

虽然PyTorch 2.4.1显示不支持sm_120，但实际测试中GPU操作可以执行。这可能是因为：
1. PyTorch使用兼容模式运行
2. 某些操作可以使用，但不是最优性能

**建议**: 先尝试运行训练，看是否能正常工作。

## 运行训练

```bash
cd /media/ldk950413/data0/DiffBEV
bash run_baseline_nuscenes.sh
```

或直接运行：

```bash
python tools/train.py configs/baseline/lss_swin_nuscenes.py \
    --work-dir runs/baseline \
    --gpu-ids 0
```

## 如果遇到问题

### 问题1: MMCV兼容性错误

如果出现MMCV相关错误，可能需要升级MMCV：

```bash
pip uninstall mmcv-full -y
pip install openmim
mim install mmcv>=2.0.0 mmengine
```

### 问题2: CUDA操作失败

如果出现"no kernel image is available"错误：
- 可能需要安装PyTorch nightly + CUDA 12.8（需要从源码编译或等待官方发布）
- 或使用CPU模式作为临时方案

### 问题3: API不兼容

如果出现API相关错误，可能需要修改代码以适应PyTorch 2.x的API变化。

## 验证GPU可用性

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    x = torch.randn(2, 3).cuda()
    print('GPU测试:', '成功' if x.device.type == 'cuda' else '失败')
"
```

## 下一步

1. 运行训练脚本测试实际效果
2. 如果遇到错误，根据错误信息进行修复
3. 如果需要完整sm_120支持，可能需要等待PyTorch官方发布或从源码编译
