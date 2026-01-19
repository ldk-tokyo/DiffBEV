# 显存利用率优化指南

## 当前状态

- **显存使用**: 约10.5GB / 32.6GB (33%)
- **GPU利用率**: 92%
- **Batch Size**: 4 per GPU
- **优化空间**: 显存利用率可以提升到60-80%

## 优化方案

### 方案1: 增加Batch Size（最简单）

**优点**:
- 实现简单，只需修改配置
- 提高训练吞吐量
- 可能提高训练稳定性

**缺点**:
- 需要相应调整学习率（线性缩放）
- 如果OOM，需要回退

**配置修改**:
```python
# configs/_base_/datasets/nuscene.py
data = dict(
    samples_per_gpu=8,  # 从4增加到8
    # ... 其他配置
)
```

**学习率调整**:
```python
# configs/_base_/schedules/schedule_200k_nuscenes.py
optimizer = dict(
    type='AdamW',
    lr=4e-4,  # 从2e-4增加到4e-4（batch size翻倍，学习率翻倍）
    # ... 其他配置
)
```

### 方案2: 启用FP16混合精度训练（推荐）

**优点**:
- 显存使用减少约50%
- 训练速度提升1.5-2倍
- 允许使用更大的batch size
- 对精度影响很小（通常<0.5%）

**缺点**:
- 需要PyTorch >= 1.6.0
- 某些操作可能不支持FP16

**配置修改**:
```python
# configs/_base_/default_runtime.py 或新建 default_runtime_fp16.py
fp16 = dict(loss_scale=512.0)  # 动态loss scaling
```

**使用方式**:
```python
# 在配置文件中继承
_base_ = [
    '../_base_/datasets/nuscene.py',
    '../_base_/default_runtime_fp16.py',  # 使用FP16配置
    '../_base_/schedules/schedule_200k_nuscenes.py'
]
```

### 方案3: 组合优化（Batch Size + FP16）

**预期效果**:
- Batch Size: 4 → 8
- FP16: 启用
- 显存使用: 10GB → 18-20GB (60-65%利用率)
- 训练速度: 提升约2-3倍

**配置文件**: `configs/baseline/lss_swin_nuscenes_high_memory.py`

### 方案4: 梯度累积（如果OOM）

如果直接增加batch size导致OOM，可以使用梯度累积：

```python
# 在runner配置中
runner = dict(
    type='IterBasedRunner', 
    max_iters=200000,
    accumulative_counts=2  # 累积2次梯度，等效batch size=8
)
```

## 显存使用估算

### 当前配置（Batch Size=4）
- 模型参数: ~100MB
- 激活值: ~8GB
- 优化器状态: ~200MB
- 其他: ~2GB
- **总计**: ~10.5GB

### 优化后（Batch Size=8 + FP16）
- 模型参数: ~50MB (FP16)
- 激活值: ~12GB (FP16, batch size翻倍)
- 优化器状态: ~100MB (FP16)
- 其他: ~2GB
- **总计**: ~18-20GB

## 实施步骤

### Step 1: 测试FP16（推荐先测试）

```bash
cd /media/ldk950413/data0/DiffBEV
export DIFFBEV_ALLOW_PYTORCH2=1

# 使用FP16配置进行短时间测试（300 iterations）
python tools/train.py configs/baseline/lss_swin_nuscenes_high_memory.py \
    --work-dir runs/baseline_fp16_test \
    --gpu-ids 0 \
    --options runner.max_iters=300
```

### Step 2: 检查显存使用

```bash
# 在另一个终端监控
watch -n 1 nvidia-smi
```

### Step 3: 如果测试成功，正式训练

```bash
# 从checkpoint恢复（如果需要）
python tools/train.py configs/baseline/lss_swin_nuscenes_high_memory.py \
    --work-dir runs/baseline_high_memory \
    --gpu-ids 0 \
    --resume-from runs/baseline/best_mIoU.pth \
    2>&1 | tee runs/baseline_high_memory/train.log
```

## 注意事项

1. **学习率调整**: 
   - Batch size翻倍 → 学习率翻倍（线性缩放）
   - 或使用梯度累积保持原学习率

2. **FP16兼容性**:
   - 确保PyTorch >= 1.6.0
   - RTX 5090完全支持FP16（Tensor Cores）

3. **监控训练**:
   - 观察loss是否正常下降
   - 检查是否有NaN
   - 对比FP32和FP16的精度差异

4. **回退方案**:
   - 如果FP16导致精度下降，可以只增加batch size
   - 如果OOM，可以减少batch size或使用梯度累积

## 预期收益

| 方案 | 显存利用率 | 训练速度 | 精度影响 |
|------|-----------|---------|---------|
| 当前 | 33% | 基准 | - |
| Batch Size=8 | 50-55% | +15% | 无 |
| FP16 | 20-25% | +50-100% | <0.5% |
| Batch Size=8 + FP16 | 60-65% | +100-200% | <0.5% |

## 快速测试脚本

```bash
#!/bin/bash
# 快速测试不同配置的显存使用

echo "测试1: Batch Size=4 (当前)"
python tools/train.py configs/baseline/lss_swin_nuscenes.py \
    --work-dir runs/test_bs4 --gpu-ids 0 \
    --options runner.max_iters=100 data.samples_per_gpu=4

echo "测试2: Batch Size=8"
python tools/train.py configs/baseline/lss_swin_nuscenes.py \
    --work-dir runs/test_bs8 --gpu-ids 0 \
    --options runner.max_iters=100 data.samples_per_gpu=8

echo "测试3: Batch Size=8 + FP16"
python tools/train.py configs/baseline/lss_swin_nuscenes_high_memory.py \
    --work-dir runs/test_bs8_fp16 --gpu-ids 0 \
    --options runner.max_iters=100
```
