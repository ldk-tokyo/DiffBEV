# FP16训练NaN梯度问题修复指南

## 问题描述

在FP16训练过程中，检测到梯度为NaN：
```
RuntimeError: ❌ 训练终止: Iter 27295 时检测到参数 'backbone.patch_embed.projection.weight' 的梯度为 NaN！
```

## 原因分析

FP16训练中NaN梯度的常见原因：
1. **Loss scale太小**：梯度下溢，导致NaN
2. **梯度爆炸**：梯度值过大，超出FP16范围
3. **数值不稳定**：某些操作在FP16下不稳定
4. **学习率过大**：与FP16结合时更容易出现NaN

## 修复方案

### 方案1：增加Loss Scale（已应用）

**修改**：`configs/_base_/default_runtime_fp16.py`
```python
fp16 = dict(loss_scale=1024.0)  # 从512增加到1024
```

**原理**：
- Loss scale用于防止FP16梯度下溢
- 更大的loss scale可以处理更大的梯度范围
- 如果仍然出现NaN，可以继续增加到2048或4096

### 方案2：添加梯度裁剪（已应用）

**修改**：`configs/_base_/schedules/schedule_200k_nuscenes_bs6.py`
```python
optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2)  # 梯度裁剪：最大范数35
)
```

**原理**：
- 限制梯度范数，防止梯度爆炸
- 提高训练稳定性
- 对FP16训练特别重要

### 方案3：改进NaN处理（已应用）

**修改**：`mmseg/utils/runner_compat.py`

**改进内容**：
1. **使用GradScaler.unscale_()**：在step之前unscale梯度，可以检测NaN
2. **自动跳过NaN更新**：如果检测到NaN，跳过此次更新并降低loss_scale
3. **记录警告而非终止**：允许训练继续，scaler会自动调整

**行为**：
- 如果检测到NaN，会跳过此次更新
- GradScaler会自动降低loss_scale
- 训练会继续，不会终止

## 恢复训练

### 方法1：从checkpoint恢复（推荐）

```bash
# 找到最新的checkpoint（在NaN之前的）
ls -lh runs/baseline_optimized/*.pth

# 从checkpoint恢复训练
python tools/train.py configs/baseline/lss_swin_nuscenes_optimized.py \
    --work-dir runs/baseline_optimized \
    --gpu-ids 0 \
    --resume-from runs/baseline_optimized/iter_27000.pth  # 使用NaN之前的checkpoint
```

### 方法2：从头开始（如果checkpoint不可用）

```bash
# 使用修复后的配置重新训练
bash scripts/training/run_baseline_optimized.sh
```

## 进一步优化

### 如果仍然出现NaN

1. **进一步增加loss_scale**：
   ```python
   fp16 = dict(loss_scale=2048.0)  # 或4096.0
   ```

2. **降低学习率**：
   ```python
   optimizer = dict(lr=2e-4)  # 从3e-4降低到2e-4
   ```

3. **使用更小的batch size**：
   ```python
   samples_per_gpu=4  # 从6降低到4
   ```

4. **禁用FP16（最后手段）**：
   ```python
   # 移除fp16配置，使用FP32训练
   # 虽然速度慢，但更稳定
   ```

### 监控训练

```bash
# 监控训练日志，查看loss_scale变化
tail -f runs/baseline_optimized/train_*.log | grep -E "(loss_scale|NaN|跳过)"

# 监控GPU使用
watch -n 1 nvidia-smi
```

## 预期效果

修复后：
- **Loss scale自动调整**：如果出现NaN，scaler会自动降低loss_scale
- **训练不会终止**：遇到NaN会跳过此次更新，继续训练
- **稳定性提升**：梯度裁剪防止梯度爆炸
- **日志记录**：会记录NaN事件和loss_scale变化

## 验证修复

训练开始后，检查日志中是否有：
```
✅ FP16混合精度训练已启用 (loss_scale=1024.0)
```

如果出现NaN，应该看到：
```
⚠️  Iter XXXX: 检测到参数 'XXX' 的梯度为 NaN/inf，跳过此次更新。当前loss_scale: XXX.XX
```

而不是训练终止。

## 注意事项

1. **Loss scale会自动调整**：
   - 如果频繁出现NaN，scaler会持续降低loss_scale
   - 如果训练稳定，scaler可能会增加loss_scale

2. **梯度裁剪参数**：
   - `max_norm=35`：根据经验值设置
   - 如果仍然不稳定，可以降低到20或25

3. **学习率**：
   - FP16训练时，学习率可能需要稍微降低
   - 如果频繁出现NaN，考虑降低学习率

4. **Batch size**：
   - 更大的batch size可能增加NaN风险
   - 如果问题持续，考虑减小batch size

## 总结

已应用的修复：
- ✅ 增加loss_scale：512 → 1024
- ✅ 添加梯度裁剪：max_norm=35
- ✅ 改进NaN处理：自动跳过NaN更新
- ✅ 更好的日志记录：记录NaN事件

现在可以安全地恢复训练，即使偶尔出现NaN也不会终止训练。
