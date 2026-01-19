# 训练速度优化指南

## 问题分析

从测试结果看：
- **Batch Size=4**: 7.45 iter/s
- **Batch Size=8**: 3.64 iter/s（速度降低51%）

**原因分析**：
1. Batch size翻倍后，每次迭代的计算量增加，但GPU利用率可能未充分利用
2. 数据加载可能成为瓶颈（虽然workers已增加）
3. FP16可能未充分利用Tensor Cores
4. 其他性能瓶颈（内存分配、同步等）

## 优化策略

### 策略1：平衡Batch Size（推荐）

**方案**：使用Batch Size=6 + FP16

**优点**：
- 显存利用率：45-55%（15-18GB）
- 训练速度：比batch size=8快，比batch size=4稍慢但吞吐量更高
- 学习率：3e-4（线性缩放）

**配置文件**：`configs/baseline/lss_swin_nuscenes_optimized.py`

### 策略2：数据加载优化

**优化项**：
1. **增加workers**：从8增加到12（充分利用32核CPU）
2. **启用prefetch**：`prefetch_factor=4`（PyTorch 2.0+）
3. **保持persistent_workers**：避免重复创建进程
4. **启用pin_memory**：加速CPU到GPU传输

**配置示例**：
```python
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=12,  # 增加workers
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,  # 预取4个batch
    # ...
)
```

### 策略3：PyTorch 2.0编译优化（实验性）

**使用torch.compile加速模型**：

```python
# 在train.py中添加
if hasattr(torch, 'compile'):
    model = torch.compile(model, mode='reduce-overhead')
```

**注意**：
- 需要PyTorch >= 2.0
- 首次运行会编译模型，可能较慢
- 后续运行会显著加速

### 策略4：混合精度优化

**确保FP16充分利用Tensor Cores**：

1. **检查FP16是否启用**：
   - 查看日志中是否有"FP16混合精度训练已启用"
   - 使用`nvidia-smi dmon`监控Tensor Core使用率

2. **优化loss_scale**：
   - 如果出现NaN，增加`loss_scale`（默认512.0）
   - 如果训练稳定，可以尝试降低以提升速度

3. **确保操作支持FP16**：
   - 某些操作会自动回退到FP32（这是正常的）
   - 但应避免不必要的FP32操作

### 策略5：减少评估频率

**优化评估间隔**：

```python
evaluation = dict(
    interval=40000,  # 从20000增加到40000，减少评估次数
    metric='mIoU',
    efficient_test=True,
    pre_eval=True
)
```

### 策略6：梯度累积（如果batch size太大）

**如果batch size=8导致速度下降，可以使用梯度累积**：

```python
runner = dict(
    type='IterBasedRunner',
    max_iters=200000,
    accumulative_counts=2  # 累积2次梯度，等效batch size=16
)
```

然后使用batch size=4，等效batch size=8。

## 推荐配置

### 配置1：平衡版（推荐）

**文件**：`configs/baseline/lss_swin_nuscenes_optimized.py`

**特点**：
- Batch Size=6
- FP16启用
- Workers=12
- Prefetch=4
- 预期速度：5-6 iter/s
- 预期显存：15-18GB

### 配置2：速度优先版

**修改**：在optimized配置基础上
- Batch Size=4（保持原样）
- FP16启用
- Workers=16（最大化）
- Prefetch=8
- 预期速度：8-9 iter/s（FP16加速后）
- 预期显存：8-10GB

### 配置3：显存优先版

**文件**：`configs/baseline/lss_swin_nuscenes_high_memory.py`

**特点**：
- Batch Size=8
- FP16启用
- 预期速度：4-5 iter/s
- 预期显存：18-20GB

## 性能监控

### 监控GPU利用率

```bash
# 实时监控GPU使用情况
watch -n 1 nvidia-smi

# 监控Tensor Core使用率（需要nvidia-smi dmon）
nvidia-smi dmon -s u
```

### 监控数据加载

```bash
# 检查数据加载是否成为瓶颈
# 如果GPU利用率<80%，可能是数据加载瓶颈
# 如果GPU利用率>95%，可能是计算瓶颈
```

### 性能分析

```bash
# 使用PyTorch Profiler
python -m torch.utils.bottleneck tools/train.py configs/baseline/lss_swin_nuscenes_optimized.py \
    --work-dir runs/profiling \
    --gpu-ids 0 \
    --options runner.max_iters=100
```

## 预期效果对比

| 配置 | Batch Size | FP16 | Workers | 预期速度 | 显存使用 | 推荐场景 |
|------|-----------|------|---------|---------|---------|---------|
| 原始 | 4 | 否 | 8 | 7.45 iter/s | 10GB | 基准 |
| 高显存 | 8 | 是 | 8 | 3.64 iter/s | 18-20GB | 显存充足但速度慢 |
| **优化版** | **6** | **是** | **12** | **5-6 iter/s** | **15-18GB** | **推荐** |
| 速度优先 | 4 | 是 | 16 | 8-9 iter/s | 8-10GB | 速度优先 |

## 实施步骤

1. **测试优化配置**：
   ```bash
   python tools/train.py configs/baseline/lss_swin_nuscenes_optimized.py \
       --work-dir runs/test_optimized \
       --gpu-ids 0 \
       --options runner.max_iters=100
   ```

2. **监控性能**：
   - 观察iter/s速度
   - 检查GPU利用率
   - 确认显存使用

3. **如果速度仍不理想**：
   - 尝试PyTorch 2.0编译
   - 进一步优化数据加载
   - 考虑使用梯度累积

4. **正式训练**：
   ```bash
   bash scripts/training/run_baseline_optimized.sh
   ```

## 注意事项

1. **学习率调整**：Batch size变化时，学习率需要线性缩放
2. **FP16稳定性**：如果出现NaN，增加loss_scale
3. **数据加载**：确保数据在SSD上，而不是HDD
4. **CPU核心数**：Workers数量不应超过CPU核心数
5. **内存**：增加workers会增加内存使用，确保有足够RAM

## 故障排除

### 问题1：速度没有提升

**检查**：
- GPU利用率是否提升
- 数据加载是否成为瓶颈
- FP16是否真正启用

**解决**：
- 增加workers
- 启用prefetch
- 检查数据存储位置（SSD vs HDD）

### 问题2：显存不足

**解决**：
- 减少batch size
- 减少workers
- 只启用FP16，不增加batch size

### 问题3：训练不稳定

**解决**：
- 检查FP16 loss_scale
- 减少batch size
- 检查学习率是否正确缩放
