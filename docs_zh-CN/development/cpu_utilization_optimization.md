# CPU利用率优化指南

## CPU核心数分析

**系统配置**：
- CPU核心数：32核
- 当前workers_per_gpu：12（仅使用37.5%的核心）
- **优化后workers_per_gpu：20（使用62.5%的核心）**

## Workers数量选择原则

### 推荐配置

对于32核CPU，推荐使用：
- **workers_per_gpu = 20**（62.5%的核心利用率）

**原因**：
1. **留出核心给系统**：需要保留一些核心给操作系统、GPU驱动、其他进程
2. **避免过度竞争**：太多workers可能导致进程间竞争，反而降低效率
3. **经验值**：通常使用CPU核心数的50-75%作为workers

### Workers数量对比

| Workers | CPU利用率 | 适用场景 | 备注 |
|---------|----------|---------|------|
| 8 | 25% | 保守配置 | 适合有其他重要进程运行 |
| 12 | 37.5% | 平衡配置 | 当前配置，未充分利用 |
| **20** | **62.5%** | **推荐配置** | **充分利用CPU，留出核心给系统** |
| 24 | 75% | 激进配置 | 如果系统负载低可以尝试 |
| 32 | 100% | 不推荐 | 可能导致系统响应变慢 |

## 优化效果

### 预期提升

从12个workers增加到20个workers：
- **数据加载速度**：提升约40-60%
- **GPU利用率**：从可能的等待状态提升到持续工作
- **整体训练速度**：提升约15-25%

### 内存影响

每个worker大约占用：
- 基础内存：50-100MB
- 数据缓存：取决于prefetch_factor和batch size

**20个workers总内存**：约1-2GB（可接受）

## 配置更新

### 已更新的配置文件

1. **`configs/_base_/datasets/nuscene.py`**
   - workers_per_gpu: 8 → 20
   - 添加prefetch_factor=4

2. **`configs/_base_/datasets/nuscene_optimized.py`**
   - workers_per_gpu: 12 → 20
   - 保持prefetch_factor=4

3. **`configs/_base_/datasets/nuscene_high_memory.py`**
   - workers_per_gpu: 8 → 20
   - 添加prefetch_factor=4

## 验证方法

### 1. 检查CPU利用率

```bash
# 实时监控CPU使用情况
watch -n 1 "top -bn1 | grep 'Cpu(s)'"

# 或使用htop（如果已安装）
htop
```

### 2. 检查数据加载是否成为瓶颈

```bash
# 监控训练时的CPU和GPU使用
watch -n 1 nvidia-smi

# 如果GPU利用率<80%，可能是数据加载瓶颈
# 如果GPU利用率>95%，数据加载充足
```

### 3. 性能测试

```bash
# 测试不同workers数量的性能
for workers in 12 16 20 24; do
    echo "Testing with workers=$workers"
    python tools/train.py configs/baseline/lss_swin_nuscenes_optimized.py \
        --work-dir runs/test_workers_$workers \
        --gpu-ids 0 \
        --options runner.max_iters=100 data.workers_per_gpu=$workers
done
```

## 进一步优化

### 如果20个workers还不够

如果发现GPU利用率仍然<80%，可以尝试：

1. **增加到24个workers**：
   ```python
   workers_per_gpu=24  # 75%核心利用率
   ```

2. **增加prefetch_factor**：
   ```python
   prefetch_factor=8  # 从4增加到8
   ```

3. **检查数据存储**：
   - 确保数据在SSD上，而不是HDD
   - 如果数据在HDD，考虑使用更少的workers（HDD I/O是瓶颈）

### 如果系统响应变慢

如果发现系统响应变慢，可以减少workers：

```python
workers_per_gpu=16  # 50%核心利用率，更保守
```

## 注意事项

1. **内存使用**：
   - 每个worker会占用内存
   - 确保有足够的RAM（建议至少32GB）

2. **数据存储**：
   - SSD vs HDD：SSD可以支持更多workers
   - 网络存储：可能需要减少workers

3. **其他进程**：
   - 如果有其他重要进程运行，减少workers
   - 监控系统负载，避免过载

4. **PyTorch版本**：
   - prefetch_factor需要PyTorch >= 1.8.0
   - persistent_workers需要PyTorch >= 1.7.0

## 预期效果总结

| 指标 | 优化前 (12 workers) | 优化后 (20 workers) | 提升 |
|------|-------------------|-------------------|------|
| CPU利用率 | 37.5% | 62.5% | +66% |
| 数据加载速度 | 基准 | +40-60% | - |
| GPU利用率 | 可能<80% | >90% | +10-15% |
| 训练速度 | 基准 | +15-25% | - |
| 内存使用 | 基准 | +0.5-1GB | 可接受 |

## 快速应用

所有配置文件已更新，直接使用即可：

```bash
# 使用优化配置
bash scripts/training/run_baseline_optimized.sh

# 或使用高显存配置
bash scripts/training/run_baseline_high_memory.sh
```

配置会自动使用20个workers，充分利用32核CPU。
