# 训练卡顿优化指南

## 问题现象

训练过程中出现"一顿一顿"的现象，速度波动很大：
- 正常速度：6-7 iter/s
- 突然变慢：1-2 iter/s，甚至 1.03s/iter, 1.24s/iter
- 然后又恢复：3-5 iter/s

## 可能原因

### 1. 数据加载瓶颈（最常见）

**症状**：GPU利用率正常，但训练速度不稳定

**原因**：
- `workers_per_gpu=2` 可能不够
- 数据预处理耗时（图像解码、resize等）
- 数据从磁盘读取慢

**解决方案**：
```python
# 在 configs/_base_/datasets/nuscene.py 中增加
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,  # 从2增加到4或8（根据CPU核心数）
    pin_memory=True,    # 启用pin_memory加速数据传输
    persistent_workers=True,  # 保持worker进程，避免重复创建
    # ...
)
```

### 2. Checkpoint保存阻塞

**症状**：每5000次迭代时速度突然变慢

**原因**：
- Checkpoint文件较大（340MB）
- 保存到磁盘时阻塞训练

**解决方案**：
```python
# 在 configs/_base_/schedules/schedule_200k_nuscenes.py 中
checkpoint_config = dict(
    by_epoch=False, 
    interval=5000,
    max_keep_ckpts=3,  # 只保留最近3个checkpoint
    save_optimizer=False  # 不保存optimizer状态（如果不需要resume）
)
```

### 3. 评估阶段阻塞

**症状**：每20000次迭代时速度突然变慢

**原因**：
- 评估需要遍历整个验证集
- 评估时模型切换到eval模式，影响训练流程

**解决方案**：
```python
# 在 configs/_base_/schedules/schedule_200k_nuscenes.py 中
evaluation = dict(
    interval=20000, 
    metric='mIoU',
    efficient_test=True,  # 启用efficient_test模式，节省内存
    pre_eval=True  # 使用pre_eval模式，避免内存峰值
)
```

### 4. 日志写入频繁

**症状**：每次迭代都写入日志导致IO阻塞

**解决方案**：
```python
# 在 configs/_base_/default_runtime.py 中
log_config = dict(
    interval=50,  # 每50次迭代记录一次日志（而不是每次）
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False)
    ]
)
```

### 5. 内存不足导致Swap

**症状**：系统内存使用率高，swap被使用

**检查方法**：
```bash
free -h
# 如果Swap被使用，说明内存不足
```

**解决方案**：
- 减少 `samples_per_gpu`
- 减少 `workers_per_gpu`
- 使用 `efficient_test=True` 进行评估

### 6. GPU利用率不稳定

**症状**：GPU利用率波动大

**检查方法**：
```bash
nvidia-smi dmon -s u -c 10
```

**解决方案**：
- 增加数据预加载：`prefetch_factor=2`
- 使用 `pin_memory=True`
- 确保数据管道足够快

## 推荐配置

### 优化后的数据配置

```python
# configs/_base_/datasets/nuscene.py
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,  # 增加到4（如果CPU核心数足够，可以到8）
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,  # 预加载2个batch
    train=dict(...),
    val=dict(...),
    test=dict(...)
)
```

### 优化后的训练配置

```python
# configs/_base_/schedules/schedule_200k_nuscenes.py
checkpoint_config = dict(
    by_epoch=False, 
    interval=5000,
    max_keep_ckpts=3  # 只保留最近3个checkpoint
)

evaluation = dict(
    interval=20000, 
    metric='mIoU',
    efficient_test=True,
    pre_eval=True
)

log_config = dict(
    interval=50,  # 减少日志频率
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False)
    ]
)
```

## 诊断命令

### 1. 检查数据加载速度

```bash
# 监控训练进程的IO
iotop -p $(pgrep -f "python.*train.py") -o

# 检查数据目录的读取速度
dd if=/media/ldk950413/data0/nuScenes/img_dir/train/xxx.jpg of=/dev/null bs=1M count=100
```

### 2. 检查GPU利用率

```bash
# 实时监控GPU利用率
watch -n 1 nvidia-smi

# 或者使用dmon
nvidia-smi dmon -s u -c 100
```

### 3. 检查内存使用

```bash
# 监控内存和swap使用
watch -n 1 free -h

# 检查训练进程的内存使用
ps aux | grep "python.*train.py" | awk '{print $2}' | xargs -I {} cat /proc/{}/status | grep -E "VmRSS|VmSwap"
```

### 4. 检查磁盘IO

```bash
# 监控磁盘IO
iostat -x 1

# 检查checkpoint保存时的IO
strace -e trace=write -p $(pgrep -f "python.*train.py") 2>&1 | grep -E "\.pth"
```

## 快速修复

如果训练正在进行中，可以：

1. **临时增加workers**（需要重启训练）：
   ```bash
   # 修改配置文件后重启训练
   ```

2. **减少checkpoint保存频率**（需要重启训练）：
   ```python
   checkpoint_config = dict(by_epoch=False, interval=10000)  # 从5000改为10000
   ```

3. **禁用评估**（临时，不推荐）：
   ```bash
   python tools/train.py configs/baseline/lss_swin_nuscenes.py --no-validate
   ```

## 预期效果

优化后应该看到：
- 训练速度稳定在 6-7 iter/s
- GPU利用率稳定在 90%+
- 内存使用稳定，无swap
- 速度波动 < 10%
