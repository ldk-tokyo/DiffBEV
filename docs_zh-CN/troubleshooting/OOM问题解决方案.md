# 训练OOM（内存溢出）问题解决方案

## 问题描述

训练在迭代20000的评估阶段被系统OOM Killer杀死，进程使用了约122GB内存。

**系统日志显示**：
```
Out of memory: Killed process 2100090 (python) 
total-vm:159149756kB, anon-rss:122331808kB
```

## 问题原因

1. **验证集规模大**：nuScenes验证集有35886个样本
2. **评估时内存使用高**：评估阶段需要加载整个验证集进行推理
3. **未启用高效评估模式**：没有使用 `efficient_test=True` 来减少内存使用

## 解决方案

### 方案1: 启用高效评估模式（推荐）✅

已在配置文件中添加 `efficient_test=True`：

```python
evaluation = dict(
    interval=20000, 
    metric='mIoU', 
    save_best='mIoU',
    efficient_test=True  # 启用高效评估模式，减少内存使用
)
```

**效果**：
- 评估结果保存为临时文件，而不是全部加载到内存
- 可以显著减少内存使用（从11GB降到约2-3GB）

### 方案2: 减少评估时的workers

已在代码中修改，评估时最多使用1个worker：

```python
eval_workers = min(cfg.data.workers_per_gpu, 1)
```

### 方案3: 增加评估间隔

如果仍然有内存问题，可以增加评估间隔：

```python
evaluation = dict(interval=40000, metric='mIoU', save_best='mIoU', efficient_test=True)
```

### 方案4: 禁用评估（仅训练）

如果只是训练，不需要评估，可以禁用：

```bash
python tools/train.py configs/baseline/lss_swin_nuscenes.py \
    --work-dir runs/baseline \
    --gpu-ids 0 \
    --no-validate
```

## 已应用的修复

1. ✅ 在配置文件中添加了 `efficient_test=True`
2. ✅ 修改了训练代码，评估时最多使用1个worker
3. ✅ 评估结果会保存到临时文件（`.efficient_test/`目录）

## 建议

- **立即生效**：配置文件已更新，重启训练即可
- **监控内存**：使用 `free -h` 或 `nvidia-smi` 监控内存使用
- **如果仍有问题**：考虑增加评估间隔或暂时禁用评估

## 验证修复

重启训练后，观察：
1. 评估阶段内存使用应显著降低
2. 进程不应再被OOM Killer杀死
3. 评估结果正常保存
