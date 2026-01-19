# 显存利用率优化总结

## 当前状态

- **显存使用**: 10.5GB / 32.6GB (33%)
- **GPU利用率**: 92%
- **Batch Size**: 4 per GPU
- **优化空间**: 可以提升到60-80%

## 已创建的优化配置

### 1. 高显存利用率配置（推荐）

**配置文件**: `configs/baseline/lss_swin_nuscenes_high_memory.py`

**优化内容**:
- Batch Size: 4 → 8
- FP16混合精度训练: 启用
- 学习率: 2e-4 → 4e-4（线性缩放）

**预期效果**:
- 显存利用率: 33% → 60-65%
- 训练速度: 提升2-3倍
- 精度影响: <0.5%

**启动方式**:
```bash
bash scripts/training/run_baseline_high_memory.sh
```

### 2. 仅增加Batch Size（保守方案）

如果不想使用FP16，可以只增加batch size：

**修改配置**:
```python
# configs/_base_/datasets/nuscene.py
data = dict(
    samples_per_gpu=8,  # 从4增加到8
    # ... 其他配置
)

# configs/_base_/schedules/schedule_200k_nuscenes.py
optimizer = dict(
    lr=4e-4,  # 从2e-4增加到4e-4
    # ... 其他配置
)
```

**预期效果**:
- 显存利用率: 33% → 50-55%
- 训练速度: 提升约15-20%
- 精度影响: 无

### 3. 仅启用FP16（如果batch size不能增加）

**修改配置**:
```python
# 在配置文件中继承
_base_ = [
    '../_base_/datasets/nuscene.py',
    '../_base_/default_runtime_fp16.py',  # 使用FP16配置
    '../_base_/schedules/schedule_200k_nuscenes.py'
]
```

**预期效果**:
- 显存使用: 10GB → 5-6GB（减少50%）
- 训练速度: 提升50-100%
- 精度影响: <0.5%
- 允许使用更大的batch size

## 快速测试

### 测试不同配置的显存使用

```bash
# 运行测试脚本（会测试3种配置）
bash scripts/training/test_memory_configs.sh
```

### 手动测试

```bash
# 测试1: Batch Size=8
python tools/train.py configs/baseline/lss_swin_nuscenes.py \
    --work-dir runs/test_bs8 \
    --gpu-ids 0 \
    --options runner.max_iters=100 data.samples_per_gpu=8 optimizer.lr=4e-4

# 测试2: Batch Size=8 + FP16
python tools/train.py configs/baseline/lss_swin_nuscenes_high_memory.py \
    --work-dir runs/test_bs8_fp16 \
    --gpu-ids 0 \
    --options runner.max_iters=100
```

## 监控显存使用

### 实时监控

```bash
# 方法1: 使用nvidia-smi
watch -n 1 nvidia-smi

# 方法2: 使用诊断脚本
watch -n 5 bash scripts/utils/check_training_performance.sh
```

### 检查训练日志

```bash
# 查看显存使用情况
tail -f runs/baseline_high_memory/train_*.log | grep -E "(memory|显存|GPU)"
```

## 注意事项

1. **学习率调整**:
   - Batch size翻倍 → 学习率翻倍（线性缩放规则）
   - 或使用梯度累积保持原学习率

2. **FP16兼容性**:
   - RTX 5090完全支持FP16（Tensor Cores）
   - 某些操作可能不支持FP16，会自动回退到FP32

3. **如果OOM**:
   - 减少batch size（8 → 6）
   - 或只启用FP16，不增加batch size
   - 或使用梯度累积

4. **精度验证**:
   - 对比FP32和FP16的mIoU
   - 如果精度下降>1%，考虑只增加batch size

## 推荐方案

**最佳方案**: Batch Size=8 + FP16
- 配置文件: `configs/baseline/lss_swin_nuscenes_high_memory.py`
- 启动脚本: `scripts/training/run_baseline_high_memory.sh`
- 预期显存: 18-20GB (60-65%)
- 预期速度: 提升2-3倍

**保守方案**: 只增加Batch Size到8
- 修改: `configs/_base_/datasets/nuscene.py` 和 `schedule_200k_nuscenes.py`
- 预期显存: 16-18GB (50-55%)
- 预期速度: 提升15-20%

## 故障排除

### 问题1: FP16导致NaN

**解决方案**:
- 增加loss_scale: `fp16 = dict(loss_scale=1024.0)`
- 或禁用FP16，只使用更大的batch size

### 问题2: OOM错误

**解决方案**:
- 减少batch size: 8 → 6 → 4
- 或只启用FP16，不增加batch size

### 问题3: 训练速度没有提升

**检查**:
- GPU利用率是否提升
- 数据加载是否成为瓶颈（检查workers_per_gpu）
- FP16是否真正启用（查看日志）

## 相关文件

- `configs/baseline/lss_swin_nuscenes_high_memory.py` - 高显存配置
- `configs/_base_/datasets/nuscene_high_memory.py` - 数据配置
- `configs/_base_/default_runtime_fp16.py` - FP16配置
- `configs/_base_/schedules/schedule_200k_nuscenes_bs8.py` - 学习率配置
- `scripts/training/run_baseline_high_memory.sh` - 启动脚本
- `docs_zh-CN/development/memory_optimization.md` - 详细文档
