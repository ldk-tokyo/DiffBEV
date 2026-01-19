# 训练指标对比可视化工具

## 功能说明

`tools/plot_metrics.py` 用于从多个实验的 `metrics.csv` 文件中读取指标数据，生成对比曲线图和总结报告。

### 主要功能

1. **自动读取多个runs目录下的metrics.csv**
   - 自动搜索 `metrics.csv` 文件
   - 支持子目录中的文件

2. **生成对比曲线**
   - mIoU（平均IoU）
   - NDS（NuScenes Detection Score）
   - mAP（mean Average Precision）
   - Lwce（加权交叉熵损失）
   - Ldepth（深度损失）
   - Ldiff（扩散损失）
   - lr（学习率）

3. **输出PNG图像**
   - 保存到 `reports/plots/` 目录
   - 高分辨率PNG格式

4. **自动生成summary.md**
   - 列出每个实验的最佳指标
   - 显示对应的checkpoint路径

## 使用方法

### 基本用法

对比两个实验的指标：

```bash
python tools/plot_metrics.py \
    runs/baseline \
    runs/diffbev_default \
    --output-dir reports/comparison
```

### 对比多个实验

```bash
python tools/plot_metrics.py \
    runs/baseline \
    runs/diffbev_default \
    runs/experiment1 \
    runs/experiment2 \
    --output-dir reports/multi_comparison
```

### 自定义实验名称

```bash
python tools/plot_metrics.py \
    runs/baseline \
    runs/diffbev_default \
    --experiment-names "Baseline (LSS)" "DiffBEV (FS-BEV)" \
    --output-dir reports/named_comparison
```

### 自定义图像参数

```bash
python tools/plot_metrics.py \
    runs/baseline \
    runs/diffbev_default \
    --output-dir reports/comparison \
    --fig-size 16 8 \
    --dpi 200
```

## 参数说明

### 必需参数

- `runs_dirs`: 一个或多个runs目录路径，每个目录应包含 `metrics.csv` 文件

### 可选参数

- `--output-dir`: 输出目录（默认: `reports`）
  - 图像保存在 `{output-dir}/plots/`
  - summary.md保存在 `{output-dir}/summary.md`

- `--experiment-names`: 实验名称列表（默认: 使用目录名）
  - 数量必须与runs_dirs数量一致

- `--fig-size`: 图像大小，格式为 width height（默认: 12 6）

- `--dpi`: 图像分辨率（默认: 150）

## 输出说明

### 目录结构

```
reports/
├── plots/
│   ├── mIoU_comparison.png
│   ├── NDS_comparison.png
│   ├── mAP_comparison.png
│   ├── Lwce_comparison.png
│   ├── Ldepth_comparison.png
│   ├── Ldiff_comparison.png
│   └── lr_comparison.png
└── summary.md
```

### summary.md格式

```markdown
# 实验指标总结

## Baseline
**实验目录**: `runs/baseline`

### 最佳指标
| 指标 | 最佳值 | Step |
|------|--------|------|
| mIoU | 0.6523 | 20000 |
| Lwce_avg | 0.4123 | 20000 |

### Checkpoints
- **best_mIoU.pth**: `runs/baseline/best_mIoU.pth`

---
```

## 示例

### 最小可运行示例

假设已有两个实验的metrics.csv：

```bash
# 对比baseline和diffbev
python tools/plot_metrics.py \
    runs/baseline \
    runs/diffbev_default \
    --output-dir reports/comparison
```

### 完整示例

```bash
python tools/plot_metrics.py \
    runs/baseline \
    runs/diffbev_default \
    runs/experiment_ablation1 \
    runs/experiment_ablation2 \
    --experiment-names \
        "Baseline" \
        "DiffBEV (FS-BEV)" \
        "Ablation: w/o Diffusion" \
        "Ablation: w/o Depth Loss" \
    --output-dir reports/full_comparison \
    --fig-size 14 7 \
    --dpi 200
```

## 注意事项

1. **metrics.csv格式**: 脚本期望CSV文件包含以下列：
   - `step` 或 `iter` 或 `epoch`（用于x轴）
   - `mIoU`, `NDS`, `mAP`（评估指标）
   - `Lwce`, `Ldepth`, `Ldiff`（损失指标）
   - `lr`（学习率）

2. **缺失指标**: 如果某个实验缺少某些指标，该指标会被跳过，不影响其他指标的绘制

3. **Checkpoint查找**: 脚本会自动查找以下常见checkpoint文件名：
   - `best_mIoU.pth`
   - `latest.pth`
   - `epoch_*.pth`
   - `iter_*.pth`

4. **最佳值标记**: 图中会用星号标记每个实验的最佳值点

## 技术细节

### 最佳指标计算

- **评估指标** (mIoU, NDS, mAP): 取最大值及其对应的step
- **损失指标** (Lwce, Ldepth, Ldiff): 取最后100个step的平均值作为稳定值

### 图像绘制

- 使用matplotlib的非交互式后端（Agg）
- 每个实验使用不同颜色
- 自动网格和图例
- 支持自定义图像大小和DPI

### CSV文件格式

脚本兼容由 `MetricsLogger` 生成的CSV格式：
- 第一列通常是 `step`, `iter`, 或 `epoch`
- 指标列可能包含NaN值（训练过程中的评估指标）
- 自动处理缺失列
