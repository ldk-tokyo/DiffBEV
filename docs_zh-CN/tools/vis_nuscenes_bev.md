# nuScenes BEV分割结果可视化工具

## 功能说明

`tools/vis_nuscenes_bev.py` 用于可视化nuScenes BEV分割的对比结果，支持：
- 对比GT、Baseline预测、DiffBEV预测
- 使用统一的色表和阈值规则
- 批量处理多个样本
- 结果保存为PNG图像

## 使用方法

### 基本用法（仅Baseline）

```bash
python tools/vis_nuscenes_bev.py \
    configs/baseline/lss_swin_nuscenes.py \
    runs/baseline/best_mIoU.pth \
    --indices 0 10 20 30 \
    --work-dir runs/vis_comparison
```

### 完整对比（Baseline + DiffBEV）

```bash
python tools/vis_nuscenes_bev.py \
    configs/baseline/lss_swin_nuscenes.py \
    runs/baseline/best_mIoU.pth \
    --diffbev-config configs/diffbev/diffbev_lss_swin_nuscenes.py \
    --diffbev-checkpoint runs/diffbev_default/best_mIoU.pth \
    --indices 0 10 20 30 40 50 \
    --work-dir runs/vis_comparison \
    --threshold 0.5
```

## 参数说明

### 必需参数

- `baseline_config`: Baseline模型的配置文件路径
- `baseline_checkpoint`: Baseline模型的checkpoint路径

### 可选参数

- `--diffbev-config`: DiffBEV模型的配置文件路径（可选）
- `--diffbev-checkpoint`: DiffBEV模型的checkpoint路径（可选）
- `--indices`: 要可视化的样本索引列表（默认: 0 10 20 30）
- `--work-dir`: 输出目录（默认: work_dirs/vis）
- `--device`: 推理设备（默认: cuda:0）
- `--threshold`: 二值化阈值（默认: 0.5）
- `--fig-size`: 图像大小，格式为 width height（默认: 24 8）

## 输出说明

结果保存在 `{work_dir}/vis/` 目录下，文件命名格式：
- `sample_0000.png`: 样本0的可视化结果
- `sample_0010.png`: 样本10的可视化结果
- ...

每个图像包含三列对比：
- **左列**: Ground Truth
- **中列**: Baseline预测
- **右列**: DiffBEV预测（如果提供）

## 示例

### 最小可运行示例

假设已有训练好的baseline模型：

```bash
# 1. 激活环境
conda activate diffbev

# 2. 运行可视化（仅Baseline，单个样本）
python tools/vis_nuscenes_bev.py \
    configs/baseline/lss_swin_nuscenes.py \
    runs/baseline/best_mIoU.pth \
    --indices 0 \
    --work-dir runs/vis_baseline
```

### 完整对比示例

```bash
# Baseline + DiffBEV 对比可视化
python tools/vis_nuscenes_bev.py \
    configs/baseline/lss_swin_nuscenes.py \
    runs/baseline/best_mIoU.pth \
    --diffbev-config configs/diffbev/diffbev_lss_swin_nuscenes.py \
    --diffbev-checkpoint runs/diffbev_default/best_mIoU.pth \
    --indices 0 10 20 30 40 50 60 70 80 90 \
    --work-dir runs/vis_comparison \
    --threshold 0.5 \
    --fig-size 30 10
```

## 注意事项

1. **Checkpoint路径**: 确保checkpoint文件存在且路径正确
2. **配置文件**: 确保配置文件路径正确，且与训练时使用的配置一致
3. **数据集**: 脚本会自动使用配置文件中的数据集设置
4. **内存**: 如果处理大量样本，注意GPU内存使用
5. **阈值**: 默认阈值为0.5，可根据实际效果调整

## 技术细节

### 色表

使用nuScenes数据集的统一色表（定义在 `mmseg/datasets/nuscenes.py` 中）：
- 14个类别，每个类别有对应的RGB颜色
- 使用 `visualize_map_mask()` 函数进行可视化

### 推理流程

1. 加载模型和checkpoint
2. 从数据集中获取指定索引的样本
3. 对样本进行推理，获得logits
4. 使用sigmoid + 阈值进行二值化
5. 转换为可视化图像并保存

### 输出格式

- PNG格式，150 DPI
- 三列布局：GT | Baseline | DiffBEV
- 图像标题显示样本索引和阈值
