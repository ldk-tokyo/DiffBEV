# Baseline 配置文件说明

## lss_swin_nuscenes.py

这是DiffBEV的baseline配置，用于复现**不使用diffusion模块**的原始LSS (Lift-Splat-Shoot) Transformer模型。

### 核心特点

1. **完全关闭diffusion模块**
   - 使用 `BEVSegmentor`（不包含diffusion功能）
   - 使用 `TransformerLiftSplatShoot`（原始LSS transformer）
   - 使用 `PyramidHead`（只包含分割损失，无diffusion损失）

2. **关闭深度相关功能**
   - `outdepth=False`：不输出深度信息
   - `depthsup=False`：不使用深度监督

3. **训练设置符合论文规范**
   - Batch size: 4 per GPU
   - 图像分辨率: 800x600
   - 优化器: AdamW (lr=2e-4, weight_decay=0.01)
   - 训练迭代数: 200k iterations
   - Warmup: 1500 iterations
   - 评估间隔: 每20k iterations

### 使用方法

```bash
# 一键启动baseline训练
bash scripts/training/run_baseline_nuscenes.sh
```

训练结果将自动保存到 `runs/baseline/` 目录：
- 日志文件：`train_YYYYMMDD_HHMMSS.log`
- Checkpoints：`iter_*.pth`
- 最佳模型：`best_mIoU.pth`

### 模型架构

```
Swin-T Backbone (768 dim)
    ↓
LSS Transformer (View Transformation)
    ↓
BEV Features (64 channels, 98x100)
    ↓
PyramidHead (Topdown Network)
    ↓
Classification Head
    ↓
BEV Semantic Segmentation (14 classes)
```

### 损失函数

只使用分割损失（Balanced Binary Cross-Entropy Loss）：
- `loss_seg`: 平衡的二元交叉熵损失，处理类别不平衡问题
- 使用类别权重：`sqrt_inverse` 模式

**不包含**：
- ❌ Diffusion损失 (Ldiff)
- ❌ 深度损失 (Ldepth)

### 与完整DiffBEV模型的区别

| 特性 | Baseline | Full DiffBEV |
|------|----------|--------------|
| View Transformer | LSS Transformer | LSS Transformer |
| Diffusion模块 | ❌ 关闭 | ✅ 启用 |
| 深度输出 | ❌ 关闭 | ✅ 可选 |
| 深度监督 | ❌ 关闭 | ✅ 可选 |
| 损失函数 | 仅分割损失 | 分割 + Diffusion + 深度 |

### 数据集要求

- 数据集路径：`/media/ldk950413/data0/nuScenes`
- 训练集：`img_dir/train` 和 `ann_bev_dir/train`
- 验证集：`img_dir/val` 和 `ann_bev_dir/val`

如需修改数据集路径，请编辑配置文件中的 `data_root` 字段。
