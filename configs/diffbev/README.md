# DiffBEV 配置文件说明

## diffbev_lss_swin_nuscenes.py

这是DiffBEV的完整配置，启用**diffusion模块**，用于复现完整的DiffBEV模型。

### 核心特点

1. **启用diffusion模块**
   - 使用 `DiffusionHead`（支持diffusion功能）
   - 使用 `TransformerLiftSplatShoot`（带深度输出和监督）

2. **Diffusion配置**
   - **条件输入**: FS-BEV（Full-Scale BEV条件）
   - **融合方式**: Cross-Attention（跨注意力融合）
   - **xt编码方式**: 论文默认实现（通常是Conv编码）
   - **扩散步数**: 1000步（论文默认）
   - **噪声调度**: linear（线性调度）

3. **深度相关功能**
   - `outdepth=True`：输出深度信息（diffusion需要）
   - `depthsup=True`：使用深度监督（diffusion需要）

4. **损失函数配置（按照论文规范）**
   - `Lseg = Lwce + 10*Ldepth + 1*Ldiff`
   - `loss_depth_weight=10.0`：深度损失权重
   - `loss_diff_weight=1.0`：扩散损失权重

5. **训练设置符合论文规范**
   - Batch size: 4 per GPU
   - 图像分辨率: 800x600
   - 优化器: AdamW (lr=2e-4, weight_decay=0.01)
   - 训练迭代数: 200k iterations
   - Warmup: 1500 iterations
   - 评估间隔: 每20k iterations

### 使用方法

```bash
# 一键启动DiffBEV训练
bash scripts/training/run_diffbev_nuscenes.sh
```

训练结果将自动保存到 `runs/diffbev_default/` 目录：
- 日志文件：`train_YYYYMMDD_HHMMSS.log`
- Checkpoints：`iter_*.pth`
- 最佳模型：`best_mIoU.pth`

### 模型架构

```
Swin-T Backbone (768 dim)
    ↓
LSS Transformer (View Transformation)
    ↓  [输出BEV特征 + 深度信息]
BEV Features (64 channels, 98x100) + Depth
    ↓
DiffusionHead
    ├─ FS-BEV条件输入（Full-Scale BEV）
    ├─ Cross-Attention融合
    ├─ xt编码（论文默认）
    └─ Diffusion过程（1000步）
    ↓
BEV Semantic Segmentation (14 classes)
```

### 损失函数

使用组合损失函数（按照论文规范）：
- `loss_seg`: 分割损失（Balanced BCE Loss）
- `loss_depth`: 深度损失（权重=10.0）
- `loss_diff`: 扩散损失（权重=1.0）

总损失：`Ltotal = Lseg + 10*Ldepth + 1*Ldiff`

### 与Baseline的区别

| 特性 | Baseline | DiffBEV |
|------|----------|---------|
| **View Transformer** | LSS Transformer | LSS Transformer |
| **Diffusion模块** | ❌ 关闭 | ✅ 启用 |
| **条件输入** | - | FS-BEV |
| **融合方式** | - | Cross-Attention |
| **深度输出** | ❌ 关闭 | ✅ 启用 |
| **深度监督** | ❌ 关闭 | ✅ 启用（权重=10） |
| **损失函数** | 仅分割损失 | 分割 + 深度 + 扩散 |

### 配置参数说明

#### DiffusionHead参数

- `use_diffusion=True`: 启用diffusion模块
- `condition_type='FS-BEV'`: 使用Full-Scale BEV作为条件输入
- `fusion_type='Cross-Attention'`: 使用跨注意力机制进行特征融合
- `xt_encoding='default'`: 使用论文默认的xt编码方式
- `loss_depth_weight=10.0`: 深度损失权重（论文规范）
- `loss_diff_weight=1.0`: 扩散损失权重（论文规范）
- `diffusion_steps=1000`: 扩散过程的步数（论文默认）
- `noise_schedule='linear'`: 噪声调度方式（线性调度）

### 数据集要求

- 数据集路径：`/media/ldk950413/data0/nuScenes`
- 训练集：`img_dir/train` 和 `ann_bev_dir/train`
- 验证集：`img_dir/val` 和 `ann_bev_dir/val`

如需修改数据集路径，请编辑配置文件中的 `data_root` 字段。

### 注意事项

1. **内存需求**: Diffusion模块需要更多GPU内存，建议至少16GB显存
2. **训练时间**: 相比baseline，DiffBEV训练时间会更长（因为有diffusion过程）
3. **模型实现**: 确保 `DiffusionHead` 已正确实现并注册到MMSegmentation中
4. **依赖检查**: 确保所有diffusion相关的依赖已安装

### 常见问题

#### Q: 如果遇到 `DiffusionHead` 未找到的错误？
A: 需要确保 `DiffusionHead` 已在 `mmseg/models/decode_heads/__init__.py` 中注册。

#### Q: 如果GPU内存不足？
A: 可以尝试：
- 减小 `samples_per_gpu`（batch size）
- 启用 `efficient_test=True`（已启用）
- 使用梯度累积

#### Q: 如何修改条件输入类型？
A: 修改 `condition_type` 参数：
- `'FS-BEV'`: Full-Scale BEV（默认）
- `'FO-BEV'`: First-Order BEV
- `'sum'`: 求和条件

#### Q: 如何修改融合方式？
A: 修改 `fusion_type` 参数：
- `'Cross-Attention'`: 跨注意力融合（默认）
- `'Concat'`: 特征拼接
- `'Add'`: 特征相加
