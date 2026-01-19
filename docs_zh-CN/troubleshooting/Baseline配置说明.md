# Baseline配置说明

## 概述

本baseline配置使用**原始LSS (Lift-Splat-Shoot) Transformer**作为视图变换方法，**完全关闭diffusion模块与相关loss**，其他训练设置保持与论文规范一致。

## 配置文件

**位置**：`configs/baseline/lss_swin_nuscenes.py`

### 关键配置

#### 1. 模型架构
- **Segmentor类型**：`BEVSegmentor`（自定义实现，支持transformer和calib）
- **Backbone**：Swin Transformer（与论文规范一致）
- **Transformer**：`TransformerLiftSplatShoot`（LSS方法，baseline）
  - `outdepth=False`：关闭深度输出
  - `depthsup=False`：关闭深度监督
- **Decode Head**：`PyramidHead`（只包含Lwce损失，无Ldepth和Ldiff）

#### 2. 损失函数
- **只包含Lwce**：加权交叉熵损失（通过`OccupancyCriterion`实现）
- **无Ldepth**：深度损失已关闭（`depthsup=False`）
- **无Ldiff**：扩散模型损失已关闭（无diffusion模块）

#### 3. 训练设置（与论文规范一致）
- **优化器**：AdamW
- **学习率**：2e-4
- **权重衰减**：0.01
- **总迭代次数**：200,000
- **Warmup迭代次数**：1,500
- **输入分辨率**：800×600
- **Batch Size**：4 per GPU

#### 4. 数据集
- **数据集**：nuScenes v1.0-trainval
- **数据集路径**：`/media/ldk950413/data0/nuscenes`
- **类别数**：14个BEV语义类别

## 新增代码

### BEVSegmentor实现

**文件**：`mmseg/models/segmentors/bev_segmentor.py`

**功能**：
- 继承自`EncoderDecoder`
- 支持`transformer`参数（作为neck的别名）
- 自动从`img_metas`中提取`calib`信息并传递给transformer
- 兼容LSS transformer的calib需求

**关键方法**：
- `extract_feat(img, img_metas)`: 提取特征，支持calib传递
- `forward_train()`: 训练前向传播
- `encode_decode()`: 推理编码解码

## 使用方法

### 方法1：使用运行脚本（推荐）

```bash
# 1. 激活环境
micromamba activate diffbev

# 2. 运行baseline实验
bash run_baseline_nuscenes.sh
```

### 方法2：手动运行

```bash
# 1. 激活环境
micromamba activate diffbev

# 2. 进入项目目录
cd /media/ldk950413/data0/DiffBEV

# 3. 运行训练
python tools/train.py configs/baseline/lss_swin_nuscenes.py \
    --work-dir runs/baseline \
    --gpu-ids 0
```

## 输出目录结构

训练结果将保存在 `runs/baseline/` 目录下：

```
runs/baseline/
├── train_YYYYMMDD_HHMMSS.log          # 训练日志
├── configs/
│   └── lss_swin_nuscenes.py           # 保存的配置文件
├── iter_5000.pth                       # 每5k iterations的checkpoint
├── iter_10000.pth
├── ...
├── iter_200000.pth                    # 最终checkpoint
└── best_mIoU_iter_XXXXX.pth           # 最佳mIoU的checkpoint
```

## Baseline vs DiffBEV对比

| 特性 | Baseline (LSS) | DiffBEV (论文方法) |
|------|---------------|-------------------|
| View Transformer | LSS | PYVA + Diffusion |
| Diffusion模块 | ❌ 关闭 | ✅ 启用 |
| 损失函数 | Lwce | Lwce + 10*Ldepth + 1*Ldiff |
| 深度监督 | ❌ 关闭 | ✅ 启用 |
| 其他设置 | 与论文规范一致 | 与论文规范一致 |

## 验证Baseline配置

### 1. 检查配置是否正确

```bash
python tools/train.py configs/baseline/lss_swin_nuscenes.py \
    --work-dir runs/baseline_test \
    --gpu-ids 0 \
    --no-validate  # 仅检查配置，不实际训练
```

### 2. 检查模型结构

```python
from mmengine import Config
from mmseg.models import build_segmentor

cfg = Config.fromfile('configs/baseline/lss_swin_nuscenes.py')
model = build_segmentor(cfg.model)
print(model)
```

### 3. 验证损失函数

Baseline配置的损失函数只包含：
- `loss_seg`：分割损失（Lwce）
- `acc_seg`：IoU准确率

不包含：
- `loss_depth`：深度损失（已关闭）
- `loss_diff`：扩散损失（无diffusion模块）

## 注意事项

1. **预训练权重**：需要更新配置文件中的`pretrained`路径
2. **数据集路径**：已统一配置为`/media/ldk950413/data0/nuscenes`
3. **GPU内存**：batch_size=4, 分辨率800×600，确保GPU内存足够
4. **训练时间**：200k iterations可能需要较长时间，建议使用多GPU

## 与论文规范的对比

### 符合规范的部分
- ✅ 优化器：AdamW, lr=2e-4, weight_decay=0.01
- ✅ 训练：200k iterations, warmup 1500 iter
- ✅ 输入分辨率：800×600
- ✅ Batch size：4 per GPU
- ✅ 数据集：nuScenes v1.0-trainval

### Baseline特有的设置
- ❌ 不使用diffusion模块
- ❌ 不使用深度监督（Ldepth）
- ❌ 不使用扩散损失（Ldiff）
- ✅ 使用LSS transformer（baseline方法）

## 预期结果

Baseline实验将提供：
- **BEV语义分割性能**：mIoU和各类IoU
- **与DiffBEV方法的对比基准**
- **训练曲线和日志**：保存在`runs/baseline/`目录

## 故障排除

### 问题1：找不到BEVSegmentor

**解决方案**：确保已安装项目
```bash
pip install -v -e .
```

### 问题2：calib信息缺失

**解决方案**：检查数据pipeline中是否包含`with_calib=True`

### 问题3：transformer参数错误

**解决方案**：检查LSS transformer的配置参数是否正确

## 相关文件

- **配置文件**：`configs/baseline/lss_swin_nuscenes.py`
- **运行脚本**：`run_baseline_nuscenes.sh`
- **BEVSegmentor实现**：`mmseg/models/segmentors/bev_segmentor.py`
- **LSS Transformer**：`mmseg/models/necks/lift_splat_shoot_transformer.py`
- **PyramidHead**：`mmseg/models/decode_heads/pyramid_head.py`
