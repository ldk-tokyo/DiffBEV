# Baseline实验总结

## 已完成的工作

### 1. Baseline配置文件

**文件**：`configs/baseline/lss_swin_nuscenes.py`

**特点**：
- ✅ 使用LSS (Lift-Splat-Shoot) Transformer作为视图变换方法
- ✅ 完全关闭diffusion模块（无diffusion相关代码）
- ✅ 关闭深度监督（`depthsup=False`, `outdepth=False`）
- ✅ 损失函数只包含Lwce（加权交叉熵），无Ldepth和Ldiff
- ✅ 其他训练设置与论文规范完全一致

### 2. BEVSegmentor实现

**文件**：`mmseg/models/segmentors/bev_segmentor.py`

**功能**：
- 继承自`EncoderDecoder`
- 支持`transformer`参数（作为neck使用）
- 自动从`img_metas`中提取`calib`信息
- 智能检测transformer是否需要calib参数
- 兼容LSS transformer的`intrinstics`参数需求

### 3. 运行脚本

**文件**：`run_baseline_nuscenes.sh`

**功能**：
- 一键启动baseline实验
- 自动检查环境（diffbev）
- 自动创建输出目录（`runs/baseline`）
- 记录训练日志
- 显示GPU信息
- 提供训练结果查看指引

### 4. 文档

- **Baseline配置说明.md**：详细的配置和使用说明
- **Baseline实验总结.md**：本文档

## 配置对比

### Baseline配置 vs DiffBEV配置

| 项目 | Baseline (LSS) | DiffBEV (PYVA) |
|------|----------------|----------------|
| **View Transformer** | `TransformerLiftSplatShoot` | `v4_Pyva_transformer` |
| **Segmentor类型** | `BEVSegmentor` | `new_pyva_BEVSegmentor` |
| **Diffusion模块** | ❌ 无 | ✅ 有（待实现） |
| **深度监督** | ❌ 关闭 | ✅ 启用 |
| **损失函数** | Lwce | Lwce + 10*Ldepth + 1*Ldiff |
| **其他设置** | 与论文规范一致 | 与论文规范一致 |

## 使用方法

### 快速启动

```bash
# 1. 激活环境
micromamba activate diffbev

# 2. 运行baseline实验
bash run_baseline_nuscenes.sh
```

### 手动运行

```bash
python tools/train.py configs/baseline/lss_swin_nuscenes.py \
    --work-dir runs/baseline \
    --gpu-ids 0
```

## 输出结构

```
runs/baseline/
├── train_YYYYMMDD_HHMMSS.log      # 训练日志
├── configs/
│   └── lss_swin_nuscenes.py       # 保存的配置
├── iter_5000.pth                  # Checkpoints
├── iter_10000.pth
├── ...
├── iter_200000.pth                # 最终checkpoint
└── best_mIoU_iter_XXXXX.pth       # 最佳模型
```

## 验证清单

在运行前，请确认：

- [ ] 已激活 `diffbev` 环境
- [ ] 数据集路径正确：`/media/ldk950413/data0/nuscenes`
- [ ] 数据集关键目录存在：`samples/`, `sweeps/`, `v1.0-trainval/`
- [ ] 预训练权重路径已更新（`pretrained`字段）
- [ ] GPU内存足够（batch_size=4, 分辨率800×600）

## 预期结果

Baseline实验将提供：
- **BEV语义分割性能**：mIoU和各类IoU指标
- **训练曲线**：损失和准确率变化
- **与DiffBEV的对比基准**：用于评估diffusion模块的贡献

## 下一步

1. **运行baseline实验**：使用提供的脚本启动训练
2. **记录结果**：保存训练曲线和评估指标
3. **实现DiffBEV**：添加diffusion模块和相关loss
4. **对比分析**：比较baseline和DiffBEV的性能差异
