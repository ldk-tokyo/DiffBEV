# DiffBEV训练流程Runbook

## 前提条件

- **仓库路径**: `/media/ldk950413/data0/DiffBEV`
- **nuScenes数据路径**: `/media/ldk950413/data0/nuScenes` (注意：配置文件使用大写S)
- **训练设备**: 单GPU (GPU 0)
- **训练迭代数**: 200k iterations (正式训练), 300 iterations (冒烟测试)

---

## 阶段0：进入目录与激活环境

### Step 0.1: 进入项目目录

```bash
cd /media/ldk950413/data0/DiffBEV
```

**成功判据**: 当前目录为 `/media/ldk950413/data0/DiffBEV`

### Step 0.2: 激活conda环境

```bash
conda activate diffbev_py310
```

或者使用micromamba:

```bash
micromamba activate diffbev_py310
```

**成功判据**: 命令提示符显示 `(diffbev_py310)` 或类似的环境名称

### Step 0.3: 设置环境变量

```bash
export DIFFBEV_ALLOW_PYTORCH2=1
```

**成功判据**: 执行后无错误输出

---

## 阶段1：数据路径与目录结构检查

### Step 1.1: 检查nuScenes数据目录是否存在

```bash
ls -ld /media/ldk950413/data0/nuScenes
```

**成功判据**: 显示目录信息，无 "No such file or directory" 错误

**注意**: 如果目录不存在，可能是大小写问题，尝试 `/media/ldk950413/data0/nuscenes`（小写s）

### Step 1.2: 检查数据子目录结构

```bash
ls -d \
    /media/ldk950413/data0/nuScenes/img_dir/train \
    /media/ldk950413/data0/nuScenes/img_dir/val \
    /media/ldk950413/data0/nuScenes/ann_bev_dir/train \
    /media/ldk950413/data0/nuScenes/ann_bev_dir/val
```

**成功判据**: 所有4个目录都存在

### Step 1.3: 检查深度GT目录结构（DiffBEV需要）

```bash
ls -d \
    /media/ldk950413/data0/nuScenes/ann_bev_dir/train_depth \
    /media/ldk950413/data0/nuScenes/ann_bev_dir/val_depth
```

**成功判据**: 两个目录都存在（如果不存在，按阶段4.0生成）

### Step 1.4: 检查训练脚本是否存在

```bash
test -f /media/ldk950413/data0/DiffBEV/scripts/training/run_baseline_nuscenes.sh && \
test -f /media/ldk950413/data0/DiffBEV/scripts/training/run_diffbev_nuscenes.sh && \
echo "Scripts exist"
```

**成功判据**: 输出 "Scripts exist"

### Step 1.5: 检查配置文件是否存在

```bash
test -f /media/ldk950413/data0/DiffBEV/configs/baseline/lss_swin_nuscenes.py && \
test -f /media/ldk950413/data0/DiffBEV/configs/diffbev/diffbev_lss_swin_nuscenes.py && \
echo "Configs exist"
```

**成功判据**: 输出 "Configs exist"

### Step 1.6: 检查GPU可用性

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | head -1
```

**成功判据**: 显示GPU信息（索引、名称、总内存、空闲内存）

---

## 阶段2：Baseline冒烟测试（300 iterations）

### Step 2.1: 创建冒烟测试配置文件

```bash
cp /media/ldk950413/data0/DiffBEV/configs/baseline/lss_swin_nuscenes.py /media/ldk950413/data0/DiffBEV/configs/baseline/lss_swin_nuscenes_smoke.py
```

**成功判据**: 文件复制成功，无错误

### Step 2.2: 修改冒烟测试配置文件的训练迭代数和checkpoint间隔

```bash
cat >> /media/ldk950413/data0/DiffBEV/configs/baseline/lss_swin_nuscenes_smoke.py << 'EOF'
# 冒烟测试：覆盖runner配置，将max_iters设置为300
runner = dict(type='IterBasedRunner', max_iters=300)
# 冒烟测试：覆盖checkpoint_config，设置更小的interval以便保存checkpoint
checkpoint_config = dict(by_epoch=False, interval=100)
EOF
```

**成功判据**: 执行后无错误输出

### Step 2.3: 验证修改成功

```bash
grep "max_iters=300" /media/ldk950413/data0/DiffBEV/configs/baseline/lss_swin_nuscenes_smoke.py
```

**成功判据**: 输出包含 `max_iters=300`

### Step 2.4: 运行Baseline冒烟测试

```bash
cd /media/ldk950413/data0/DiffBEV
export DIFFBEV_ALLOW_PYTORCH2=1
python tools/train.py configs/baseline/lss_swin_nuscenes_smoke.py --work-dir runs/baseline_smoke --gpu-ids 0 2>&1 | tee runs/baseline_smoke/smoke_test.log
```

**成功判据**: 
- 训练开始，看到loss输出
- 日志文件 `runs/baseline_smoke/smoke_test.log` 生成
- 训练完成时退出码为0

### Step 2.5: 检查冒烟测试输出

```bash
# 检查checkpoint文件
if ls /media/ldk950413/data0/DiffBEV/runs/baseline_smoke/*.pth 1>/dev/null 2>&1; then
    echo "✓ 找到checkpoint文件:"
    ls -lh /media/ldk950413/data0/DiffBEV/runs/baseline_smoke/*.pth
else
    echo "⚠️  未找到checkpoint文件（使用修复后的配置会在iter 100/200/300时保存）"
fi

# 检查训练是否完成
echo "训练进度检查:"
tail -1 /media/ldk950413/data0/DiffBEV/runs/baseline_smoke/metrics.csv
```

**成功判据**: 
- checkpoint文件：如果使用修复后的配置（interval=100），应该显示至少一个 `.pth` 文件；如果使用旧配置（interval=5000），不会显示文件（这是正常的）
- metrics.csv最后一行应该显示 `300,xxx,xxx`，表示训练已完成300次迭代

### Step 2.6: 检查metrics.csv是否生成

```bash
test -f /media/ldk950413/data0/DiffBEV/runs/baseline_smoke/metrics.csv && \
wc -l /media/ldk950413/data0/DiffBEV/runs/baseline_smoke/metrics.csv
```

**成功判据**: 文件存在且行数大于1（包含表头）

---

## 阶段3：Baseline正式训练

### Step 3.1: 确保输出目录存在

```bash
mkdir -p /media/ldk950413/data0/DiffBEV/runs/baseline
```

**成功判据**: 目录创建成功，无错误

### Step 3.2: 运行Baseline正式训练

```bash
cd /media/ldk950413/data0/DiffBEV
export DIFFBEV_ALLOW_PYTORCH2=1
bash scripts/training/run_baseline_nuscenes.sh
```

**成功判据**: 
- 脚本开始执行，显示配置信息
- 训练开始，看到loss输出
- 训练完成后显示 "✓ 训练完成！"

### Step 3.3: 检查训练日志文件

```bash
ls -lht /media/ldk950413/data0/DiffBEV/runs/baseline/train_*.log | head -1
```

**成功判据**: 显示最新的日志文件

### Step 3.4: 检查checkpoint文件

```bash
ls -lh /media/ldk950413/data0/DiffBEV/runs/baseline/*.pth
```

**成功判据**: 显示多个checkpoint文件，包括 `best_mIoU.pth`

### Step 3.5: 检查metrics.csv文件

```bash
test -f /media/ldk950413/data0/DiffBEV/runs/baseline/metrics.csv && \
tail -5 /media/ldk950413/data0/DiffBEV/runs/baseline/metrics.csv
```

**成功判据**: 文件存在且显示最后5行数据（包含step、mIoU等指标）

---

## 阶段4：DiffBEV正式训练

### Step 4.0: 生成离线深度GT（DiffBEV必需）

```bash
python /media/ldk950413/data0/DiffBEV/tools/data/prepare_nuscenes_depth_gt.py \
  --nuscenes-root /media/ldk950413/data0/nuScenes \
  --version v1.0-trainval \
  --output-dir /media/ldk950413/data0/nuScenes/ann_bev_dir \
  --split train \
  --img-dir /media/ldk950413/data0/nuScenes/img_dir/train \
  --cameras CAM_FRONT \
  --height 600 --width 800 \
  --min-depth 1 --max-depth 50
```

```bash
python /media/ldk950413/data0/DiffBEV/tools/data/prepare_nuscenes_depth_gt.py \
  --nuscenes-root /media/ldk950413/data0/nuScenes \
  --version v1.0-trainval \
  --output-dir /media/ldk950413/data0/nuScenes/ann_bev_dir \
  --split val \
  --img-dir /media/ldk950413/data0/nuScenes/img_dir/val \
  --cameras CAM_FRONT \
  --height 600 --width 800 \
  --min-depth 1 --max-depth 50
```

**成功判据**: 
- 生成目录 `/media/ldk950413/data0/nuScenes/ann_bev_dir/train_depth` 和 `val_depth`
- 目录内存在 `.npz` 文件（每个样本一个）

### Step 4.1: 确保输出目录存在

```bash
mkdir -p /media/ldk950413/data0/DiffBEV/runs/diffbev_default
```

**成功判据**: 目录创建成功，无错误

### Step 4.1.1: 深度GT自检（确保dataloader输出depth字段）

```bash
python /media/ldk950413/data0/DiffBEV/tools/debug/check_depth_gt.py \
  --config /media/ldk950413/data0/DiffBEV/configs/diffbev/diffbev_lss_swin_nuscenes.py
```

**成功判据**: 
- 输出包含 `gt_depth` 和 `gt_depth_mask`
- `gt_depth` 形状为 `[1, 1, H, W]`（H/W与训练输入一致）
- `gt_depth_mask valid_ratio` > 0

### Step 4.2: 运行DiffBEV正式训练

```bash
cd /media/ldk950413/data0/DiffBEV
export DIFFBEV_ALLOW_PYTORCH2=1
bash scripts/training/run_diffbev_nuscenes.sh
```

**成功判据**: 
- 脚本开始执行，显示DiffBEV配置信息
- 训练开始，看到loss输出（包括Lwce、Ldepth、Ldiff）
- 训练完成后显示 "✓ 训练完成！"

### Step 4.3: 检查训练日志文件

```bash
ls -lht /media/ldk950413/data0/DiffBEV/runs/diffbev_default/train_*.log | head -1
```

**成功判据**: 显示最新的日志文件

### Step 4.4: 检查checkpoint文件

```bash
ls -lh /media/ldk950413/data0/DiffBEV/runs/diffbev_default/*.pth
```

**成功判据**: 显示多个checkpoint文件，包括 `best_mIoU.pth`

### Step 4.5: 检查metrics.csv文件

```bash
test -f /media/ldk950413/data0/DiffBEV/runs/diffbev_default/metrics.csv && \
tail -5 /media/ldk950413/data0/DiffBEV/runs/diffbev_default/metrics.csv
```

**成功判据**: 文件存在且显示最后5行数据（包含`mode/phase/step`以及`loss/Lwce/Ldepth/Ldiff/lr/mIoU`等）

---

## 阶段5：汇总出图（plot_metrics）

### Step 5.1: 创建reports输出目录

```bash
mkdir -p /media/ldk950413/data0/DiffBEV/reports/comparison/plots
```

**成功判据**: 目录创建成功

### Step 5.2: 运行指标对比可视化

```bash
cd /media/ldk950413/data0/DiffBEV
python tools/plot_metrics.py \
    /media/ldk950413/data0/DiffBEV/runs/baseline \
    /media/ldk950413/data0/DiffBEV/runs/diffbev_default \
    --output-dir /media/ldk950413/data0/DiffBEV/reports/comparison \
    --experiment-names "Baseline" "DiffBEV"
```

**成功判据**: 
- 脚本执行，显示 "Loading metrics from..." 信息
- 显示 "Generating plots..." 和 "✓ Generated: ..." 信息
- 最后显示 "✓ All plots and summary generated"

### Step 5.3: 检查生成的图像文件

```bash
ls -lh /media/ldk950413/data0/DiffBEV/reports/comparison/plots/*.png
```

**成功判据**: 显示多个PNG文件（至少包含mIoU_comparison.png, Lwce_comparison.png, Ldepth_comparison.png, Ldiff_comparison.png, lr_comparison.png）

### Step 5.4: 检查生成的summary.md

```bash
test -f /media/ldk950413/data0/DiffBEV/reports/comparison/summary.md && \
head -20 /media/ldk950413/data0/DiffBEV/reports/comparison/summary.md
```

**成功判据**: 文件存在且包含实验名称、最佳指标表格

---

## 阶段6：导出可视化对比图（vis_nuscenes_bev）

### Step 6.1: 确定要可视化的样本索引

```bash
echo "0 10 20 30 40 50"
```

**成功判据**: 输出样本索引列表

### Step 6.2: 检查baseline和diffbev的checkpoint文件

```bash
test -f /media/ldk950413/data0/DiffBEV/runs/baseline/best_mIoU.pth && \
test -f /media/ldk950413/data0/DiffBEV/runs/diffbev_default/best_mIoU.pth && \
echo "Both checkpoints exist"
```

**成功判据**: 输出 "Both checkpoints exist"

### Step 6.3: 运行可视化对比脚本

```bash
cd /media/ldk950413/data0/DiffBEV
python tools/vis_nuscenes_bev.py \
    /media/ldk950413/data0/DiffBEV/configs/baseline/lss_swin_nuscenes.py \
    /media/ldk950413/data0/DiffBEV/runs/baseline/best_mIoU.pth \
    --diffbev-config /media/ldk950413/data0/DiffBEV/configs/diffbev/diffbev_lss_swin_nuscenes.py \
    --diffbev-checkpoint /media/ldk950413/data0/DiffBEV/runs/diffbev_default/best_mIoU.pth \
    --indices 0 10 20 30 40 50 \
    --work-dir /media/ldk950413/data0/DiffBEV/runs/vis_comparison \
    --device cuda:0
```

**成功判据**: 
- 脚本执行，显示 "Loading dataset..." 和 "Loading model..." 信息
- 显示 "Processing sample X..." 和 "✓ Saved to ..." 信息
- 最后显示 "Visualization complete!"

### Step 6.4: 检查生成的可视化图像

```bash
ls -lh /media/ldk950413/data0/DiffBEV/runs/vis_comparison/vis/*.png
```

**成功判据**: 显示6个PNG文件（sample_0000.png, sample_0010.png, sample_0020.png, sample_0030.png, sample_0040.png, sample_0050.png）

---

## 常见失败与快速定位

### 问题1: 训练过程中出现CUDA out of memory错误

**定位命令**:
```bash
nvidia-smi
```

**说明**: 查看GPU内存使用情况，确认是否有其他进程占用内存

**解决方案**: 
- 降低batch size（修改config中的 `samples_per_gpu`）
- 或使用 `CUDA_VISIBLE_DEVICES=1` 切换到其他GPU

---

### 问题2: 训练脚本执行失败，无法找到配置文件

**定位命令**:
```bash
ls -la /media/ldk950413/data0/DiffBEV/configs/baseline/lss_swin_nuscenes.py /media/ldk950413/data0/DiffBEV/configs/diffbev/diffbev_lss_swin_nuscenes.py
```

**说明**: 检查配置文件是否存在，确认路径正确

**解决方案**: 确认当前目录为 `/media/ldk950413/data0/DiffBEV`，或使用绝对路径

---

### 问题3: 数据加载失败，找不到数据文件

**定位命令**:
```bash
ls -d /media/ldk950413/data0/nuScenes/img_dir/train /media/ldk950413/data0/nuScenes/ann_bev_dir/train | wc -l
```

**说明**: 检查数据目录是否存在，应该输出2

**解决方案**: 
- 确认数据路径为 `/media/ldk950413/data0/nuScenes`（配置文件使用大写S）
- 如果实际路径是小写，需要修改config文件或创建符号链接
- 检查config文件 `/media/ldk950413/data0/DiffBEV/configs/_base_/datasets/nuscene.py` 中的 `data_root` 设置

---

### 问题4: metrics.csv文件未生成或为空

**定位命令**:
```bash
tail -20 /media/ldk950413/data0/DiffBEV/runs/baseline/train_*.log | grep -i "metrics\|loss\|mIoU"
```

**说明**: 检查训练日志中是否有指标记录相关的错误信息

**解决方案**: 
- 确认训练至少完成一次评估（20k iterations）以写入mIoU
- 训练loss应每`log_config.interval`写入CSV；检查`mmseg/core/hooks/metrics_logger_hook.py`
- 检查 `mmseg/utils/metrics_logger.py` 是否正确集成
- 如果CSV列缺失，重新启动训练以生成新表头

---

### 问题5: 可视化脚本失败，无法加载模型

**定位命令**:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

**说明**: 检查CUDA和GPU是否可用

**解决方案**: 
- 确认conda环境已激活
- 确认GPU驱动和CUDA版本正确
- 使用 `--device cpu` 进行CPU推理测试

---

### 问题6: mIoU长期停留在0.046左右（建议使用诊断工具）

**定位命令（只做统计，不保存图）**:
```bash
python /media/ldk950413/data0/DiffBEV/tools/debug/diagnose_segmentation.py \
  --config /media/ldk950413/data0/DiffBEV/configs/diffbev/diffbev_lss_swin_nuscenes.py \
  --checkpoint /media/ldk950413/data0/DiffBEV/runs/diffbev_default/latest.pth \
  --num-samples 50 \
  --vis-num 0 \
  --out-dir /media/ldk950413/data0/DiffBEV/runs/diagnose \
  --device cuda:0 \
  --seed 42
```

**定位命令（统计+保存8个样本可视化）**:
```bash
python /media/ldk950413/data0/DiffBEV/tools/debug/diagnose_segmentation.py \
  --config /media/ldk950413/data0/DiffBEV/configs/diffbev/diffbev_lss_swin_nuscenes.py \
  --checkpoint /media/ldk950413/data0/DiffBEV/runs/diffbev_default/latest.pth \
  --num-samples 50 \
  --vis-num 8 \
  --out-dir /media/ldk950413/data0/DiffBEV/runs/diagnose \
  --device cuda:0 \
  --seed 42
```

**说明**: 诊断工具会输出GT/Pred分布、per-class IoU表、mIoU与overall acc，并在`summary.txt`保存结果
**注意**: 当前val样本文件名不包含相机名时，请不要使用`--camera CAM_FRONT`，否则会跳过全部样本

**解决方案**:
- 若Pred分布塌缩（单一类别占比>95%），优先检查loss权重与数据标签
- 若GT分布异常或存在越界label，检查标注文件与pipeline的`reduce_zero_label/ignore_index`
- 若per-class IoU长期为0，优先检查模型输出是否与GT格式对齐

## 最终输出检查清单

训练完成后，应存在以下文件和目录：

```bash
echo "=== Baseline输出 ==="
ls -lh /media/ldk950413/data0/DiffBEV/runs/baseline/best_mIoU.pth
ls -lh /media/ldk950413/data0/DiffBEV/runs/baseline/metrics.csv
echo "=== DiffBEV输出 ==="
ls -lh /media/ldk950413/data0/DiffBEV/runs/diffbev_default/best_mIoU.pth
ls -lh /media/ldk950413/data0/DiffBEV/runs/diffbev_default/metrics.csv
echo "=== 对比图表 ==="
ls -lh /media/ldk950413/data0/DiffBEV/reports/comparison/plots/*.png | wc -l
echo "=== 可视化对比图 ==="
ls -lh /media/ldk950413/data0/DiffBEV/runs/vis_comparison/vis/*.png | wc -l
```

**成功判据**: 
- 显示baseline和diffbev的best_mIoU.pth文件
- 显示两个metrics.csv文件
- 显示7个对比图表PNG文件
- 显示6个可视化对比PNG文件
