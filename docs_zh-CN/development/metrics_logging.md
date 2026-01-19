# 指标记录系统说明

## 概述

本系统实现了统一的指标记录功能，支持将训练和评估指标同时写入**TensorBoard**和**CSV文件**。

## 功能特性

### 1. 自动记录分割指标
- **mIoU**: 平均IoU
- **各类别IoU**: 每个类别的IoU值（如 `IoU.vehicle`, `IoU.drivable_area` 等）

### 2. 自动记录检测指标（如果可用）
- **NDS**: NuScenes Detection Score
- **mAP**: mean Average Precision
- **ATE**: Average Translation Error
- **ASE**: Average Scale Error
- **AOE**: Average Orientation Error
- **AVE**: Average Velocity Error
- **AAE**: Average Attribute Error

### 3. 自动记录训练过程指标
- **Lwce**: 加权交叉熵损失（Weighted Cross-Entropy Loss）
- **Ldepth**: 深度损失（Depth Loss）
- **Ldiff**: 扩散损失（Diffusion Loss）
- **lr**: 学习率（Learning Rate）

## 输出文件

### CSV文件
- **位置**: `{work_dir}/metrics.csv`
- **格式**: 包含所有指标的CSV表格，每行代表一次记录
- **列**: step, iter, epoch, mIoU, IoU.{class_name}, NDS, mAP, Lwce, Ldepth, Ldiff, lr, 等

### TensorBoard日志
- **位置**: `{work_dir}/tf_logs/`
- **查看方式**: 
  ```bash
  tensorboard --logdir={work_dir}/tf_logs
  ```
- **组织方式**:
  - `train/`: 训练指标（Lwce, Ldepth, Ldiff, lr, loss等）
  - `eval/segmentation/`: 分割评估指标（mIoU, 各类别IoU）
  - `eval/detection/`: 检测评估指标（NDS, mAP, ATE等）

## 使用方式

### 自动记录

指标记录是**自动的**，无需额外配置。当训练或评估运行时，指标会自动记录：

1. **训练过程**: 每次记录日志时（默认每50次迭代）自动记录训练指标
2. **评估过程**: 每次评估时自动记录评估指标

### 手动使用MetricsLogger

如果需要手动记录指标，可以使用 `MetricsLogger` 类：

```python
from mmseg.utils.metrics_logger import MetricsLogger

# 初始化
metrics_logger = MetricsLogger(
    work_dir='./runs/experiment',
    csv_filename='metrics.csv',
    mode='eval'
)

# 记录分割指标
metrics_logger.log_segmentation_metrics(
    mIoU=0.75,
    class_IoU={'vehicle': 0.80, 'drivable_area': 0.70},
    step=1000
)

# 记录检测指标
metrics_logger.log_detection_metrics(
    NDS=0.45,
    mAP=0.40,
    ATE=0.25,
    step=1000
)

# 记录训练损失
metrics_logger.log_training_losses(
    Lwce=0.5,
    Ldepth=0.1,
    Ldiff=0.05,
    learning_rate=2e-4,
    step=1000
)

# 记录自定义指标
metrics_logger.log({
    'custom_metric1': 0.8,
    'custom_metric2': 0.6
}, step=1000, prefix='custom')

# 关闭
metrics_logger.close()
```

## 实现细节

### 文件结构

```
mmseg/
├── utils/
│   ├── metrics_logger.py      # MetricsLogger类实现
│   └── runner_compat.py        # 训练过程中的指标记录
├── core/
│   └── evaluation/
│       └── eval_hooks.py       # 评估过程中的指标记录
└── datasets/
    └── nuscenes.py             # 返回各类别IoU到eval_results
```

### 指标提取逻辑

#### 训练指标提取
代码会自动从 `log_vars` 中提取以下指标：
- `loss_seg` 或 `loss_decode.loss_seg` → Lwce
- `loss_depth` → Ldepth
- `loss_diff` 或 `loss_diffusion` → Ldiff
- `loss` → 总损失
- 从优化器或 `log_vars['lr']` / `log_vars['learning_rate']` → 学习率

#### 评估指标提取
- 从 `eval_results` 中提取 `mIoU` 和 `IoU.{class_name}` 格式的指标
- 从 `eval_results` 中提取检测相关指标（如果存在）

## 注意事项

1. **TensorBoard依赖**: 如果TensorBoard不可用，系统会自动降级为仅记录CSV
2. **文件覆盖**: CSV文件会追加写入，不会覆盖历史记录
3. **性能影响**: 记录操作很轻量，不会显著影响训练速度
4. **错误处理**: 如果记录失败，会记录警告但不会中断训练/评估

## 查看结果

### 查看CSV文件
```bash
# 使用pandas
python -c "import pandas as pd; df = pd.read_csv('runs/baseline/metrics.csv'); print(df.tail())"

# 使用Excel/LibreOffice打开
libreoffice runs/baseline/metrics.csv
```

### 查看TensorBoard
```bash
# 启动TensorBoard
tensorboard --logdir=runs/baseline/tf_logs --port=6006

# 然后在浏览器中打开 http://localhost:6006
```

## 示例输出

### CSV文件示例
```csv
step,iter,epoch,mIoU,IoU.vehicle,IoU.drivable_area,Lwce,Ldepth,Ldiff,lr
100,100,0,0.65,0.70,0.60,0.5,0.1,0.05,0.0002
200,200,0,0.68,0.72,0.64,0.48,0.09,0.04,0.0002
```

### TensorBoard标签组织
- `train/Lwce`
- `train/Ldepth`
- `train/Ldiff`
- `train/lr`
- `eval/segmentation/mIoU`
- `eval/segmentation/IoU.vehicle`
- `eval/detection/NDS`
- `eval/detection/mAP`

## 故障排查

### 问题：CSV文件没有创建
- 检查 `work_dir` 路径是否正确
- 检查是否有写入权限

### 问题：TensorBoard中没有数据
- 确认TensorBoard已安装：`pip install tensorboard`
- 检查 `tf_logs` 目录是否存在
- 确认指标记录没有被错误捕获

### 问题：某些指标没有记录
- 检查模型输出的 `log_vars` 中是否包含相应的键
- 检查评估结果的 `eval_results` 中是否包含相应的键
- 查看日志中的警告信息
