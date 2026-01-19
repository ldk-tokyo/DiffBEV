# Loss结构自检系统说明

## 概述

Loss结构自检系统在训练启动时自动检查loss结构，验证权重配置，并监控前100次迭代的loss值。如果检测到loss为0、NaN或未参与反向传播，会直接抛出异常终止训练。

## 功能特性

### 1. Loss结构检查

在第一次训练迭代后，自动分析并打印loss结构：

```
Lseg = Lwce + 10 * Ldepth + 1 * Ldiff
```

或（对于baseline配置）：

```
Lseg = Lwce
```

### 2. 权重配置验证

自动从配置文件中读取并验证：
- `lambda_depth = 10.0`（深度损失权重）
- `lambda_diff = 1.0`（扩散损失权重）

### 3. Loss监控

在前100次迭代内，每10次迭代打印一次三项loss的均值：
- **Lwce**: 加权交叉熵损失的平均值
- **Ldepth**: 深度损失的平均值（如果启用）
- **Ldiff**: 扩散损失的平均值（如果启用）

### 4. 异常检测与终止

自动检测以下问题并终止训练：
- ✅ Loss为NaN
- ✅ Loss为0（Lwce不应该为0）
- ✅ Loss未参与反向传播（requires_grad=False）
- ✅ 梯度为NaN

## 输出示例

### 训练启动时的自检输出

```
================================================================================
🔍 Loss结构自检开始
================================================================================

📋 Decode Head配置检查:
   ⚠️  decode_head中没有loss_depth_weight属性
   ⚠️  decode_head中没有loss_diff_weight属性
   ⚠️  decode_head中没有use_diffusion属性

⚖️  Loss权重验证:
   预期 lambda_depth = 10.0
   预期 lambda_diff = 1.0
   ✓ 权重配置验证完成（将在首次迭代时确认实际使用值）
================================================================================

================================================================================
📊 Loss结构检查（基于第一次迭代的输出）:
================================================================================

   检测到的Loss键: ['loss_seg', 'acc_seg']

   ✅ Lseg = Lwce

   各项Loss说明:
   - Lwce (来自 'loss_seg'): 加权交叉熵损失

================================================================================
```

### 前100次迭代的监控输出

```
📈 Iter   10 - Loss统计 (前10次迭代的平均值):
   平均: Lwce=0.523456 | Ldepth=N/A (可能未启用) | Ldiff=N/A (可能未启用)

📈 Iter   20 - Loss统计 (前20次迭代的平均值):
   平均: Lwce=0.498765 | Ldepth=N/A (可能未启用) | Ldiff=N/A (可能未启用)

...

📈 Iter  100 - Loss统计 (前100次迭代的平均值):
   平均: Lwce=0.412345 | Ldepth=N/A (可能未启用) | Ldiff=N/A (可能未启用)
```

## 实现细节

### Hook注册

Loss检查Hook在训练启动时自动注册（`mmseg/apis/train.py`）：

```python
from mmseg.core.hooks.loss_check_hook import LossCheckHook

loss_check_hook = LossCheckHook(
    check_interval=10,      # 每10次迭代检查一次
    monitor_iters=100,      # 前100次迭代进行监控
    lambda_depth=10.0,      # 预期深度损失权重
    lambda_diff=1.0         # 预期扩散损失权重
)
runner.register_hook(loss_check_hook, priority='HIGHEST')
```

### Loss检测逻辑

#### 1. Loss结构识别

系统会从`log_vars`中自动识别：
- **Lwce**: 查找包含`loss_seg`或`loss_decode`的键
- **Ldepth**: 查找包含`loss_depth`的键
- **Ldiff**: 查找包含`loss_diff`或`loss_diffusion`的键

#### 2. 异常检测

在每次反向传播前后进行检查：

**反向传播前**：
- 检查loss是否为NaN
- 检查loss是否为0
- 检查loss是否参与计算图（`requires_grad=True`）

**反向传播后**：
- 检查梯度是否存在
- 检查梯度是否为NaN

### 权重配置来源

系统从以下位置读取权重配置：

1. **配置文件** (`configs/*/*.py`):
   ```python
   decode_head=dict(
       loss_depth_weight=10.0,  # lambda_depth
       loss_diff_weight=1.0,    # lambda_diff
   )
   ```

2. **模型实例** (运行时):
   ```python
   model.decode_head.loss_depth_weight
   model.decode_head.loss_diff_weight
   ```

## 异常处理

### 检测到NaN时的异常信息

```
❌ 训练终止: Iter 25 时检测到 Lwce 为 NaN！
这通常表示训练不稳定或数值溢出。请检查：
   1. 学习率是否过大
   2. 输入数据是否包含异常值
   3. 模型初始化是否正确
```

### 检测到Loss为0时的异常信息

```
❌ 训练终止: Iter 10 时检测到 Lwce 为 0！
Lwce不应该为0，这可能表示：
   1. Loss计算有误
   2. 模型输出异常
   3. 标签数据问题
   实际值: 0.0
```

### 检测到Loss未参与反向传播时的异常信息

```
❌ 训练终止: Iter 5 时检测到loss未参与反向传播！
loss.requires_grad = False。
请检查loss计算是否正确。
```

## 配置说明

### Baseline配置（无Diffusion）

对于baseline配置，系统会检测到：
- ✅ Lwce存在
- ⚠️  Ldepth不存在（正常，baseline不使用）
- ⚠️  Ldiff不存在（正常，baseline不使用）

输出：`Lseg = Lwce`

### DiffBEV配置（有Diffusion）

对于DiffBEV配置，系统会检测到：
- ✅ Lwce存在
- ✅ Ldepth存在（如果启用）
- ✅ Ldiff存在（如果启用）

输出：`Lseg = Lwce + 10 * Ldepth + 1 * Ldiff`

## 注意事项

1. **自动注册**: Loss检查Hook会在训练启动时自动注册，无需手动配置
2. **性能影响**: 检查操作非常轻量，不会显著影响训练速度
3. **早期检测**: 在前100次迭代中进行详细监控，能够早期发现问题
4. **自动终止**: 如果检测到严重问题（NaN、0、未参与反向传播），会立即终止训练，避免浪费计算资源

## 故障排查

### 问题：Loss结构检查未显示Ldepth或Ldiff

**可能原因**：
1. 模型配置中未启用depth/diffusion loss
2. Loss键名不匹配（系统会自动尝试多种键名）

**解决方案**：
- 检查模型配置中的`loss_depth_weight`和`loss_diff_weight`
- 检查模型输出的`log_vars`中实际的loss键名

### 问题：权重配置验证不通过

**可能原因**：
- 配置文件中的权重值与预期值（10.0和1.0）不匹配

**解决方案**：
- 检查配置文件中的`loss_depth_weight`和`loss_diff_weight`值
- 确保符合论文规范：`lambda_depth=10.0`, `lambda_diff=1.0`

### 问题：训练被异常终止

**可能原因**：
- Loss计算出现问题
- 模型输出异常
- 数据问题

**解决方案**：
- 查看异常信息，确定具体原因
- 检查模型配置和数据加载
- 降低学习率重试
