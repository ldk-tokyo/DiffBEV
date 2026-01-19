# MMCV 升级到 2.x 完整指南

## 概述

本指南将帮助你将项目从 `mmcv-full 1.3.x` 升级到 `mmcv 2.x`，以支持 PyTorch 2.x 和 RTX 5090。

## 重要变化

### MMCV 2.x 主要变化

1. **mmcv.runner 迁移到 mmengine**
   - `mmcv.runner.BaseModule` → `mmengine.model.BaseModule`
   - `mmcv.runner.load_checkpoint` → `mmengine.runner.CheckpointLoader.load_checkpoint`
   - `mmcv.runner.init_dist` → `mmengine.runner.init_dist`
   - `mmcv.runner.get_dist_info` → `mmengine.runner.get_dist_info`

2. **mmcv.parallel 迁移到 mmengine**
   - `mmcv.parallel.MMDataParallel` → `mmengine.parallel.MMDataParallel`
   - `mmcv.parallel.MMDistributedDataParallel` → `mmengine.parallel.MMDistributedDataParallel`
   - `mmcv.parallel.collate` → `mmengine.device.auto_fp16` 或保持使用

3. **mmcv.utils 部分变化**
   - `mmcv.Config` → `mmengine.Config`
   - `mmcv.ConfigDict` → `mmengine.ConfigDict`
   - `mmcv.build_from_cfg` → `mmengine.registry.build_from_cfg`

4. **其他变化**
   - `mmcv.cnn` 大部分API保持不变
   - `mmcv.ops` 保持不变
   - `mmcv.utils.print_log` → `mmengine.logging.print_log`

## 升级步骤

### 步骤1：安装新的依赖

```bash
# 1. 激活环境
micromamba activate diffbev

# 2. 卸载旧版本
pip uninstall mmcv mmcv-full -y

# 3. 安装mmcv 2.x（支持PyTorch 2.x）
# 方法A：使用mim（推荐）
pip install -U openmim
mim install mmcv>=2.1.0

# 方法B：使用pip指定版本
pip install mmcv>=2.1.0

# 4. 安装mmengine（必需，因为runner等迁移到这里）
pip install mmengine>=0.10.0

# 5. 验证安装
python -c "import mmcv; import mmengine; print(f'MMCV: {mmcv.__version__}, MMEngine: {mmengine.__version__}')"
```

### 步骤2：更新兼容性检查

已更新 `mmseg/utils/compat.py`，支持检测 mmcv 2.x。

### 步骤3：API迁移

需要修改的主要API调用：

#### 3.1 mmcv.runner → mmengine

**文件列表：**
- `tools/train.py`
- `tools/test.py`
- `mmseg/apis/train.py`
- `mmseg/apis/inference.py`
- `mmseg/models/**/*.py` (所有使用BaseModule的文件)
- `mmseg/core/evaluation/eval_hooks.py`

**替换示例：**
```python
# 旧代码 (mmcv 1.x)
from mmcv.runner import BaseModule, load_checkpoint, init_dist, get_dist_info

# 新代码 (mmcv 2.x)
from mmengine.model import BaseModule
from mmengine.runner import load_checkpoint, init_dist, get_dist_info
```

#### 3.2 mmcv.parallel → mmengine

**文件列表：**
- `tools/test.py`
- `tools/train.py`
- `mmseg/apis/inference.py`

**替换示例：**
```python
# 旧代码
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

# 新代码
from mmengine.parallel import MMDataParallel, MMDistributedDataParallel
```

#### 3.3 mmcv.Config → mmengine.Config

**文件列表：**
- `tools/train.py`
- `tools/test.py`
- `mmseg/apis/inference.py`
- 所有使用 `mmcv.Config.fromfile()` 的文件

**替换示例：**
```python
# 旧代码
import mmcv
cfg = mmcv.Config.fromfile('config.py')

# 新代码
from mmengine import Config
cfg = Config.fromfile('config.py')
```

#### 3.4 mmcv.utils → mmengine

**文件列表：**
- `mmseg/datasets/**/*.py`
- `mmseg/core/seg/builder.py`
- `mmseg/datasets/builder.py`

**替换示例：**
```python
# 旧代码
from mmcv.utils import print_log, Registry, build_from_cfg, ConfigDict

# 新代码
from mmengine.logging import print_log
from mmengine.registry import Registry, build_from_cfg
from mmengine import ConfigDict
```

### 步骤4：更新版本检查

在 `mmseg/__init__.py` 中，更新MMCV版本检查：

```python
MMCV_MIN = '2.1.0'  # 从 '1.3.13' 改为 '2.1.0'
MMCV_MAX = '3.0.0'  # 从 '1.4.0' 改为 '3.0.0'
```

### 步骤5：处理特殊情况

#### 5.1 mmcv.cnn.utils.sync_bn

```python
# 旧代码
from mmcv.cnn.utils.sync_bn import revert_sync_batchnorm

# 新代码（可能需要检查）
from mmcv.cnn.utils import revert_sync_batchnorm
# 或
from mmengine.model import revert_sync_batchnorm
```

#### 5.2 其他可能需要检查的模块

- `mmcv.utils.parrots_wrapper._BatchNorm` → 可能需要替换为 `torch.nn.BatchNorm2d`
- `mmcv.cnn.bricks` → 基本保持不变
- `mmcv.ops` → 基本保持不变

## 自动迁移脚本

我将在下一步创建自动迁移脚本，批量替换这些API调用。

## 验证步骤

升级后，运行以下验证：

```bash
# 1. 检查版本
python -c "import mmcv; import mmengine; print(f'MMCV: {mmcv.__version__}'); print(f'MMEngine: {mmengine.__version__}')"

# 2. 测试导入
python -c "from mmseg.models import build_segmentor; print('导入成功')"

# 3. 测试配置加载
python -c "from mmengine import Config; cfg = Config.fromfile('configs/_base_/default_runtime.py'); print('配置加载成功')"

# 4. 运行训练（小规模测试）
python tools/train.py configs/pyva/pyva_swin_nuscenes.py --work-dir work_dirs/test_mmcv2 --gpus 1
```

## 可能遇到的问题

### 问题1：ImportError: cannot import name 'XXX' from 'mmcv'

**解决方案：** 检查该API是否已迁移到 `mmengine`，参考上面的替换表。

### 问题2：API参数变化

**解决方案：** 查看 `mmengine` 文档，某些API的参数可能有变化。

### 问题3：运行时错误

**解决方案：**
1. 检查是否所有API都已迁移
2. 查看错误堆栈，定位问题文件
3. 参考 mmcv 2.x 迁移文档

## 回退方案

如果升级后遇到无法解决的问题，可以回退：

```bash
# 卸载新版本
pip uninstall mmcv mmengine -y

# 重新安装旧版本
pip install mmcv-full==1.3.15 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

## 相关文档

- [MMCV 2.x 迁移指南](https://mmcv.readthedocs.io/en/2.x/migration.html)
- [MMEngine 文档](https://mmengine.readthedocs.io/)
- [OpenMMLab 迁移指南](https://mmengine.readthedocs.io/en/latest/migration/)

## 下一步

执行自动迁移脚本来批量替换API调用。
