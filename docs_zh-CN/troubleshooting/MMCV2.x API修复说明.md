# MMCV 2.x API修复说明

## 问题

MMCV 2.1.0中，`revert_sync_batchnorm`函数的导入路径发生了变化。

**错误信息**:
```
ImportError: cannot import name 'revert_sync_batchnorm' from 'mmcv.cnn.utils'
```

## 解决方案

### API变化

**MMCV 1.x**:
```python
from mmcv.cnn.utils import revert_sync_batchnorm
```

**MMCV 2.x**:
```python
from mmengine.model import revert_sync_batchnorm
```

### 兼容性修复

已在`tools/train.py`中添加兼容性导入：

```python
# 尝试导入revert_sync_batchnorm（兼容MMCV 1.x和2.x）
try:
    from mmcv.cnn.utils import revert_sync_batchnorm
except ImportError:
    try:
        # MMCV 2.x可能在mmengine.model中
        from mmengine.model import revert_sync_batchnorm
    except ImportError:
        # 如果都失败，使用PyTorch内置实现
        def revert_sync_batchnorm(module):
            """将SyncBatchNorm转换为BatchNorm2d（兼容性函数）"""
            # ... 实现代码 ...
```

### 验证

修复后，可以通过以下方式验证：

```python
from tools.train import revert_sync_batchnorm
print('✓ revert_sync_batchnorm导入成功')
```

## 已修复的文件

1. `tools/train.py` - 训练脚本
2. `tests/test_models/test_forward.py` - 测试脚本

## 相关链接

- MMCV 2.x迁移指南: https://mmcv.readthedocs.io/en/2.x/migration.html
- MMEngine API文档: https://mmengine.readthedocs.io/en/latest/api/model.html
