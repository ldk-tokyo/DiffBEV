# MMCV 2.x API迁移进展

## 当前状态

正在将代码从MMCV 1.x迁移到MMCV 2.x，以支持PyTorch 2.x和RTX 5090 GPU。

## 已修复的API

### 1. `revert_sync_batchnorm`
- **旧位置**: `mmcv.cnn.utils`
- **新位置**: `mmengine.model`
- **文件**: `tools/train.py`, `tests/test_models/test_forward.py`

### 2. `EvalHook` 和 `DistEvalHook`
- **旧位置**: `mmcv.runner`
- **新位置**: 已创建兼容实现（基于`mmengine.hooks.Hook`）
- **文件**: `mmseg/core/evaluation/eval_hooks.py`

### 3. `get_git_hash`
- **旧位置**: `mmcv.utils`
- **新位置**: `mmengine.utils` 或简化实现
- **文件**: `mmseg/utils/collect_env.py`, `tools/train.py`

### 4. `get_logger`
- **旧位置**: `mmcv.utils`
- **新位置**: 简化实现
- **文件**: `mmseg/utils/logger.py`

### 5. `DataContainer`
- **旧位置**: `mmcv.parallel`
- **新位置**: 兼容实现
- **文件**: `mmseg/datasets/pipelines/formatting.py`

### 6. `deprecated_api_warning` 和 `is_tuple_of`
- **旧位置**: `mmcv.utils`
- **新位置**: `mmengine.utils`（deprecated_api_warning）或简化实现
- **文件**: `mmseg/datasets/pipelines/transforms.py`

### 7. `_BatchNorm`
- **旧位置**: `mmcv.utils.parrots_wrapper`
- **新位置**: `torch.nn.modules.batchnorm`
- **文件**: `mmseg/models/backbones/cgnet.py`

### 8. MMCV版本检查
- **修改**: 允许MMCV 2.x（通过环境变量控制）
- **文件**: `mmseg/__init__.py`

## 已知问题和待修复

由于MMCV 2.x的API变化较大，可能还有其他文件需要修复。如果遇到新的导入错误，按照类似的模式进行修复：

1. 尝试从mmcv 1.x路径导入
2. 尝试从mmengine/mmcv 2.x路径导入
3. 如果都失败，创建兼容实现或使用PyTorch原生替代

## 使用方法

确保设置以下环境变量：
```bash
export DIFFBEV_ALLOW_PYTORCH2=1
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## 注意事项

1. **兼容实现**: 某些兼容实现可能不包含所有原始功能，建议在实际使用中验证功能完整性。

2. **逐步迁移**: 如果遇到新的导入错误，可以继续按照相同的模式进行修复。

3. **测试**: 修复后建议运行完整的测试套件以确保功能正常。
