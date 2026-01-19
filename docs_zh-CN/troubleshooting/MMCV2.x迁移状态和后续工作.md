# MMCV 2.x迁移状态和后续工作

## 当前进展

已成功修复多个MMCV 2.x API变更问题，训练脚本可以进一步启动。但由于MMCV 2.x的API变化较大，可能还会遇到更多的导入错误。

## 已修复的API（10+个）

1. ✅ `revert_sync_batchnorm` - 从`mmengine.model`导入
2. ✅ `EvalHook`和`DistEvalHook` - 兼容实现
3. ✅ `get_git_hash` - 从`mmengine.utils`导入或简化实现
4. ✅ `get_logger` - 简化实现
5. ✅ `DataContainer` - 兼容实现
6. ✅ `deprecated_api_warning`和`is_tuple_of` - 兼容实现
7. ✅ `_BatchNorm` - 批量修复多个backbone文件
8. ✅ `to_2tuple` - 兼容实现
9. ✅ `constant_init` - 从`mmengine.model`导入或简化实现
10. ✅ MMCV版本检查 - 允许MMCV 2.x

## 修复模式

所有修复都采用相同的兼容性策略：

```python
# 模式1: 优先从mmcv 1.x导入
try:
    from mmcv.xxx import xxx
except ImportError:
    # 模式2: 从mmengine/mmcv 2.x导入
    try:
        from mmengine.xxx import xxx
    except ImportError:
        # 模式3: 创建兼容实现
        def xxx(...):
            # 简化实现
            pass
```

## 可能还需要修复的API

如果训练过程中遇到新的导入错误，可以按照以下步骤修复：

1. **查找错误**: 记录完整的错误信息，包括模块名和函数名
2. **检查位置**: 
   ```bash
   python -c "from mmengine.xxx import xxx"  # 尝试mmengine
   python -c "from mmcv.xxx import xxx"      # 尝试mmcv
   ```
3. **应用修复**: 按照上面的模式修改相应的文件
4. **测试**: 重新运行训练脚本

## 常见API映射

| MMCV 1.x | MMCV 2.x / MMEngine |
|----------|---------------------|
| `mmcv.runner.*` | `mmengine.runner.*` 或 `mmcv.runner.*`（部分保留）|
| `mmcv.utils.get_logger` | `mmengine.utils.get_logger` 或简化实现 |
| `mmcv.parallel.DataContainer` | 兼容实现 |
| `mmcv.cnn.utils.revert_sync_batchnorm` | `mmengine.model.revert_sync_batchnorm` |
| `mmcv.utils.parrots_wrapper._BatchNorm` | `torch.nn.modules.batchnorm._BatchNorm` |

## 使用建议

1. **环境变量**: 始终设置 `DIFFBEV_ALLOW_PYTORCH2=1`
2. **CUDA环境**: 确保设置正确的CUDA路径
3. **逐步修复**: 如果遇到新的导入错误，逐个修复
4. **测试验证**: 每个修复后测试是否能继续运行

## 当前状态

训练脚本已经可以加载大部分模块，但仍可能遇到新的导入错误。如果遇到新的错误，请：

1. 查看错误信息
2. 按照上述模式修复
3. 继续运行直到所有导入问题解决

## 成功标志

当训练脚本能够成功启动并开始加载数据时，说明API迁移基本完成。
