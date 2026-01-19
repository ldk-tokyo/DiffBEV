# NumPy 兼容性修复

## 问题描述

在评估阶段出现错误：
```
AttributeError: module 'numpy' has no attribute 'bool'.
`np.bool` was a deprecated alias for the builtin `bool`.
```

## 原因

NumPy 1.20+ 版本中，`np.bool` 已被弃用。在 NumPy 1.24+ 版本中，`np.bool` 被完全移除。

## 修复内容

将所有 `np.bool` 替换为 `bool`：

### 修复的文件

1. **`mmseg/datasets/nuscenes.py`**
   - 第186行：`dtype=np.bool` → `dtype=bool`
   - 第206行：`dtype=np.bool` → `dtype=bool`

2. **`mmseg/datasets/pipelines/loading.py`**
   - 第194行：`.astype(np.bool)` → `dtype=bool` 或 `.astype(bool)`
   - 第196行：`.astype(np.bool)` → `.astype(bool)`
   - 第197行：`.astype(np.bool)` → `.astype(bool)`

## 修复效果

- ✅ 兼容新版本 NumPy (1.24+)
- ✅ 不影响功能，`bool` 和 `np.bool` 功能相同
- ✅ 语法检查通过

## 验证

运行以下命令验证修复：

```bash
python3 -m py_compile mmseg/datasets/nuscenes.py mmseg/datasets/pipelines/loading.py
```

## 相关链接

- [NumPy 1.20.0 Release Notes](https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations)
- [NumPy Migration Guide](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)
