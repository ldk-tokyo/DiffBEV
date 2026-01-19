# MMCV 升级到 2.x 快速指南

## 快速开始

### 方法1：使用自动安装脚本（推荐）

```bash
# 1. 激活环境
micromamba activate diffbev

# 2. 运行升级脚本
bash upgrade_mmcv_to_2x.sh

# 3. 运行自动迁移脚本
python migrate_mmcv_to_2x.py

# 4. 测试
python -c "import mmcv; import mmengine; print('✓ 导入成功')"
```

### 方法2：手动安装

```bash
# 1. 卸载旧版本
pip uninstall mmcv mmcv-full -y

# 2. 安装MMCV 2.x
pip install -U openmim
mim install mmcv>=2.1.0

# 3. 安装MMEngine（必需）
pip install mmengine>=0.10.0

# 4. 验证
python -c "import mmcv; import mmengine; print(f'MMCV: {mmcv.__version__}, MMEngine: {mmengine.__version__}')"
```

## 为什么需要升级？

- ✅ **支持 RTX 5090**: MMCV 2.x 支持 PyTorch 2.x，而 RTX 5090 需要 PyTorch 2.x
- ✅ **支持新架构**: 支持 Blackwell (sm_120) 等新GPU架构
- ✅ **更好的性能**: MMCV 2.x 针对新版本 PyTorch 优化

## 主要变化

| 旧版本 (MMCV 1.x) | 新版本 (MMCV 2.x) |
|------------------|------------------|
| `mmcv.runner` | `mmengine.runner` |
| `mmcv.parallel` | `mmengine.parallel` |
| `mmcv.Config` | `mmengine.Config` |
| `mmcv.utils.Registry` | `mmengine.registry.Registry` |

## 迁移步骤

1. **安装新版本** - 使用上面的脚本
2. **运行自动迁移** - `python migrate_mmcv_to_2x.py`
3. **检查代码** - 某些API可能需要手动调整
4. **测试** - 运行小规模训练验证

## 详细文档

- **完整升级指南**: `mmcv升级指南.md`
- **自动迁移脚本**: `migrate_mmcv_to_2x.py`
- **RTX 5090兼容性**: `RTX5090兼容性解决方案.md`

## 常见问题

### Q: 升级后代码无法运行？

A: 检查是否所有API都已迁移，参考 `mmcv升级指南.md` 中的API替换表。

### Q: 可以回退吗？

A: 可以。备份文件已保存为 `*.backup`，可以恢复。

### Q: 必须升级吗？

A: 如果使用 RTX 5090，必须升级。其他GPU可以继续使用 MMCV 1.x。

## 需要帮助？

查看详细文档：
- `mmcv升级指南.md` - 完整升级步骤
- `RTX5090兼容性解决方案.md` - RTX 5090相关问题
