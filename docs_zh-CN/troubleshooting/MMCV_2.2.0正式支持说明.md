# MMCV 2.2.0 正式支持说明

## 更新内容

本项目现已**正式支持 MMCV 2.2.0**（最新版本），不再需要通过环境变量来允许使用。

### ✅ 主要改进

1. **默认支持MMCV 2.x**: 项目默认使用MMCV 2.x（推荐配置）
2. **自动版本检测**: 自动检测并验证MMCV版本兼容性
3. **清晰的错误提示**: 如果版本不兼容，会给出清晰的提示信息
4. **向后兼容**: 仍支持通过环境变量使用MMCV 1.x（用于旧GPU/PyTorch 1.x）

## 版本要求

### 推荐配置（默认）

- **MMCV**: >= 2.0.0, < 3.0.0 （当前使用 2.2.0）
- **MMEngine**: >= 0.10.0 （当前使用 0.10.7）
- **PyTorch**: >= 2.0.0 （推荐 2.4.1+，用于支持新GPU）

### 兼容配置（通过环境变量）

如果需要使用MMCV 1.x（仅用于旧GPU/PyTorch 1.x）：

```bash
export DIFFBEV_USE_MMCV1=1
```

然后安装MMCV 1.x：
```bash
pip install mmcv-full==1.4.0
```

## 代码更新

### 1. 版本检查逻辑更新

**文件**: `mmseg/__init__.py`

- 默认使用MMCV 2.x（`USE_MMCV2 = True`）
- 自动检测版本兼容性
- 提供清晰的错误提示

### 2. 兼容性检查函数更新

**文件**: `mmseg/utils/compat.py`

- `check_mmcv_version()` 函数已更新，自动检测是否使用MMCV 2.x
- 支持通过参数或环境变量控制

## 使用方法

### 标准使用（MMCV 2.x，推荐）

```bash
# 1. 确保MMCV 2.x已安装
pip install --upgrade mmcv>=2.0.0

# 2. 确保MMEngine已安装
pip install mmengine>=0.10.0

# 3. 正常使用（无需设置环境变量）
python tools/train.py configs/baseline/lss_swin_nuscenes.py --work-dir runs/baseline --gpu-ids 0
```

### 兼容模式（MMCV 1.x，仅用于旧配置）

```bash
# 1. 设置环境变量
export DIFFBEV_USE_MMCV1=1

# 2. 安装MMCV 1.x
pip install mmcv-full==1.4.0

# 3. 正常使用
python tools/train.py configs/baseline/lss_swin_nuscenes.py --work-dir runs/baseline --gpu-ids 0
```

## 验证安装

运行以下命令验证MMCV版本：

```bash
python -c "import mmcv; print(f'MMCV版本: {mmcv.__version__}'); import mmengine; print(f'MMEngine版本: {mmengine.__version__}')"
```

预期输出：
```
MMCV版本: 2.2.0
MMEngine版本: 0.10.7
```

## 常见问题

### Q: 为什么默认使用MMCV 2.x？

A: MMCV 2.x支持PyTorch 2.x和新GPU架构（如RTX 5090），这是未来的趋势。MMCV 1.x仅维护到2024年底。

### Q: 如果我的环境是PyTorch 1.x，怎么办？

A: 有两种选择：
1. **推荐**: 升级到PyTorch 2.x和MMCV 2.x
2. **兼容**: 设置`DIFFBEV_USE_MMCV1=1`并使用MMCV 1.x

### Q: 升级MMCV 2.x会影响现有代码吗？

A: 不会。项目已经实现了完整的兼容层，确保MMCV 2.x可以正常工作。所有API调用都已通过兼容层处理。

### Q: 如何回退到MMCV 1.x？

A: 
1. 设置环境变量：`export DIFFBEV_USE_MMCV1=1`
2. 安装MMCV 1.x：`pip install mmcv-full==1.4.0`
3. 重新运行

## 技术细节

### 兼容层说明

项目已实现以下兼容层：

1. **DataContainer**: 自动处理MMCV 1.x和2.x的差异
2. **collate函数**: 支持DataContainer的collate函数
3. **Runner兼容**: MMCVRunnerCompat包装类
4. **导入兼容**: 自动从mmcv或mmengine导入API

### 版本检测逻辑

1. 默认检查MMCV 2.x版本要求（>=2.0.0, <3.0.0）
2. 如果检测到MMCV 1.x，检查环境变量`DIFFBEV_USE_MMCV1`
3. 如果环境变量未设置，提示升级到MMCV 2.x
4. 如果环境变量已设置，允许使用MMCV 1.x

## 更新日志

- **2025-01-17**: 正式支持MMCV 2.2.0，默认使用MMCV 2.x
- **2025-01-17**: 升级MMCV从2.1.0到2.2.0
- **之前**: 通过环境变量`DIFFBEV_ALLOW_PYTORCH2=1`允许使用MMCV 2.x

## 相关文档

- **MMCV GitHub**: https://github.com/open-mmlab/mmcv
- **MMCV升级记录**: `MMCV_2.2.0升级记录.md`
- **代码兼容性处理**: `代码兼容性处理说明.md`
- **README**: `README.md`