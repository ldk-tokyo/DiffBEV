# PyTorch 2.x 正式支持说明

## ✅ 更新内容

本项目现已**正式支持 PyTorch 2.x**，默认使用PyTorch 2.x（推荐配置），无需设置环境变量。

## 📦 当前配置

### 推荐配置（默认）

- **PyTorch**: >= 2.0.0 （当前使用 2.4.1+cu124）
- **MMCV**: >= 2.0.0 （当前使用 2.2.0）
- **MMEngine**: >= 0.10.0 （当前使用 0.10.7）
- **CUDA**: 11.1+ / 12.4+ （推荐12.4+用于新GPU如RTX 5090）

### 兼容配置（通过环境变量）

如果需要使用PyTorch 1.x（仅用于旧配置）：

```bash
export DIFFBEV_USE_PYTORCH1=1
export DIFFBEV_USE_MMCV1=1
```

## 🚀 快速开始

### 标准使用（PyTorch 2.x，推荐）

```bash
# 1. 创建conda环境
conda create -n diffbev python=3.8
conda activate diffbev

# 2. 安装PyTorch 2.x（推荐，支持新GPU如RTX 5090）
# 对于CUDA 12.4:
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
# 或访问 https://pytorch.org/get-started/locally/ 获取适合您系统的安装命令

# 3. 安装MMCV 2.x（推荐，支持PyTorch 2.x）
pip install -U openmim
mim install mmcv>=2.0.0

# 4. 安装MMEngine（MMCV 2.x必需）
pip install mmengine>=0.10.0

# 5. 正常使用（无需设置环境变量）
python tools/train.py configs/baseline/lss_swin_nuscenes.py --work-dir runs/baseline --gpu-ids 0
```

### 兼容模式（PyTorch 1.x，仅用于旧配置）

如果需要使用PyTorch 1.x（仅用于旧GPU/PyTorch 1.x）：

```bash
# 1. 设置环境变量
export DIFFBEV_USE_PYTORCH1=1
export DIFFBEV_USE_MMCV1=1

# 2. 创建conda环境
conda create -n diffbev python=3.7
conda activate diffbev

# 3. 安装PyTorch 1.x
conda install pytorch==1.9.1 torchvision==0.10.1 cudatoolkit=11.1 -c pytorch

# 4. 安装MMCV 1.x
pip install -U openmim
mim install mmcv-full==1.4.0

# 5. 正常使用
python tools/train.py configs/baseline/lss_swin_nuscenes.py --work-dir runs/baseline --gpu-ids 0
```

## 📋 版本要求

### 推荐配置（默认）

- **PyTorch**: >= 2.0.0
- **MMCV**: >= 2.0.0, < 3.0.0
- **MMEngine**: >= 0.10.0

### 兼容配置（通过环境变量）

- **PyTorch**: 1.9.1
- **MMCV**: 1.3.13 - 1.4.0

## 🔧 技术细节

### 版本检测逻辑

1. **默认检查PyTorch 2.x版本要求**（>=2.0.0）
2. **如果检测到PyTorch 1.x**，检查环境变量`DIFFBEV_USE_PYTORCH1`
3. **如果环境变量未设置**，提示升级到PyTorch 2.x（不抛出异常，向后兼容）
4. **如果环境变量已设置**，允许使用PyTorch 1.x

### 兼容层

项目已实现完整的兼容层，确保PyTorch 2.x和MMCV 2.x正常工作：

1. **DataContainer**: 自动处理MMCV 1.x和2.x的差异
2. **collate函数**: 支持DataContainer的collate函数
3. **Runner兼容**: MMCVRunnerCompat包装类
4. **导入兼容**: 自动从mmcv或mmengine导入API

## 📚 相关文档

- **PyTorch安装指南**: https://pytorch.org/get-started/locally/
- **MMCV GitHub**: https://github.com/open-mmlab/mmcv
- **MMCV升级记录**: `MMCV_2.2.0升级记录.md`
- **代码兼容性处理**: `代码兼容性处理说明.md`

## ❓ 常见问题

### Q: 为什么默认使用PyTorch 2.x？

A: PyTorch 2.x支持新GPU架构（如RTX 5090的sm_120），性能更好，并且是未来的趋势。

### Q: 升级PyTorch 2.x会影响现有代码吗？

A: 不会。项目已经实现了完整的兼容层，确保PyTorch 2.x和MMCV 2.x可以正常工作。所有API调用都已通过兼容层处理。

### Q: 如何验证PyTorch版本？

A: 运行以下命令：

```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
```

预期输出：`PyTorch版本: 2.4.1+cu124`（或类似版本）

### Q: 如果我的环境是PyTorch 1.x，怎么办？

A: 有两种选择：
1. **推荐**: 升级到PyTorch 2.x和MMCV 2.x
2. **兼容**: 设置`DIFFBEV_USE_PYTORCH1=1`和`DIFFBEV_USE_MMCV1=1`并使用PyTorch 1.x

## 🎯 优势

使用PyTorch 2.x的优势：

1. ✅ **支持新GPU架构**: 支持RTX 5090等新GPU架构（sm_120）
2. ✅ **更好的性能**: PyTorch 2.x针对新硬件优化
3. ✅ **长期支持**: PyTorch 1.x维护有限，2.x是未来趋势
4. ✅ **完整兼容**: 项目已实现完整的兼容层

## 更新日志

- **2025-01-17**: 正式支持PyTorch 2.x，默认使用PyTorch 2.x
- **之前**: 通过环境变量`DIFFBEV_ALLOW_PYTORCH2=1`允许使用PyTorch 2.x