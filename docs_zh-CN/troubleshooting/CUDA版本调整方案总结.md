# CUDA版本调整方案总结

## 问题分析

**目标**: 通过调整系统CUDA版本来匹配PyTorch 2.4.1（CUDA 12.4）

**当前状态**:
- 系统CUDA: 13.0
- PyTorch CUDA: 12.4
- 不匹配导致MMCV编译失败

## 已尝试的方案

### ✅ 方案1: 使用CUDA 13.0编译（已禁用版本检查）

**步骤**:
1. ✓ 配置CUDA_HOME指向CUDA 13.0
2. ✓ 禁用PyTorch的CUDA版本检查
3. ✓ nvcc编译器可用

**结果**: ❌ 编译失败
- 原因: CUDA 13.0和12.4之间的API不兼容
- 即使绕过版本检查，API差异仍然导致编译错误

### 方案2: 安装CUDA 12.4 toolkit

这是最理想的解决方案，但需要：

1. **下载CUDA 12.4 toolkit**:
   - 官网: https://developer.nvidia.com/cuda-12-4-0-download-archive
   - 需要选择正确的Linux发行版和架构

2. **安装**:
   ```bash
   # 下载安装包后
   sudo sh cuda_12.4.0_550.54.14_linux.run
   ```

3. **配置环境**:
   ```bash
   export CUDA_HOME=/usr/local/cuda-12.4
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

4. **编译MMCV**:
   ```bash
   export DISABLE_CUDA_VERSION_CHECK=1
   pip install mmcv==2.1.0
   ```

**优点**: 完美匹配PyTorch的CUDA版本
**缺点**: 需要管理员权限，可能需要较多磁盘空间

### 方案3: 使用conda安装CUDA 12.4（推荐）

如果系统允许，可以使用conda安装CUDA 12.4 toolkit：

```bash
# 安装CUDA 12.4 toolkit到conda环境
conda install -c nvidia cuda-toolkit=12.4 -y

# 或使用cudatoolkit
conda install -c conda-forge cudatoolkit=12.4 -y
```

然后配置环境变量使用conda的CUDA。

## 推荐方案

### 方案A: 使用conda安装CUDA 12.4（最简单）

```bash
micromamba activate diffbev
micromamba install -c nvidia cuda-toolkit=12.4 -y

# 配置环境
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 编译MMCV
export DISABLE_CUDA_VERSION_CHECK=1
export DIFFBEV_ALLOW_PYTORCH2=1
pip install mmcv==2.1.0
```

### 方案B: 使用CPU模式（已验证可用）

```bash
bash run_baseline_cpu.sh
```

### 方案C: 等待官方支持

等待OpenMMLab发布PyTorch 2.4 + CUDA 12.4的预编译MMCV wheel。

## 下一步

建议尝试**方案A**（conda安装CUDA 12.4），这是最简单且不破坏系统配置的方法。
