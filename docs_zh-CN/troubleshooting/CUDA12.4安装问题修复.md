# CUDA 12.4安装问题修复指南

## 问题分析

**系统环境**:
- Ubuntu 24.04.3 LTS (noble)
- 使用了Ubuntu 22.04 (jammy)的CUDA仓库

**错误信息**:
```
nsight-systems-2023.4.4 : 依赖: libtinfo5 但无法安装它
```

**根本原因**:
- Ubuntu 24.04中`libtinfo5`已被移除或重命名
- `nsight-systems`工具包需要这个依赖
- 包管理器无法解决依赖冲突

## 解决方案

### ✅ 方案1: 使用runfile安装（推荐）

**优点**:
- 不依赖包管理器
- 可以选择性安装组件
- 避免依赖冲突
- 最可靠的方法

**安装步骤**:

```bash
cd /media/ldk950413/data0/DiffBEV

# 运行自动安装脚本
bash 使用runfile安装CUDA12.4.sh
```

**手动安装**:

```bash
# 1. 下载CUDA 12.4 runfile（约3.5GB）
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

# 2. 运行安装程序
sudo sh cuda_12.4.0_550.54.14_linux.run

# 3. 在安装界面中选择：
#    [X] CUDA Toolkit 12.4
#    [ ] Driver（如果已安装驱动，取消选择）
#    [ ] Documentation（可选）
#    [ ] Samples（可选）
#    [ ] nsight-systems（取消选择，避免依赖问题）
```

**静默安装（只安装toolkit）**:

```bash
sudo sh cuda_12.4.0_550.54.14_linux.run \
    --toolkit \
    --silent \
    --toolkitpath=/usr/local/cuda-12.4 \
    --no-opengl-libs \
    --no-man-page
```

### 方案2: 仅安装核心包（跳过nsight-systems）

```bash
# 安装核心CUDA toolkit包，跳过有问题的依赖
sudo apt-get install -y --no-install-recommends \
    cuda-compiler-12-4 \
    cuda-cudart-dev-12-4 \
    cuda-libraries-dev-12-4 \
    cuda-nvcc-12-4 \
    libcurand-dev-12-4 \
    libcublas-dev-12-4 \
    libcusolver-dev-12-4 \
    libcusparse-dev-12-4 \
    libcufft-dev-12-4

# 验证安装
/usr/local/cuda-12.4/bin/nvcc --version
```

### 方案3: 使用Ubuntu 24.04的CUDA仓库

如果NVIDIA提供了Ubuntu 24.04的仓库，可以使用：

```bash
# 移除Ubuntu 22.04的keyring（如果已安装）
sudo dpkg -r cuda-keyring || true

# 下载Ubuntu 24.04的keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# 安装CUDA 12.4
sudo apt-get install -y cuda-toolkit-12-4
```

## 推荐步骤

**推荐使用方案1（runfile）**，因为：
1. 最可靠，不会遇到依赖问题
2. 安装速度快
3. 可以选择性安装组件
4. 兼容性最好

**安装后**:
1. 验证安装: `/usr/local/cuda-12.4/bin/nvcc --version`
2. 运行编译脚本: `bash 安装系统CUDA12.4.sh`

## 验证安装

```bash
# 检查nvcc
/usr/local/cuda-12.4/bin/nvcc --version

# 应该显示: release 12.4, V12.4.x

# 检查CUDA目录结构
ls -la /usr/local/cuda-12.4/bin/ | grep nvcc
ls -la /usr/local/cuda-12.4/include/ | head -10
ls -la /usr/local/cuda-12.4/lib64/ | head -10
```

## 常见问题

### Q: 为什么Ubuntu 24.04没有libtinfo5？
A: Ubuntu 24.04中ncurses库进行了重构，`libtinfo5`被合并到`libtinfo6`中。但某些旧版本的CUDA工具（如nsight-systems）仍需要libtinfo5。

### Q: 可以不安装nsight-systems吗？
A: 可以。nsight-systems是性能分析工具，不是编译MMCV所必需的。只安装CUDA toolkit即可。

### Q: 安装后如何使用CUDA 12.4？
A: 设置环境变量：
```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Q: 多个CUDA版本如何共存？
A: 可以通过环境变量切换。安装后：
- CUDA 12.4: `/usr/local/cuda-12.4`
- CUDA 13.0: `/usr/local/cuda-13.0`
- 通过`CUDA_HOME`指定使用的版本

## 下一步

安装CUDA 12.4后，运行：
```bash
bash 安装系统CUDA12.4.sh
```

这将自动配置环境并编译MMCV 2.1.0。
