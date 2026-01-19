# 系统CUDA 12.4 toolkit安装指南

## 概述

为了成功编译MMCV 2.1.0，需要使用系统完整的CUDA 12.4 toolkit（而非conda版本）。本指南提供详细的安装步骤。

## 当前状态

- **系统CUDA**: 13.0
- **conda CUDA**: 12.4（不完整）
- **需要**: 系统级CUDA 12.4 toolkit

## 安装步骤

### 方法1: 使用deb包安装（推荐，Ubuntu 22.04）

```bash
# 1. 下载并安装CUDA keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# 2. 更新apt源
sudo apt-get update

# 3. 安装CUDA 12.4 toolkit
sudo apt-get -y install cuda-toolkit-12-4

# 4. 验证安装
/usr/local/cuda-12.4/bin/nvcc --version
```

### 方法2: 使用runfile安装

1. **访问下载页面**:
   - https://developer.nvidia.com/cuda-12-4-0-download-archive

2. **选择安装类型**:
   - Linux > x86_64 > Ubuntu > 22.04 > runfile (local)

3. **下载并安装**:
   ```bash
   # 下载安装包
   wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
   
   # 运行安装程序
   sudo sh cuda_12.4.0_550.54.14_linux.run
   
   # 安装时选择：
   # - [X] CUDA Toolkit 12.4
   # - [ ] 取消选择驱动（如果已安装）
   ```

### 方法3: 使用apt仓库（其他Ubuntu版本）

对于Ubuntu 20.04:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
```

## 验证安装

安装完成后，验证CUDA 12.4：

```bash
# 检查安装位置
ls -la /usr/local/cuda-12.4/bin/nvcc

# 检查版本
/usr/local/cuda-12.4/bin/nvcc --version

# 应该显示: release 12.4, V12.4.x
```

## 使用安装脚本编译MMCV

安装CUDA 12.4后，运行安装脚本：

```bash
cd /media/ldk950413/data0/DiffBEV
bash 安装系统CUDA12.4.sh
```

脚本将自动：
1. 检测CUDA 12.4安装位置
2. 配置环境变量
3. 创建thrust库符号链接
4. 编译MMCV 2.1.0

## 手动配置环境变量（可选）

如果不想运行脚本，可以手动配置：

```bash
# 设置CUDA 12.4环境
export CUDA_HOME=/usr/local/cuda-12.4
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 配置include路径
export CUDA_INCLUDE_PATH="$CUDA_HOME/include:$CUDA_HOME/targets/x86_64-linux/include:$CUDA_HOME/targets/x86_64-linux/include/cccl"
export CPLUS_INCLUDE_PATH="$CUDA_INCLUDE_PATH:$CPLUS_INCLUDE_PATH"
export CPATH="$CUDA_INCLUDE_PATH:$CPATH"

# 创建thrust符号链接（如果需要）
if [ -d "$CUDA_HOME/targets/x86_64-linux/include/cccl/thrust" ] && [ ! -d "$CUDA_HOME/include/thrust" ]; then
    sudo ln -sfn "$CUDA_HOME/targets/x86_64-linux/include/cccl/thrust" "$CUDA_HOME/include/thrust"
fi

# 编译MMCV
export DISABLE_CUDA_VERSION_CHECK=1
export DIFFBEV_ALLOW_PYTORCH2=1
pip uninstall -y mmcv mmcv-full
pip install mmcv==2.1.0 --no-cache-dir
```

## 将环境变量添加到bashrc（持久化）

```bash
cat >> ~/.bashrc << 'EOF'

# CUDA 12.4环境配置（用于MMCV）
export CUDA_HOME=/usr/local/cuda-12.4
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF

source ~/.bashrc
```

## 验证MMCV编译

编译完成后，验证：

```python
import mmcv
print(f'MMCV版本: {mmcv.__version__}')

from mmcv.ops import nms
print('✓ CUDA扩展模块可用')
```

## 常见问题

### Q: 如何在同一系统上安装多个CUDA版本？
A: CUDA允许多版本共存。每个版本安装在不同的目录（如`/usr/local/cuda-12.4`和`/usr/local/cuda-13.0`），通过环境变量切换使用哪个版本。

### Q: 安装CUDA 12.4会影响CUDA 13.0吗？
A: 不会。两个版本可以共存。只需通过`CUDA_HOME`环境变量指定要使用的版本。

### Q: 需要卸载conda的CUDA吗？
A: 不需要。系统CUDA优先于conda CUDA（当设置了`CUDA_HOME`时）。两者可以共存。

### Q: 编译仍然失败怎么办？
A: 检查日志文件 `/tmp/mmcv_build_system_cuda124.log`，查看具体错误信息。可能需要：
- 安装额外的编译依赖（如g++、make等）
- 检查CUDA toolkit是否完整安装
- 查看MMCV编译要求

## 依赖要求

编译MMCV需要：
- CUDA 12.4 toolkit（完整版）
- GCC 7+ 或 GCC 9+
- CMake 3.18+
- Ninja
- Python 3.8+

## 参考链接

- CUDA 12.4下载: https://developer.nvidia.com/cuda-12-4-0-download-archive
- CUDA安装指南: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
- MMCV编译文档: https://mmcv.readthedocs.io/en/latest/get_started/installation.html
