# MMCV编译问题修复总结

## 问题诊断过程

### 主要错误

编译MMCV 2.1.0时遇到的关键错误：
```
fatal error: thrust/complex.h: 没有那个文件或目录
```

### 根本原因

1. **CUDA Thrust库位置变化**: 在CUDA 12.x中，Thrust库被移动到 `cccl/thrust` 目录下，而不是直接在 `include/thrust`
2. **include路径不完整**: 编译器找不到 `thrust/complex.h`，因为标准include路径中缺少 `cccl` 目录

### 解决方案

#### 步骤1: 创建thrust符号链接

```bash
ln -sfn $CONDA_PREFIX/targets/x86_64-linux/include/cccl/thrust \
        $CONDA_PREFIX/include/thrust
```

#### 步骤2: 配置完整的include路径

```bash
export CUDA_HOME=$CONDA_PREFIX
export CUDA_INCLUDE_PATH="$CONDA_PREFIX/include:$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/targets/x86_64-linux/include/cccl"
export CPLUS_INCLUDE_PATH="$CUDA_INCLUDE_PATH:$CPLUS_INCLUDE_PATH"
export CPATH="$CUDA_INCLUDE_PATH:$CPATH"
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export DISABLE_CUDA_VERSION_CHECK=1
export DIFFBEV_ALLOW_PYTORCH2=1
```

#### 步骤3: 安装MMCV

```bash
pip uninstall -y mmcv mmcv-full
pip install mmcv==2.1.0 --no-cache-dir
```

## 完整修复脚本

```bash
#!/bin/bash
# MMCV编译修复脚本

set -e

# 配置环境变量
export CUDA_HOME=$CONDA_PREFIX
export CUDA_INCLUDE_PATH="$CONDA_PREFIX/include:$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/targets/x86_64-linux/include/cccl"
export CPLUS_INCLUDE_PATH="$CUDA_INCLUDE_PATH:$CPLUS_INCLUDE_PATH"
export CPATH="$CUDA_INCLUDE_PATH:$CPATH"
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export DISABLE_CUDA_VERSION_CHECK=1
export DIFFBEV_ALLOW_PYTORCH2=1

# 创建thrust符号链接
if [ -d "$CONDA_PREFIX/targets/x86_64-linux/include/cccl/thrust" ] && [ ! -d "$CONDA_PREFIX/include/thrust" ]; then
    ln -sfn "$CONDA_PREFIX/targets/x86_64-linux/include/cccl/thrust" "$CONDA_PREFIX/include/thrust"
    echo "✓ 已创建thrust符号链接"
fi

# 卸载旧版MMCV
pip uninstall -y mmcv mmcv-full 2>/dev/null || true

# 编译安装MMCV
echo "开始编译MMCV（这可能需要10-30分钟）..."
pip install mmcv==2.1.0 --no-cache-dir

# 验证安装
python -c "import mmcv; print('✓ MMCV版本:', mmcv.__version__)"
```

## 技术细节

### CUDA 12.x的变化

- **旧路径**: `include/thrust/`
- **新路径**: `targets/x86_64-linux/include/cccl/thrust/`

PyTorch的 `c10/util/complex.h` 仍然使用旧的 `#include <thrust/complex.h>` 语法，所以需要通过符号链接或include路径来解决兼容性。

### 为什么需要多个include路径

1. `$CONDA_PREFIX/include`: 标准头文件位置
2. `$CONDA_PREFIX/targets/x86_64-linux/include`: CUDA平台特定头文件
3. `$CONDA_PREFIX/targets/x86_64-linux/include/cccl`: CUDA 12.x的CCCL库（包含thrust）

## 验证

编译成功后，验证MMCV安装：

```python
import mmcv
print(f'MMCV版本: {mmcv.__version__}')

from mmcv.ops import nms
print('✓ CUDA扩展模块可用')
```

## 相关文件

- CUDA 12.4 toolkit: 通过conda安装
- nvcc编译器: 版本12.4
- PyTorch: 2.4.1+cu124
- MMCV: 2.1.0（从源码编译）
