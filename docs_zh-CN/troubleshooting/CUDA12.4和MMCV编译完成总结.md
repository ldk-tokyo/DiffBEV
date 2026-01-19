# CUDA 12.4和MMCV编译完成总结

## ✅ 完成状态

### 1. CUDA 12.4 toolkit
- **安装方式**: 系统级安装（runfile方式）
- **安装路径**: `/usr/local/cuda-12.4`
- **nvcc版本**: 12.4.99
- **验证**: ✓ 可用

### 2. MMCV 2.1.0
- **版本**: 2.1.0
- **编译状态**: ✓ 成功
- **CUDA扩展**: ✓ 可用
- **验证**: ✓ 通过

### 3. 环境配置
- **PyTorch**: 2.4.1+cu124
- **CUDA版本**: 12.4
- **Python环境**: conda diffbev (Python 3.8.20)

## 解决的问题

1. ✅ **CUDA版本匹配**: 系统CUDA 12.4匹配PyTorch的CUDA 12.4
2. ✅ **thrust库路径**: 通过符号链接解决
3. ✅ **API兼容性**: 使用系统完整CUDA toolkit解决了API问题
4. ✅ **依赖冲突**: 使用runfile安装避免了Ubuntu 24.04的依赖问题

## 使用方法

### 方法1: 使用GPU训练启动脚本

```bash
cd /media/ldk950413/data0/DiffBEV
bash GPU训练启动脚本.sh
```

### 方法2: 手动启动

```bash
# 1. 激活环境
micromamba activate diffbev

# 2. 配置CUDA环境
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export DIFFBEV_ALLOW_PYTORCH2=1

# 3. 运行训练
cd /media/ldk950413/data0/DiffBEV
python tools/train.py configs/baseline/lss_swin_nuscenes.py \
    --work-dir runs/baseline \
    --gpu-ids 0
```

### 方法3: 使用原有脚本

```bash
# 配置环境变量后
bash run_baseline_nuscenes.sh
```

## 持久化环境变量（可选）

将以下内容添加到`~/.bashrc`以持久化配置：

```bash
# CUDA 12.4环境配置
export CUDA_HOME=/usr/local/cuda-12.4
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export DIFFBEV_ALLOW_PYTORCH2=1
```

然后运行：
```bash
source ~/.bashrc
```

## 验证安装

```bash
# 验证CUDA
/usr/local/cuda-12.4/bin/nvcc --version

# 验证PyTorch和MMCV
python -c "
import torch
import mmcv
print(f'PyTorch: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'CUDA版本: {torch.version.cuda}')
print(f'MMCV: {mmcv.__version__}')
from mmcv.ops import nms
print('✓ MMCV CUDA扩展模块可用')
"
```

## 文件清单

创建的工具脚本：
- `使用runfile安装CUDA12.4.sh` - CUDA安装脚本
- `安装系统CUDA12.4.sh` - 配置和编译脚本
- `GPU训练启动脚本.sh` - GPU训练启动脚本
- `解决CUDA12.4依赖问题.sh` - 依赖问题修复脚本

文档：
- `系统CUDA12.4安装指南.md` - 安装指南
- `CUDA12.4安装问题修复.md` - 问题修复指南
- `CUDA版本调整方案总结.md` - 版本调整总结
- `MMCV编译问题修复总结.md` - 编译问题修复

## 下一步

现在可以开始训练了！

```bash
bash GPU训练启动脚本.sh
```

或使用baseline脚本：

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export DIFFBEV_ALLOW_PYTORCH2=1
bash run_baseline_nuscenes.sh
```

## 注意事项

1. **每次新终端**: 如果未持久化环境变量，需要重新设置CUDA环境变量
2. **多个CUDA版本**: 系统同时有CUDA 12.4和13.0，通过`CUDA_HOME`环境变量指定使用的版本
3. **conda环境**: 确保在`diffbev` conda环境中运行

## 成功标志

✓ CUDA 12.4 toolkit安装成功
✓ MMCV 2.1.0编译成功
✓ CUDA扩展模块可用
✓ 可以开始GPU训练

🎉 **恭喜！所有准备工作已完成，可以开始训练了！**
