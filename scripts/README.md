# 脚本说明

## 训练脚本 (`training/`)

### 训练启动
- `快速训练.sh` - 快速启动训练
- `完整训练命令.sh` - 完整的训练命令（包含环境激活）
- `GPU训练启动脚本.sh` - GPU训练启动脚本

### 训练监控
- `实时监控训练.sh` - 实时监控训练进度
- `实时监控（直接运行版）.sh` - 可直接运行的监控脚本
- `启动实时监控.sh` - 启动监控的包装脚本
- `简洁监控训练.sh` - 简洁版监控脚本
- `监控训练.sh` - 监控训练脚本

## 环境配置脚本 (`setup/`)

### CUDA配置
- `使用runfile安装CUDA12.4.sh` - 使用runfile安装CUDA 12.4
- `安装系统CUDA12.4.sh` - 安装系统CUDA 12.4
- `使用CUDA13编译MMCV.sh` - 使用CUDA 13编译MMCV
- `配置CUDA12.4环境.sh` - 配置CUDA 12.4环境
- `解决CUDA12.4依赖问题.sh` - 解决CUDA 12.4依赖问题
- `设置CUDA库路径.sh` - 设置CUDA库路径

### PyTorch配置
- `升级PyTorch到支持RTX5090.sh` - 升级PyTorch支持RTX 5090
- `升级PyTorch支持RTX5090.sh` - 升级PyTorch支持RTX 5090（备用）
- `安装PyTorch_nightly_CUDA128.sh` - 安装PyTorch nightly CUDA 12.8

### MMCV配置
- `从源码编译MMCV.sh` - 从源码编译MMCV
- `绕过CUDA版本检查编译MMCV.sh` - 绕过CUDA版本检查编译MMCV
- `修复MMCV_CUDA库问题.sh` - 修复MMCV CUDA库问题
- `upgrade_mmcv_to_2x.sh` - 升级MMCV到2.x
- `migrate_mmcv_to_2x.py` - MMCV 2.x迁移脚本

## 数据准备脚本 (`data/`)

- `prepare_nuscenes_bev_data.sh` - 准备nuScenes BEV数据
- `检查地图文件.sh` - 检查地图文件
- `下载地图扩展文件.sh` - 下载地图扩展文件
- `清理临时文件.sh` - 清理临时文件

## 工具脚本 (`utils/`)

- `诊断训练进程.sh` - 诊断训练进程
- `check_environment.py` - 检查环境配置
- `download_pretrained.sh` - 下载预训练权重

## 使用说明

### 训练脚本
```bash
# 快速训练
bash scripts/training/快速训练.sh

# 监控训练
bash scripts/monitoring/实时监控训练.sh
```

### 环境配置
```bash
# 配置CUDA环境
bash scripts/setup/配置CUDA12.4环境.sh

# 编译MMCV
bash scripts/setup/从源码编译MMCV.sh
```

### 数据准备
```bash
# 准备数据
bash scripts/data/prepare_nuscenes_bev_data.sh

# 检查地图文件
bash scripts/data/检查地图文件.sh
```

### 监控脚本
```bash
# 实时监控训练
bash scripts/monitoring/实时监控训练.sh

# 启动监控（包装脚本）
bash scripts/monitoring/启动实时监控.sh
```

### 工具脚本
```bash
# 诊断训练进程
bash scripts/utils/诊断训练进程.sh
```
