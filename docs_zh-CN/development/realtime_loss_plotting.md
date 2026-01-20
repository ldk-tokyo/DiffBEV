# 实时Loss曲线绘制工具

## 简介

`tools/plot_loss_realtime.py` 是一个实时监控训练loss曲线的工具，可以自动读取 `metrics.csv` 文件并生成loss曲线图，方便实时观察训练收敛情况。

## 功能特点

- ✅ **自动监控**: 监控 `metrics.csv` 文件的变化，自动更新图表
- ✅ **多维度展示**: 同时显示总loss、各项loss（Lwce/Ldepth/Ldiff）、学习率曲线
- ✅ **统计信息**: 显示loss的当前值、平均值、最小值、最大值等统计信息
- ✅ **服务器友好**: 使用非交互式后端，适合在服务器上运行
- ✅ **自动处理**: 自动处理CSV文件格式问题（如重复表头）

## 安装依赖

```bash
pip install pandas matplotlib
```

## 使用方法

### 方法1: 直接使用Python脚本

```bash
# 基本用法
python tools/plot_loss_realtime.py --work-dir runs/baseline

# 指定输出文件名和刷新间隔
python tools/plot_loss_realtime.py --work-dir runs/baseline --output loss.png --interval 5

# 只绘制一次（不循环更新）
python tools/plot_loss_realtime.py --work-dir runs/baseline --once
```

### 方法2: 使用Bash脚本（推荐）

```bash
# 基本用法
bash scripts/monitoring/实时绘制loss曲线.sh runs/baseline

# 指定输出文件名和刷新间隔
bash scripts/monitoring/实时绘制loss曲线.sh runs/baseline loss.png 5
```

### 方法3: 后台运行（推荐，不阻塞终端）

```bash
# 后台运行，输出日志到plot_loss.log
nohup python tools/plot_loss_realtime.py --work-dir runs/baseline --interval 10 > plot_loss.log 2>&1 &

# 查看日志
tail -f plot_loss.log

# 停止后台任务
pkill -f plot_loss_realtime.py
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--work-dir` | 训练输出目录（包含metrics.csv） | 必需 |
| `--output` | 输出图片文件名 | `loss_curves.png` |
| `--interval` | 刷新间隔（秒） | `10` |
| `--title` | 图表标题 | 使用work-dir名称 |
| `--once` | 只绘制一次，不循环更新 | `False` |

## 输出内容

脚本会在 `<work-dir>/loss_curves.png` 生成包含以下内容的图表：

1. **总Loss曲线**: 显示训练的总loss变化趋势
2. **Loss组件曲线**: 分别显示 Lwce、Ldepth、Ldiff 的变化（如果存在）
3. **学习率曲线**: 显示学习率调度情况
4. **统计信息**: 显示loss的统计信息（当前值、平均值、最小值、最大值等）

## 使用示例

### 示例1: 监控Baseline训练

```bash
# 启动训练（在一个终端）
bash scripts/training/run_baseline_nuscenes.sh

# 启动loss曲线绘制（在另一个终端）
python tools/plot_loss_realtime.py --work-dir runs/baseline --interval 10
```

### 示例2: 监控DiffBEV训练

```bash
# 启动训练（在一个终端）
bash scripts/training/run_diffbev_nuscenes.sh

# 启动loss曲线绘制（在另一个终端）
python tools/plot_loss_realtime.py --work-dir runs/diffbev_default --interval 10
```

### 示例3: 后台运行并查看图片

```bash
# 后台运行绘制脚本
nohup python tools/plot_loss_realtime.py --work-dir runs/baseline > plot_loss.log 2>&1 &

# 定期查看生成的图片（使用scp或其他方式下载到本地）
# 图片位置: runs/baseline/loss_curves.png
```

## 注意事项

1. **文件格式**: 脚本会自动处理CSV文件格式问题，但如果CSV文件格式异常，可能需要手动检查
2. **刷新频率**: 建议设置合理的刷新间隔（如10秒），避免过于频繁的IO操作
3. **服务器环境**: 脚本使用非交互式后端，适合在服务器上运行，生成的图片可以通过scp等方式下载到本地查看
4. **依赖安装**: 如果缺少依赖包，脚本会提示安装，或手动安装：`pip install pandas matplotlib`

## 故障排查

### 问题1: 提示缺少依赖包

```bash
pip install pandas matplotlib
```

### 问题2: 图片没有更新

- 检查 `metrics.csv` 文件是否存在且可读
- 检查文件是否有写入权限
- 检查刷新间隔是否设置合理

### 问题3: 图片显示异常

- 检查CSV文件格式是否正确
- 检查数据是否包含NaN值
- 尝试使用 `--once` 参数只绘制一次，查看错误信息

## 相关工具

- `tools/plot_metrics.py`: 批量对比多个实验的metrics
- `scripts/monitoring/实时监控训练.sh`: 实时监控训练状态
