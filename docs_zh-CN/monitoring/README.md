# 监控文档

## 监控脚本说明

训练监控脚本位于 `scripts/monitoring/` 目录。

### 脚本列表

- `实时监控训练.sh` - 详细的实时监控脚本，显示训练进度、loss、学习率、GPU使用率等
- `简洁监控训练.sh` - 简洁版监控脚本，显示关键信息
- `实时监控（直接运行版）.sh` - 可直接运行的监控脚本，自动刷新
- `启动实时监控.sh` - 启动监控的包装脚本，支持watch命令
- `监控训练.sh` - 基础监控脚本

### 使用方法

#### 实时监控（推荐）
```bash
bash scripts/monitoring/实时监控训练.sh
```

#### 使用watch命令自动刷新
```bash
watch -n 2 -c bash scripts/monitoring/简洁监控训练.sh
```

#### 直接运行版（自动刷新）
```bash
bash scripts/monitoring/实时监控（直接运行版）.sh
# 或指定刷新间隔（秒）
bash scripts/monitoring/实时监控（直接运行版）.sh 5
```

详细使用说明请参考：[实时监控使用说明](实时监控使用说明.md)
