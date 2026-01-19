# Git工作流程指南

## 提交代码到GitHub

### Step 1: 检查当前状态

```bash
cd /media/ldk950413/data0/DiffBEV
git status
```

### Step 2: 添加需要提交的文件

**重要**：只提交源代码和配置文件，不要提交：
- `__pycache__/` 目录
- `*.log` 日志文件
- `runs/` 训练输出目录
- `*.pth` checkpoint文件
- `metrics.csv` 指标文件

```bash
# 添加所有修改的源代码文件
git add mmseg/
git add configs/
git add scripts/
git add tools/
git add docs_zh-CN/

# 添加.gitignore（如果存在）
git add .gitignore

# 或者使用更精确的方式，逐个添加重要文件
git add mmseg/apis/train.py
git add mmseg/core/evaluation/eval_hooks.py
git add mmseg/core/hooks/
git add mmseg/datasets/nuscenes.py
git add mmseg/utils/runner_compat.py
git add mmseg/utils/metrics_logger.py
git add mmseg/utils/__init__.py
git add configs/baseline/
git add configs/diffbev/
git add scripts/training/
git add scripts/monitoring/
git add tools/plot_metrics.py
git add tools/vis_nuscenes_bev.py
git add docs_zh-CN/
```

### Step 3: 检查暂存的文件

```bash
git status
```

确认只包含需要提交的文件，不包含 `__pycache__`、`.log`、`runs/` 等。

### Step 4: 提交更改

```bash
git commit -m "feat: 添加训练流程完整功能

- 修复nuscenes数据集pre_eval和evaluate方法
- 实现DistEvalHook和EvalHook的完整功能
- 添加MetricsLogger统一记录训练和评估指标
- 实现LossCheckHook进行loss结构自检
- 修复runner_compat中hook priority类型问题
- 添加baseline和diffbev配置及启动脚本
- 实现plot_metrics.py和vis_nuscenes_bev.py工具
- 修复监控脚本的进度条显示问题
- 添加完整的训练runbook文档"
```

### Step 5: 推送到GitHub

```bash
# 推送到当前分支（inmage-test）
git push origin inmage-test

# 或者如果要推送到main分支，先切换分支
# git checkout main
# git merge inmage-test
# git push origin main
```

### Step 6: 验证推送成功

```bash
# 检查远程分支状态
git log origin/inmage-test --oneline -5
```

## 常见问题

### 问题1: 误添加了不需要的文件

```bash
# 从暂存区移除文件（但保留工作区文件）
git reset HEAD <文件路径>

# 例如：
git reset HEAD mmseg/apis/__pycache__/train.cpython-310.pyc
```

### 问题2: 需要更新.gitignore

```bash
# 编辑.gitignore后，需要移除已跟踪的文件
git rm --cached -r __pycache__/
git rm --cached runs/*.log
git add .gitignore
git commit -m "chore: 更新.gitignore排除不需要的文件"
```

### 问题3: 提交信息写错了

```bash
# 修改最后一次提交信息
git commit --amend -m "新的提交信息"

# 如果已经推送，需要强制推送（谨慎使用）
git push origin inmage-test --force
```

## 推荐的工作流程

1. **开发前**：创建新分支
   ```bash
   git checkout -b feature/new-feature
   ```

2. **开发中**：频繁提交
   ```bash
   git add <修改的文件>
   git commit -m "简短描述"
   ```

3. **推送前**：检查状态
   ```bash
   git status
   git diff --staged  # 查看暂存的更改
   ```

4. **推送后**：验证
   ```bash
   git log --oneline -5
   ```
