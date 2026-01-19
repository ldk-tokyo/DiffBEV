# 修复Git历史中的大文件问题

## 问题描述

推送代码到GitHub时遇到错误：
```
remote: fatal: pack exceeds maximum allowed size (2.00 GiB)
error: 远程解包失败：index-pack failed
```

原因是Git历史中包含了超过2GB的大文件。

## 解决方案

### 方案1：清理Git历史（推荐）

使用 `git filter-branch` 从历史中移除大文件：

```bash
cd /media/ldk950413/data0/DiffBEV

# 运行清理脚本
bash scripts/utils/fix_large_files.sh
```

脚本会：
1. 创建备份分支
2. 从历史中移除大文件（CUDA安装文件、checkpoint文件等）
3. 清理引用和垃圾回收
4. 显示新的仓库大小

清理完成后，强制推送：
```bash
git push origin inmage-test --force
```

### 方案2：手动清理（如果脚本失败）

```bash
cd /media/ldk950413/data0/DiffBEV

# 1. 备份当前分支
git branch backup-$(date +%Y%m%d)

# 2. 移除大文件
git filter-branch --force --index-filter \
    "git rm --cached --ignore-unmatch \
        cuda_12.4.0_550.54.14_linux.run \
        cuda-keyring_1.1-1_all.deb \
        runs/baseline/best_mIoU.pth" \
    --prune-empty --tag-name-filter cat -- --all

# 3. 清理引用
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# 4. 检查大小
git count-objects -vH

# 5. 强制推送
git push origin inmage-test --force
```

### 方案3：使用git-filter-repo（更快，需要安装）

```bash
# 安装git-filter-repo
pip install git-filter-repo

# 移除大文件
git filter-repo --path cuda_12.4.0_550.54.14_linux.run --invert-paths
git filter-repo --path cuda-keyring_1.1-1_all.deb --invert-paths
git filter-repo --path runs/baseline/best_mIoU.pth --invert-paths

# 强制推送
git push origin inmage-test --force
```

## 预防措施

1. **使用.gitignore**：确保大文件不会被提交
   ```bash
   # 已添加到.gitignore
   *.pth
   *.pth.tar
   *.ckpt
   runs/
   *.log
   *.deb
   *.run
   ```

2. **提交前检查**：
   ```bash
   # 检查要提交的文件大小
   git diff --cached --stat
   
   # 检查大文件
   git ls-files | xargs du -h | sort -rh | head -20
   ```

3. **使用Git LFS处理大文件**（如果需要保留大文件）：
   ```bash
   # 安装Git LFS
   git lfs install
   
   # 跟踪大文件
   git lfs track "*.pth"
   git lfs track "*.pth.tar"
   ```

## 注意事项

1. **重写历史的风险**：
   - 会改变所有提交的SHA值
   - 如果其他人也在使用这个仓库，需要通知他们重新克隆
   - 需要使用 `--force` 推送

2. **备份**：
   - 脚本会自动创建备份分支
   - 如果出现问题，可以恢复：`git checkout backup-YYYYMMDD-HHMMSS`

3. **验证**：
   - 清理后检查仓库大小：`git count-objects -vH`
   - 确认大文件已移除：`git log --all -- cuda_12.4.0_550.54.14_linux.run`

## 常见问题

### Q: 清理后仓库仍然很大？
A: 可能需要多次运行 `git gc --prune=now --aggressive`，或者使用 `git-filter-repo`。

### Q: 推送时仍然失败？
A: 检查是否还有其他大文件：`git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | awk '/^blob/ {print substr($0,6)}' | sort -k2 -n -r | head -20`

### Q: 如何恢复？
A: 使用备份分支：`git checkout backup-YYYYMMDD-HHMMSS`
