# PyTorch 2.9.1 安装说明

## 问题概述

用户提到PyTorch官网已发布2.9.1版本，并提供了支持CUDA 12.8的安装指令，但尝试安装时遇到问题。

## 官方安装指令

根据用户提供的PyTorch官网信息：

**For Linux x86:**
```bash
pip3 install torch torchvision
```

**For Linux Aarch64:**
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 尝试结果

### 1. 标准pip安装

```bash
pip install torch torchvision torchaudio
```

**结果**: 安装的是 PyTorch 2.4.1（不是2.9.1）

**原因**: Python 3.8的限制，PyTorch 2.9.1可能不支持Python 3.8

### 2. 指定版本安装

```bash
pip install torch==2.9.1 torchvision torchaudio
```

**结果**: 错误 - `Could not find a version that satisfies the requirement torch==2.9.1`

**可用版本**: 最高到 2.4.1 (Python 3.8)

### 3. 使用test通道

```bash
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/test/cu128
```

**结果**: 错误 - `Could not find a version that satisfies the requirement torch==2.9.1`

### 4. 使用nightly通道

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**结果**: 错误 - `Could not find a version that satisfies the requirement torch`

## 根本原因

### Python版本限制

**当前Python版本**: 3.8.20

**PyTorch 2.9.1要求**: 很可能需要 Python 3.9 或更高版本

PyTorch 2.9.1可能没有为Python 3.8构建预编译包，因为：
1. Python 3.8已经接近EOL（End of Life）
2. 新版本PyTorch通常支持更新的Python版本
3. 从pip可用版本列表看，Python 3.8最高只支持到2.4.1

## 解决方案

### 方案1：升级Python版本（推荐）

如果PyTorch 2.9.1支持Python 3.9+：

```bash
# 创建新环境（Python 3.9或3.10）
micromamba create -n diffbev_py39 python=3.9 -y
micromamba activate diffbev_py39

# 安装PyTorch 2.9.1
pip install torch torchvision torchaudio

# 验证版本
python -c "import torch; print(torch.__version__)"

# 重新安装项目依赖
cd /media/ldk950413/data0/DiffBEV
pip install -e .
```

### 方案2：使用官方cu128索引（Aarch64指令）

如果系统是x86_64，但想使用cu128版本：

```bash
# 卸载旧版本
pip uninstall -y torch torchvision torchaudio

# 使用cu128索引（尝试）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**注意**: 可能需要Python 3.9+

### 方案3：检查PyTorch官网获取准确信息

访问PyTorch官网获取最新安装指令：
- 官网: https://pytorch.org/get-started/locally/
- 检查Python版本要求
- 检查CUDA版本要求
- 检查支持的平台

### 方案4：暂时使用PyTorch 2.4.1

如果无法升级Python版本，可以：
1. 继续使用PyTorch 2.4.1（虽然不支持sm_120）
2. 使用CPU训练进行代码测试
3. 等待PyTorch 2.9.1的Python 3.8构建版本（可能不会发布）

## 验证PyTorch版本

安装后，验证版本和CUDA支持：

```bash
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA版本: {torch.version.cuda}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'计算能力: {torch.cuda.get_device_capability(0)}')
    
    # 测试Conv2d（训练中的关键操作）
    x = torch.randn(4, 3, 800, 600).cuda()
    conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
    y = conv(x)
    print(f'✓ Conv2d测试成功！输出shape: {y.shape}')
"
```

## 当前状态

- ✅ Python 3.8.20 已安装
- ✅ PyTorch 2.4.1+cu121 已安装（不支持sm_120）
- ❌ PyTorch 2.9.1 无法安装（Python 3.8限制）
- ❌ RTX 5090 (sm_120) 无法正常训练（CUDA kernel错误）

## 建议

### 短期（立即）

1. **升级Python到3.9或3.10**
   ```bash
   micromamba create -n diffbev_py39 python=3.9 -y
   micromamba activate diffbev_py39
   ```

2. **安装PyTorch 2.9.1**
   ```bash
   pip install torch torchvision torchaudio
   ```

3. **验证RTX 5090支持**
   ```bash
   python -c "import torch; x = torch.randn(10, 10).cuda(); conv = torch.nn.Conv2d(10, 20, 3).cuda(); y = conv(x); print('成功！')"
   ```

### 中期（如果升级Python不可行）

1. **等待PyTorch 2.9.1的Python 3.8构建**（可能不会发布）
2. **考虑从源码编译PyTorch**（添加sm_120支持）
3. **使用CPU训练**（仅用于测试代码流程）

### 长期

1. **升级Python到3.9+**
   - Python 3.8将在2024年10月EOL
   - 新版本PyTorch通常不支持EOL的Python版本
   - 升级Python可以获取更好的支持

## 相关文档

- **PyTorch安装指南**: https://pytorch.org/get-started/locally/
- **PyTorch 2.9发布说明**: https://pytorch.org/blog/pytorch-2-9/
- **PyTorch 2.7发布说明**（Blackwell支持）: https://pytorch.org/blog/pytorch-2-7/
- **RTX5090支持状态总结**: `RTX5090支持状态总结.md`

## 更新日志

- **2025-01-17**: 尝试安装PyTorch 2.9.1（Python 3.8限制，失败）
- **2025-01-17**: 创建安装说明文档
- **待定**: 升级Python到3.9+并安装PyTorch 2.9.1