# RTX 5090 CUDA错误解决方案

## 错误信息

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

## 根本原因

- **RTX 5090 GPU** 使用 **sm_120** 架构（Blackwell架构）
- **PyTorch 2.4.1+cu124** 只支持 sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_90
- **不兼容**：当前PyTorch版本不支持 sm_120，无法在 RTX 5090 上执行CUDA操作

## 解决方案

### 方案1：升级PyTorch到支持sm_120的版本（推荐）

RTX 5090需要PyTorch nightly版本或未来版本（PyTorch 2.5+）才能支持sm_120：

```bash
# 方法1：安装PyTorch nightly（支持sm_120，推荐）
# 使用提供的升级脚本
bash 升级PyTorch支持RTX5090.sh

# 或手动安装
pip uninstall -y torch torchvision torchaudio
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
```

**注意**：
- ⚠️ PyTorch nightly版本可能不稳定
- ⚠️ 需要CUDA 12.4+和驱动 >= 550
- ⚠️ 如果nightly版本仍不支持，可能需要等待PyTorch 2.5+正式版

**注意**：
- 升级PyTorch可能需要升级mmcv
- 可能需要修改部分代码以适应新版本API
- 建议在单独的conda环境中测试

### 方案2：使用CPU训练（临时方案）

如果暂时无法升级PyTorch，可以使用CPU训练（性能会很慢）：

```bash
# 方法1：通过环境变量
export CUDA_VISIBLE_DEVICES=""
python tools/train.py configs/baseline/lss_swin_nuscenes.py --work-dir runs/baseline --gpu-ids 0

# 方法2：修改代码强制使用CPU（见下方代码修改）
```

**性能影响**：
- CPU训练速度会比GPU慢几十到上百倍
- 不适合完整训练（200k iterations可能需要数周）
- 仅适合测试代码流程和小规模实验

### 方案3：使用兼容的GPU（如果可用）

如果有其他兼容的GPU（如RTX 3090, RTX 4090, A100等），可以使用：

```bash
# 检查可用GPU
nvidia-smi

# 指定GPU（如果有多个）
CUDA_VISIBLE_DEVICES=1 python tools/train.py ...
```

## 立即解决方案：CPU训练

### 步骤1：修改训练脚本支持CPU模式

修改 `tools/train.py` 或直接在运行时设置环境变量：

```bash
# 隐藏所有GPU
export CUDA_VISIBLE_DEVICES=""

# 运行训练（会自动使用CPU）
cd /media/ldk950413/data0/DiffBEV
python tools/train.py configs/baseline/lss_swin_nuscenes.py \
    --work-dir runs/baseline \
    --gpu-ids 0
```

### 步骤2：验证CPU模式

```bash
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'设备数: {torch.cuda.device_count()}')"
```

如果输出 `CUDA可用: False` 和 `设备数: 0`，说明已切换到CPU模式。

## 长期解决方案：升级环境

### 步骤1：创建新的conda环境

```bash
# 创建新环境
micromamba create -n diffbev_pytorch2 python=3.8 -y
micromamba activate diffbev_pytorch2

# 安装支持sm_120的PyTorch
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); x = torch.randn(2, 3).cuda(); print('GPU测试成功！')"
```

### 步骤2：安装其他依赖

```bash
# 安装mmcv（可能需要2.x版本）
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu124/torch2.0/index.html

# 或使用mmcv 2.0+
pip install openmim
mim install mmcv>=2.0.0
mim install mmengine

# 安装项目依赖
cd /media/ldk950413/data0/DiffBEV
pip install -e .
```

### 步骤3：处理API兼容性问题

可能需要修改代码以适应PyTorch 2.x和MMCV 2.x的API变化（参考 `代码兼容性处理说明.md`）。

## 当前状态

- ✅ 数据准备完成
- ✅ 代码修复完成（calib.json问题已解决）
- ❌ **CUDA兼容性问题**（需要升级PyTorch或使用CPU）

## 建议

**短期**：使用CPU模式测试代码流程，确保没有其他错误。

**中期**：升级到支持sm_120的PyTorch版本（需要完整的依赖升级和代码修改）。

**长期**：考虑使用兼容的GPU或完整的项目升级。

## 验证命令

```bash
# 检查当前PyTorch支持的架构
python -c "import torch; print('当前PyTorch版本:', torch.__version__)"
python -c "import torch; print('CUDA可用:', torch.cuda.is_available())"
python -c "import torch; print('GPU名称:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# 测试CUDA操作
python -c "import torch; x = torch.randn(2, 3).cuda(); print('测试成功:', x.shape)" || echo "CUDA测试失败"
```
