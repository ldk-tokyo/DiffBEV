#!/bin/bash
# DiffBEV 完整训练命令 - 从激活环境到启动训练

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}DiffBEV 训练启动${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""

# 1. 激活conda/mamba环境
ENV_NAME="diffbev_py310"
ENV_PATH="$HOME/.local/share/mamba/envs/$ENV_NAME"

# 检测环境路径是否存在
if [ ! -d "$ENV_PATH" ]; then
    # 尝试其他可能的位置
    if [ -d "$HOME/anaconda3/envs/$ENV_NAME" ]; then
        ENV_PATH="$HOME/anaconda3/envs/$ENV_NAME"
    elif [ -d "$HOME/miniconda3/envs/$ENV_NAME" ]; then
        ENV_PATH="$HOME/miniconda3/envs/$ENV_NAME"
    elif [ -d "$HOME/mambaforge/envs/$ENV_NAME" ]; then
        ENV_PATH="$HOME/mambaforge/envs/$ENV_NAME"
    else
        echo -e "${RED}❌ 错误: 未找到环境 $ENV_NAME${NC}"
        echo "   请检查环境是否存在，或手动指定环境路径"
        exit 1
    fi
fi

# 如果环境未激活，尝试激活
if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    echo -e "${GREEN}正在激活环境: $ENV_NAME${NC}"
    
    # 方法1: 尝试使用micromamba shell hook
    if command -v micromamba &> /dev/null; then
        # 初始化micromamba shell hook
        eval "$(micromamba shell hook --shell bash 2>/dev/null)" 2>/dev/null || true
        if micromamba activate "$ENV_NAME" 2>/dev/null; then
            echo -e "${GREEN}✓ 已使用 micromamba 激活环境: $ENV_NAME${NC}"
        else
            # 如果激活失败，直接使用环境中的python
            echo -e "${YELLOW}⚠️  micromamba activate 失败，将直接使用环境中的Python${NC}"
            export PATH="$ENV_PATH/bin:$PATH"
        fi
    # 方法2: 直接使用环境中的Python（最可靠）
    else
        echo -e "${GREEN}直接使用环境中的Python: $ENV_PATH/bin/python${NC}"
        export PATH="$ENV_PATH/bin:$PATH"
    fi
    
    # 验证Python是否来自正确的环境
    PYTHON_ENV=$(python -c "import sys; print(sys.prefix)" 2>/dev/null || echo "")
    if [[ "$PYTHON_ENV" == *"$ENV_NAME"* ]] || [[ "$PYTHON_ENV" == "$ENV_PATH" ]]; then
        echo -e "${GREEN}✓ Python环境验证通过${NC}"
    else
        echo -e "${YELLOW}⚠️  警告: Python可能不在预期环境中${NC}"
        echo "   Python路径: $PYTHON_ENV"
        echo "   期望环境: $ENV_NAME"
    fi
else
    echo -e "${GREEN}✓ 环境已激活: $CONDA_DEFAULT_ENV${NC}"
fi

echo ""

# 2. 设置CUDA环境变量（如果需要）
export CUDA_HOME=/usr/local/cuda-12.4
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 3. 设置PyTorch 2.x相关环境变量
export DIFFBEV_ALLOW_PYTORCH2=1
export DISABLE_CUDA_VERSION_CHECK=1

echo -e "${GREEN}✓ 环境变量已设置${NC}"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  DIFFBEV_ALLOW_PYTORCH2: $DIFFBEV_ALLOW_PYTORCH2"
echo ""

# 4. 切换到项目目录
PROJECT_ROOT="/media/ldk950413/data0/DiffBEV"
cd "$PROJECT_ROOT"
echo -e "${GREEN}✓ 项目目录: $PROJECT_ROOT${NC}"
echo ""

# 5. 验证环境
echo -e "${GREEN}验证环境...${NC}"
python -c "
import torch
import sys
print(f'✓ Python: {sys.version.split()[0]}')
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA版本: {torch.version.cuda}')
    print(f'✓ GPU数量: {torch.cuda.device_count()}')
    print(f'✓ GPU 0: {torch.cuda.get_device_name(0)}')
" || {
    echo -e "${RED}❌ 环境验证失败${NC}"
    exit 1
}
echo ""

# 6. 配置文件和工作目录
CONFIG_FILE="configs/baseline/lss_swin_nuscenes.py"
WORK_DIR="runs/baseline"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ 错误: 配置文件不存在: $CONFIG_FILE${NC}"
    exit 1
fi

# 创建输出目录
mkdir -p "$WORK_DIR"

echo -e "${GREEN}训练配置:${NC}"
echo "  配置文件: $CONFIG_FILE"
echo "  工作目录: $WORK_DIR"
echo ""

# 7. 显示GPU信息
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU信息:${NC}"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | head -1
    echo ""
fi

# 8. 启动训练
echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}开始训练...${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""

# 训练命令
python tools/train.py "$CONFIG_FILE" \
    --work-dir "$WORK_DIR" \
    --gpu-ids 0

# 检查训练结果
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================================================================${NC}"
    echo -e "${GREEN}✓ 训练完成！${NC}"
    echo -e "${GREEN}================================================================================${NC}"
else
    echo ""
    echo -e "${RED}================================================================================${NC}"
    echo -e "${RED}❌ 训练失败！${NC}"
    echo -e "${RED}================================================================================${NC}"
    exit 1
fi
