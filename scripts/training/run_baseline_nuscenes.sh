#!/bin/bash
# Baseline实验运行脚本 - LSS Transformer for nuScenes
# 使用原始LSS transformer，完全关闭diffusion模块与相关loss
# 日志与结果输出到 runs/baseline 目录

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}DiffBEV Baseline实验 - LSS Transformer for nuScenes${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""

# 检查是否在diffbev环境中
if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" != "diffbev" ]]; then
    echo -e "${YELLOW}⚠️  警告: 请先激活 diffbev 环境${NC}"
    echo "   运行: micromamba activate diffbev"
    exit 1
fi

echo -e "${GREEN}✓ 当前环境: $CONDA_DEFAULT_ENV${NC}"

# 设置环境变量以允许PyTorch 2.x（用于RTX 5090等新GPU）
export DIFFBEV_ALLOW_PYTORCH2=1
echo -e "${GREEN}✓ 已设置 DIFFBEV_ALLOW_PYTORCH2=1（允许PyTorch 2.x）${NC}"
echo ""

# 项目根目录
PROJECT_ROOT="/media/ldk950413/data0/DiffBEV"
cd "$PROJECT_ROOT"

# 输出目录（确保存在）
OUTPUT_DIR="runs/baseline"
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}输出目录: $OUTPUT_DIR${NC}"

# 配置文件
CONFIG_FILE="configs/baseline/lss_swin_nuscenes.py"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ 错误: 配置文件不存在: $CONFIG_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}配置文件: $CONFIG_FILE${NC}"
echo -e "${GREEN}输出目录: $OUTPUT_DIR${NC}"
echo ""

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU信息:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | head -1
    echo ""
fi

# 设置日志文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/train_${TIMESTAMP}.log"

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}开始训练...${NC}"
echo -e "${GREEN}日志文件: $LOG_FILE${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""

# 训练命令
echo -e "${GREEN}开始训练baseline模型...${NC}"
echo ""

python tools/train.py "$CONFIG_FILE" \
    --work-dir "$OUTPUT_DIR" \
    --gpu-ids 0 \
    2>&1 | tee "$LOG_FILE"

# 检查训练是否成功
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================================================================${NC}"
    echo -e "${GREEN}✓ 训练完成！${NC}"
    echo -e "${GREEN}================================================================================${NC}"
    echo ""
    echo "结果保存在: $OUTPUT_DIR"
    echo "日志文件: $LOG_FILE"
    echo ""
    echo "查看训练日志:"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "查看最新checkpoint:"
    echo "  ls -lh $OUTPUT_DIR/*.pth | tail -1"
    echo ""
else
    echo ""
    echo -e "${RED}================================================================================${NC}"
    echo -e "${RED}❌ 训练失败！${NC}"
    echo -e "${RED}================================================================================${NC}"
    echo ""
    echo "请查看日志文件: $LOG_FILE"
    echo ""
    exit 1
fi
