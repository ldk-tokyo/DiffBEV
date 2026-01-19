#!/bin/bash
# DiffBEV实验运行脚本 - LSS Transformer with Diffusion for nuScenes
# 启用diffusion模块，使用FS-BEV条件输入和Cross-Attention融合
# 日志与结果输出到 runs/diffbev_default 目录

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}DiffBEV实验 - LSS Transformer with Diffusion for nuScenes${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# 检查是否在conda/mamba环境中（更宽松的环境检查）
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo -e "${YELLOW}⚠️  警告: 请确保在正确的conda/mamba环境中${NC}"
    echo "   建议运行: micromamba activate diffbev_py310 或 conda activate diffbev_py310"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ 当前环境: $CONDA_DEFAULT_ENV${NC}"
fi

# 设置环境变量以允许PyTorch 2.x（用于RTX 5090等新GPU）
export DIFFBEV_ALLOW_PYTORCH2=1
echo -e "${GREEN}✓ 已设置 DIFFBEV_ALLOW_PYTORCH2=1（允许PyTorch 2.x）${NC}"
echo ""

# 项目根目录（自动检测脚本所在目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
echo -e "${GREEN}✓ 项目根目录: $PROJECT_ROOT${NC}"

# 输出目录（确保存在）
OUTPUT_DIR="runs/diffbev_default"
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}输出目录: $OUTPUT_DIR${NC}"

# 配置文件
CONFIG_FILE="configs/diffbev/diffbev_lss_swin_nuscenes.py"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ 错误: 配置文件不存在: $CONFIG_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}配置文件: $CONFIG_FILE${NC}"
echo -e "${GREEN}输出目录: $OUTPUT_DIR${NC}"
echo ""

# 显示DiffBEV配置信息
echo -e "${BLUE}DiffBEV配置信息:${NC}"
echo "  - Diffusion模块: ${GREEN}启用${NC}"
echo "  - 条件输入类型: ${GREEN}FS-BEV${NC} (Full-Scale BEV)"
echo "  - 融合方式: ${GREEN}Cross-Attention${NC}"
echo "  - xt编码方式: ${GREEN}论文默认实现${NC}"
echo "  - 深度监督: ${GREEN}启用${NC} (权重=10.0)"
echo "  - Diffusion损失: ${GREEN}启用${NC} (权重=1.0)"
echo ""

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU信息:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | head -1
    echo ""
else
    echo -e "${YELLOW}⚠️  未检测到NVIDIA GPU或nvidia-smi命令不可用${NC}"
    echo ""
fi

# 设置日志文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/train_${TIMESTAMP}.log"

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}开始训练...${NC}"
echo -e "${BLUE}日志文件: $LOG_FILE${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# 训练命令
echo -e "${GREEN}开始训练DiffBEV模型（带Diffusion模块）...${NC}"
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
    echo "查看最佳模型:"
    echo "  ls -lh $OUTPUT_DIR/best_*.pth"
    echo ""
else
    echo ""
    echo -e "${RED}================================================================================${NC}"
    echo -e "${RED}❌ 训练失败！${NC}"
    echo -e "${RED}================================================================================${NC}"
    echo ""
    echo "请查看日志文件: $LOG_FILE"
    echo ""
    echo "常见问题排查:"
    echo "  1. 检查DiffusionHead是否已实现"
    echo "  2. 检查模型代码是否支持diffusion相关参数"
    echo "  3. 检查GPU内存是否足够（diffusion模块需要更多内存）"
    echo ""
    exit 1
fi
