#!/bin/bash
# Baseline实验运行脚本 - CPU模式（用于RTX 5090兼容性问题临时解决方案）

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}================================================================================${NC}"
echo -e "${YELLOW}⚠️  警告: 使用CPU模式训练（由于RTX 5090与PyTorch 1.9.1不兼容）${NC}"
echo -e "${YELLOW}================================================================================${NC}"
echo ""
echo -e "${YELLOW}注意:${NC}"
echo "  - CPU训练速度会很慢（可能需要数周完成200k iterations）"
echo "  - 建议仅用于测试代码流程"
echo "  - 长期解决方案：升级PyTorch到支持sm_120的版本"
echo ""

# 检查是否在diffbev环境中
if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" != "diffbev" ]]; then
    echo -e "${YELLOW}⚠️  警告: 请先激活 diffbev 环境${NC}"
    echo "   运行: micromamba activate diffbev"
    exit 1
fi

echo -e "${GREEN}✓ 当前环境: $CONDA_DEFAULT_ENV${NC}"
echo ""

# 隐藏GPU，强制使用CPU
export CUDA_VISIBLE_DEVICES=""
export FORCE_CPU=1
echo -e "${GREEN}✓ 已设置 CUDA_VISIBLE_DEVICES=\"\"（隐藏GPU）${NC}"
echo -e "${GREEN}✓ 已设置 FORCE_CPU=1（强制CPU模式）${NC}"
echo ""

# 项目根目录
PROJECT_ROOT="/media/ldk950413/data0/DiffBEV"
cd "$PROJECT_ROOT"

# 输出目录（确保存在）
OUTPUT_DIR="runs/baseline_cpu"
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
echo ""

# 设置日志文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/train_cpu_${TIMESTAMP}.log"

echo -e "${YELLOW}================================================================================${NC}"
echo -e "${YELLOW}开始CPU模式训练...${NC}"
echo -e "${YELLOW}日志文件: $LOG_FILE${NC}"
echo -e "${YELLOW}================================================================================${NC}"
echo ""

# 训练命令（使用CPU）
python tools/train.py "$CONFIG_FILE" \
    --work-dir "$OUTPUT_DIR" \
    --gpu-ids 0 \
    2>&1 | tee "$LOG_FILE"

# 检查训练是否成功
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================================================================${NC}"
    echo -e "${GREEN}✓ CPU训练完成！${NC}"
    echo -e "${GREEN}================================================================================${NC}"
    echo ""
    echo "结果保存在: $OUTPUT_DIR"
    echo "日志文件: $LOG_FILE"
else
    echo ""
    echo -e "${RED}================================================================================${NC}"
    echo -e "${RED}❌ CPU训练失败！${NC}"
    echo -e "${RED}================================================================================${NC}"
    echo ""
    echo "请查看日志文件: $LOG_FILE"
    exit 1
fi
