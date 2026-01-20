#!/bin/bash
# 实时绘制训练loss曲线脚本
# 监控metrics.csv文件的变化，自动更新loss曲线图

# 不使用 set -e，因为我们需要处理依赖安装失败的情况

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}实时Loss曲线绘制工具${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# 解析参数
WORK_DIR="${1:-runs/baseline}"
OUTPUT="${2:-loss_curves.png}"
INTERVAL="${3:-10}"

# 检查work_dir是否存在
if [ ! -d "$WORK_DIR" ]; then
    echo -e "${YELLOW}⚠️  目录不存在: $WORK_DIR${NC}"
    echo ""
    echo "用法: $0 [work_dir] [output_filename] [interval_seconds]"
    echo ""
    echo "示例:"
    echo "  $0 runs/baseline                    # 监控runs/baseline/metrics.csv"
    echo "  $0 runs/diffbev_default loss.png 5  # 监控runs/diffbev_default，输出loss.png，5秒刷新"
    exit 1
fi

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${GREEN}监控目录: $WORK_DIR${NC}"
echo -e "${GREEN}输出文件: $WORK_DIR/$OUTPUT${NC}"
echo -e "${GREEN}刷新间隔: ${INTERVAL}秒${NC}"
echo ""

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo -e "${YELLOW}⚠️  未找到python命令，尝试python3...${NC}"
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

# 检查是否在conda/mamba环境中
if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    echo -e "${GREEN}✓ 检测到conda/mamba环境: $CONDA_DEFAULT_ENV${NC}"
    PIP_CMD="$PYTHON_CMD -m pip"
elif command -v micromamba &> /dev/null; then
    echo -e "${YELLOW}⚠️  检测到micromamba，但未激活环境${NC}"
    echo -e "${YELLOW}   建议运行: micromamba activate diffbev_py310${NC}"
    PIP_CMD="$PYTHON_CMD -m pip"
elif command -v conda &> /dev/null; then
    echo -e "${YELLOW}⚠️  检测到conda，但未激活环境${NC}"
    echo -e "${YELLOW}   建议运行: conda activate diffbev_py310${NC}"
    PIP_CMD="$PYTHON_CMD -m pip"
else
    echo -e "${YELLOW}⚠️  未检测到conda/mamba环境${NC}"
    echo -e "${YELLOW}   如果使用系统Python，可能需要使用 --break-system-packages${NC}"
    PIP_CMD="$PYTHON_CMD -m pip"
fi

# 检查依赖
if ! $PYTHON_CMD -c "import pandas, matplotlib" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  缺少依赖包，正在安装...${NC}"
    
    # 尝试安装，如果失败则提示用户
    if $PIP_CMD install pandas matplotlib -q 2>/dev/null; then
        echo -e "${GREEN}✓ 依赖包安装成功${NC}"
    else
        echo -e "${YELLOW}⚠️  自动安装失败，请手动安装依赖包：${NC}"
        echo ""
        echo "如果在conda/mamba环境中："
        echo "  $PIP_CMD install pandas matplotlib"
        echo ""
        echo "如果使用系统Python（不推荐）："
        echo "  $PIP_CMD install --break-system-packages pandas matplotlib"
        echo ""
        echo "或者使用conda安装："
        echo "  conda install pandas matplotlib"
        echo ""
        read -p "是否尝试使用 --break-system-packages 安装? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            $PIP_CMD install --break-system-packages pandas matplotlib
        else
            echo -e "${YELLOW}请手动安装依赖包后重试${NC}"
            exit 1
        fi
    fi
else
    echo -e "${GREEN}✓ 依赖包已安装${NC}"
fi

echo -e "${GREEN}开始监控...${NC}"
echo -e "${GREEN}按 Ctrl+C 停止${NC}"
echo ""

# 运行绘制脚本（不指定--max-iter，使用自动缩放）
$PYTHON_CMD tools/plot_loss_realtime.py \
    --work-dir "$WORK_DIR" \
    --output "$OUTPUT" \
    --interval "$INTERVAL"
