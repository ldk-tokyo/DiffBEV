#!/bin/bash
# nuScenes BEV数据准备脚本
# 使用 mono-semantic-maps 项目生成 BEV 标注

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}nuScenes BEV数据准备脚本${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""

# 检查是否在diffbev环境中
if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" != "diffbev" ]]; then
    echo -e "${YELLOW}⚠️  警告: 请先激活 diffbev 环境${NC}"
    echo "   运行: micromamba activate diffbev"
    exit 1
fi

echo -e "${GREEN}✓ 当前环境: $CONDA_DEFAULT_ENV${NC}"
echo ""

# 项目根目录
PROJECT_ROOT="/media/ldk950413/data0/DiffBEV"
cd "$PROJECT_ROOT"

# 数据集路径
NUSCENES_DATA_ROOT="/media/ldk950413/data0/nuScenes"
MONO_SEMANTIC_MAPS_DIR="$PROJECT_ROOT/mono-semantic-maps"

# 检查 mono-semantic-maps 目录是否存在
if [ ! -d "$MONO_SEMANTIC_MAPS_DIR" ]; then
    echo -e "${RED}❌ 错误: mono-semantic-maps 目录不存在${NC}"
    echo "   请先克隆 mono-semantic-maps 项目:"
    echo "   git clone https://github.com/tom-roddick/mono-semantic-maps.git"
    exit 1
fi

# 检查 nuScenes 数据集
if [ ! -d "$NUSCENES_DATA_ROOT" ]; then
    echo -e "${RED}❌ 错误: nuScenes 数据集目录不存在: $NUSCENES_DATA_ROOT${NC}"
    exit 1
fi

# 检查必要的目录
if [ ! -d "$NUSCENES_DATA_ROOT/v1.0-trainval" ]; then
    echo -e "${RED}❌ 错误: nuScenes v1.0-trainval 目录不存在${NC}"
    exit 1
fi

echo -e "${GREEN}数据集路径: $NUSCENES_DATA_ROOT${NC}"
echo -e "${GREEN}mono-semantic-maps 目录: $MONO_SEMANTIC_MAPS_DIR${NC}"
echo ""

# 检查依赖
echo "检查依赖..."
python -c "import yacs; import nuscenes; import shapely; print('✓ 所有依赖已安装')" 2>&1 || {
    echo -e "${YELLOW}⚠️  缺少依赖，正在安装...${NC}"
    pip install yacs shapely python-dotenv 2>&1 | tail -3
}

echo ""

# 设置环境变量
export DATA_ROOT="/media/ldk950413/data0"
export PROCESSED_ROOT="$NUSCENES_DATA_ROOT"

echo -e "${GREEN}环境变量:${NC}"
echo "  DATA_ROOT=$DATA_ROOT"
echo "  PROCESSED_ROOT=$PROCESSED_ROOT"
echo ""

# 进入 mono-semantic-maps 目录
cd "$MONO_SEMANTIC_MAPS_DIR"

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}开始生成 BEV 标注...${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""

# 运行脚本
python scripts/make_nuscenes_labels.py

# 检查输出
OUTPUT_DIR="$PROCESSED_ROOT/map-labels-v1.2"
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR 2>/dev/null)" ]; then
    echo ""
    echo -e "${GREEN}================================================================================${NC}"
    echo -e "${GREEN}✓ BEV 标注生成成功！${NC}"
    echo -e "${GREEN}================================================================================${NC}"
    echo ""
    echo "输出目录: $OUTPUT_DIR"
    echo "生成的文件数量: $(ls -1 $OUTPUT_DIR | wc -l)"
    echo ""
    echo "下一步: 需要将标注文件组织到 img_dir 和 ann_bev_dir 目录"
    echo "  参考: 数据集准备说明.md"
else
    echo ""
    echo -e "${YELLOW}⚠️  警告: 输出目录为空或不存在${NC}"
    echo "  输出目录: $OUTPUT_DIR"
    echo "  请检查脚本执行日志"
fi
