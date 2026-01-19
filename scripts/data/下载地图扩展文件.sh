#!/bin/bash
# 下载 nuScenes 地图扩展文件脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}nuScenes 地图扩展文件下载脚本${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""

# 下载目录
DOWNLOAD_DIR="/media/ldk950413/data0/nuScenes"
FILE_NAME="nuScenes-map-expansion-v1.3.zip"
ZENODO_URL="https://zenodo.org/record/15667707/files/nuScenes-map-expansion-v1.3.zip"

# 检查下载目录是否存在
if [ ! -d "$DOWNLOAD_DIR" ]; then
    echo -e "${RED}❌ 错误: 目录不存在: $DOWNLOAD_DIR${NC}"
    exit 1
fi

cd "$DOWNLOAD_DIR"

# 检查文件是否已下载
if [ -f "$FILE_NAME" ]; then
    echo -e "${GREEN}✓ 地图扩展文件已存在: $DOWNLOAD_DIR/$FILE_NAME${NC}"
    echo -e "  文件大小: $(du -h $FILE_NAME | cut -f1)"
    echo ""
    read -p "是否重新下载？(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "跳过下载"
        skip_download=true
    else
        skip_download=false
    fi
else
    skip_download=false
fi

# 下载文件
if [ "$skip_download" = false ]; then
    echo -e "${BLUE}下载信息:${NC}"
    echo "  文件名: $FILE_NAME"
    echo "  下载地址: $ZENODO_URL"
    echo "  保存位置: $DOWNLOAD_DIR/$FILE_NAME"
    echo "  文件大小: 约 398.5 MB"
    echo ""
    echo -e "${YELLOW}⚠️  注意: 地图扩展文件较大（约400MB），下载可能需要一些时间${NC}"
    echo ""
    read -p "是否继续下载？(Y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo "取消下载"
        exit 0
    fi
    
    echo ""
    echo "开始下载..."
    
    # 尝试使用 wget 或 curl 下载
    if command -v wget &> /dev/null; then
        wget -O "$FILE_NAME" "$ZENODO_URL" --progress=bar:force 2>&1
    elif command -v curl &> /dev/null; then
        curl -L -o "$FILE_NAME" "$ZENODO_URL" --progress-bar
    else
        echo -e "${RED}❌ 错误: 未找到 wget 或 curl，请手动下载:${NC}"
        echo "  URL: $ZENODO_URL"
        echo "  保存到: $DOWNLOAD_DIR/$FILE_NAME"
        exit 1
    fi
    
    # 检查下载是否成功
    if [ -f "$FILE_NAME" ]; then
        echo ""
        echo -e "${GREEN}✓ 下载成功！${NC}"
        echo "  文件: $DOWNLOAD_DIR/$FILE_NAME"
        echo "  大小: $(du -h $FILE_NAME | cut -f1)"
    else
        echo ""
        echo -e "${RED}❌ 下载失败${NC}"
        exit 1
    fi
fi

# 解压文件
echo ""
echo -e "${BLUE}解压文件...${NC}"

# 检查是否已解压
EXPANSION_DIR="$DOWNLOAD_DIR/maps/expansion"
if [ -d "$EXPANSION_DIR" ] && [ "$(ls -A $EXPANSION_DIR/*.json 2>/dev/null | wc -l)" -ge 4 ]; then
    echo -e "${GREEN}✓ 地图扩展文件已解压${NC}"
    echo "  目录: $EXPANSION_DIR"
    echo "  包含 $(ls -1 $EXPANSION_DIR/*.json 2>/dev/null | wc -l) 个 JSON 文件"
    echo ""
    read -p "是否重新解压？(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "跳过解压"
        skip_extract=true
    else
        skip_extract=false
    fi
else
    skip_extract=false
fi

if [ "$skip_extract" = false ]; then
    # 解压文件
    if command -v unzip &> /dev/null; then
        unzip -o "$FILE_NAME" -d "$DOWNLOAD_DIR"
    elif command -v 7z &> /dev/null; then
        7z x "$FILE_NAME" -o"$DOWNLOAD_DIR"
    else
        echo -e "${RED}❌ 错误: 未找到 unzip 或 7z，请手动解压:${NC}"
        echo "  文件: $DOWNLOAD_DIR/$FILE_NAME"
        echo "  解压到: $DOWNLOAD_DIR"
        exit 1
    fi
    
    # 验证解压结果
    # 检查文件是否解压到了正确位置，或者需要移动
    EXPANSION_DIR_WRONG="$DOWNLOAD_DIR/expansion"
    
    if [ -d "$EXPANSION_DIR" ]; then
        # 文件已在正确位置
        json_count=$(ls -1 $EXPANSION_DIR/*.json 2>/dev/null | wc -l)
        if [ "$json_count" -ge 4 ]; then
            echo ""
            echo -e "${GREEN}✓ 解压成功！${NC}"
            echo "  目录: $EXPANSION_DIR"
            echo "  包含 $json_count 个 JSON 文件"
            echo ""
            echo "文件列表:"
            ls -lh "$EXPANSION_DIR"/*.json 2>/dev/null | awk '{print "  "$9" ("$5")"}'
        else
            echo ""
            echo -e "${YELLOW}⚠️  警告: 解压后 JSON 文件数量不完整（期望4个，实际${json_count}个）${NC}"
        fi
    elif [ -d "$EXPANSION_DIR_WRONG" ]; then
        # 文件解压到了错误位置，需要移动到正确位置
        echo ""
        echo -e "${YELLOW}⚠️  检测到文件解压到了错误位置，正在移动...${NC}"
        mkdir -p "$EXPANSION_DIR"
        if [ -d "$EXPANSION_DIR_WRONG" ]; then
            mv "$EXPANSION_DIR_WRONG"/*.json "$EXPANSION_DIR/" 2>/dev/null
            # 如果 expansion 目录为空，删除它
            if [ -z "$(ls -A $EXPANSION_DIR_WRONG 2>/dev/null)" ]; then
                rmdir "$EXPANSION_DIR_WRONG" 2>/dev/null
            fi
        fi
        
        json_count=$(ls -1 $EXPANSION_DIR/*.json 2>/dev/null | wc -l)
        if [ "$json_count" -ge 4 ]; then
            echo -e "${GREEN}✓ 文件已移动到正确位置！${NC}"
            echo "  目录: $EXPANSION_DIR"
            echo "  包含 $json_count 个 JSON 文件"
            echo ""
            echo "文件列表:"
            ls -lh "$EXPANSION_DIR"/*.json 2>/dev/null | awk '{print "  "$9" ("$5")"}'
        else
            echo -e "${RED}❌ 错误: 移动后文件数量不足${NC}"
            exit 1
        fi
    else
        echo ""
        echo -e "${RED}❌ 错误: 解压后未找到 expansion 目录${NC}"
        echo "  请检查解压文件的结构"
        echo "  尝试查找 JSON 文件:"
        find "$DOWNLOAD_DIR" -maxdepth 2 -name "*.json" -type f 2>/dev/null | head -5
        exit 1
    fi
fi

# 最终验证
echo ""
echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}最终验证${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""

REQUIRED_FILES=(
    "boston-seaport.json"
    "singapore-onenorth.json"
    "singapore-queenstown.json"
    "singapore-hollandvillage.json"
)

all_exist=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$EXPANSION_DIR/$file" ]; then
        echo -e "${GREEN}✓ $file${NC}"
    else
        echo -e "${RED}✗ $file${NC}"
        all_exist=false
    fi
done

echo ""

if [ "$all_exist" = true ]; then
    echo -e "${GREEN}✓ 所有地图扩展文件已就绪！${NC}"
    echo ""
    echo "现在可以运行数据准备脚本："
    echo "  bash /media/ldk950413/data0/DiffBEV/prepare_nuscenes_bev_data.sh"
else
    echo -e "${YELLOW}⚠️  部分文件缺失，请检查解压结果${NC}"
fi
