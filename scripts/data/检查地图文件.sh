#!/bin/bash
echo "=== nuScenes 地图文件检查 ==="
echo ""

MAPS_DIR="/media/ldk950413/data0/nuScenes/maps"
EXPANSION_DIR="$MAPS_DIR/expansion"

echo "1. maps 目录:"
if [ -d "$MAPS_DIR" ]; then
    echo "   ✓ 存在: $MAPS_DIR"
    file_count=$(ls -1 "$MAPS_DIR"/*.png 2>/dev/null | wc -l)
    echo "   包含 $file_count 个 PNG 文件（地图图片）"
else
    echo "   ✗ 不存在"
fi

echo ""
echo "2. expansion 目录:"
if [ -d "$EXPANSION_DIR" ]; then
    echo "   ✓ 存在: $EXPANSION_DIR"
    json_count=$(ls -1 "$EXPANSION_DIR"/*.json 2>/dev/null | wc -l)
    echo "   包含 $json_count 个 JSON 文件"
    if [ "$json_count" -ge 4 ]; then
        echo "   ✓ 地图文件完整"
        echo ""
        echo "   文件列表:"
        ls -lh "$EXPANSION_DIR"/*.json 2>/dev/null | awk '{print "     "$9" ("$5")"}'
    else
        echo "   ⚠️  地图文件不完整（需要 4 个，当前 $json_count 个）"
    fi
else
    echo "   ✗ 不存在: $EXPANSION_DIR"
    echo ""
    echo "   需要下载: nuScenes-map-expansion-v1.3.zip"
    echo "   下载地址: https://zenodo.org/record/15667707"
    echo "   文件大小: 约 398.5 MB"
    echo "   解压命令: cd /media/ldk950413/data0/nuScenes && unzip nuScenes-map-expansion-v1.3.zip"
    echo ""
    echo "   注意: 地图扩展文件不在 nuScenes 官网的 Metadata 包中，需要从 Zenodo 单独下载"
fi

echo ""
echo "3. 需要的文件:"
required_files=(
    "boston-seaport.json"
    "singapore-onenorth.json"
    "singapore-queenstown.json"
    "singapore-hollandvillage.json"
)
for file in "${required_files[@]}"; do
    if [ -f "$EXPANSION_DIR/$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ✗ $file"
    fi
done

