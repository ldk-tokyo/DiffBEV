#!/bin/bash
echo "=== 数据准备状态检查 ==="
echo ""
echo "1. 依赖检查:"
python -c "import yacs, shapely, nuscenes; print('  ✓ 所有依赖已安装')" 2>&1 || echo "  ✗ 缺少依赖"
echo ""
echo "2. 符号链接:"
test -L /media/ldk950413/data0/nuscenes && echo "  ✓ 符号链接已创建" || echo "  ✗ 符号链接未创建"
echo ""
echo "3. 地图文件:"
if [ -d "/media/ldk950413/data0/nuScenes/maps/expansion" ]; then
    count=$(ls -1 /media/ldk950413/data0/nuScenes/maps/expansion/*.json 2>/dev/null | wc -l)
    if [ "$count" -ge 4 ]; then
        echo "  ✓ 地图文件存在 ($count 个文件)"
    else
        echo "  ⚠️  地图文件不完整 ($count/4 个文件)"
    fi
else
    echo "  ✗ 地图文件目录不存在"
    echo "    需要下载: v1.0-trainval_map_expansion.tgz"
fi
echo ""
echo "4. 数据集原始数据:"
for dir in samples sweeps v1.0-trainval; do
    if [ -d "/media/ldk950413/data0/nuScenes/$dir" ]; then
        echo "  ✓ $dir"
    else
        echo "  ✗ $dir"
    fi
done
