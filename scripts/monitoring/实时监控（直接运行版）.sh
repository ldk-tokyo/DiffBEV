#!/bin/bash
# 实时训练监控（直接运行版 - 内置自动刷新）
# 直接运行此脚本即可看到实时监控，会自动刷新
# 使用方法: bash 实时监控（直接运行版）.sh [刷新间隔，默认2秒]

# 刷新间隔（秒）
REFRESH_INTERVAL=${1:-2}

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 获取项目根目录（scripts/monitoring的父目录的父目录）
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 颜色提示
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}DiffBEV 实时训练监控${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}刷新间隔: ${REFRESH_INTERVAL}秒${NC}"
echo -e "${YELLOW}按 Ctrl+C 停止监控${NC}"
echo ""
sleep 2

# 调用监控脚本，使用内置循环模式
cd "$SCRIPT_DIR"
MONITOR_SCRIPT="$SCRIPT_DIR/实时监控训练.sh"
# 如果脚本不在当前目录，尝试从scripts/monitoring查找
if [ ! -f "$MONITOR_SCRIPT" ]; then
    MONITOR_SCRIPT="$(dirname "$SCRIPT_DIR")/monitoring/实时监控训练.sh"
fi
bash "$MONITOR_SCRIPT" "$REFRESH_INTERVAL"
