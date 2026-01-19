#!/bin/bash
# 启动实时监控（自动选择最佳方式）
# 使用方法: bash 启动实时监控.sh [刷新间隔秒数，默认2秒]

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 获取项目根目录（scripts/monitoring的父目录的父目录）
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

REFRESH_INTERVAL=${1:-2}
MONITOR_SCRIPT="$SCRIPT_DIR/实时监控训练.sh"

# 检查监控脚本是否存在
if [ ! -f "$MONITOR_SCRIPT" ]; then
    echo "错误: 找不到监控脚本: $MONITOR_SCRIPT"
    exit 1
fi

cd "$PROJECT_DIR"

# 检查watch命令是否可用
if command -v watch &> /dev/null; then
    echo "使用watch命令进行实时监控（每${REFRESH_INTERVAL}秒刷新）"
    echo "按 Ctrl+C 停止监控"
    echo ""
    watch -n "$REFRESH_INTERVAL" -c bash "$MONITOR_SCRIPT"
else
    echo "watch命令不可用，使用内置循环模式（每${REFRESH_INTERVAL}秒刷新）"
    echo "按 Ctrl+C 停止监控"
    echo ""
    # 使用内置循环
    while true; do
        bash "$MONITOR_SCRIPT" "$REFRESH_INTERVAL"
        sleep "$REFRESH_INTERVAL"
    done
fi
