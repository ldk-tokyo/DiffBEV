#!/bin/bash
# 诊断训练进程检测问题
# 使用方法: bash 诊断训练进程.sh

echo "=== 训练进程检测诊断 ==="
echo ""

echo "1. 检查进程（方式1: ps aux | grep train）:"
ps aux | grep -E "[pP]ython.*train|train\.py" | grep -v grep | grep -v "grep" | head -3
echo ""

echo "2. 检查进程（方式2: ps -eo pid,cmd | grep train）:"
ps -eo pid,cmd | grep -iE "train\.py|tools/train" | grep -v grep | head -3
echo ""

echo "3. 检查进程（方式3: 通过PID目录）:"
PID=$(ps aux | grep -E "[pP]ython.*train|train\.py" | grep -v grep | head -1 | awk '{print $2}')
if [ -n "$PID" ]; then
    echo "找到PID: $PID"
    echo "进程信息:"
    ps -p "$PID" -f 2>/dev/null | head -2
    echo ""
    echo "进程启动时间:"
    if [ -f "/proc/$PID" ]; then
        echo "  /proc/$PID 存在"
        stat -c %Z "/proc/$PID" 2>/dev/null | xargs -I {} date -d "@{}" 2>/dev/null || echo "  无法读取"
    fi
else
    echo "未找到PID"
fi
echo ""

echo "4. 检查日志文件:"
WORK_DIR="/media/ldk950413/data0/DiffBEV/runs/baseline"
LATEST_LOG=$(find "$WORK_DIR" -name "*.log" -type f -printf '%C@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
if [ -n "$LATEST_LOG" ]; then
    echo "最新日志: $LATEST_LOG"
    LOG_AGE=$(($(date +%s) - $(stat -c %Y "$LATEST_LOG" 2>/dev/null || echo 0)))
    echo "日志更新时间: ${LOG_AGE}秒前"
    
    if command -v lsof &> /dev/null; then
        echo ""
        echo "5. 检查打开日志文件的进程:"
        lsof "$LATEST_LOG" 2>/dev/null | head -5
    fi
else
    echo "未找到日志文件"
fi
echo ""

echo "=== 诊断完成 ==="
