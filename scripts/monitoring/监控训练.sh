#!/bin/bash
# 训练监控脚本
# 使用方法: bash 监控训练.sh

echo "=== DiffBEV 训练监控工具 ==="
echo ""

# 检查训练进程（改进的检测逻辑）
echo "1. 训练进程状态："

# 方法1: 检测train.py进程
TRAIN_PROC=$(ps aux | grep -E "python.*train\.py|train\.py" | grep -v grep)

# 方法2: 如果方法1失败，尝试更宽泛的匹配
if [ -z "$TRAIN_PROC" ]; then
    TRAIN_PROC=$(ps aux | grep -E "tools/train|train.*baseline" | grep -v grep)
fi

# 方法3: 通过日志文件的活动判断（如果日志最近更新过，说明训练可能在运行）
if [ -z "$TRAIN_PROC" ]; then
    WORK_DIR="/media/ldk950413/data0/DiffBEV/runs/baseline"
    LATEST_LOG=$(find "$WORK_DIR" -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -n "$LATEST_LOG" ]; then
        LOG_AGE=$(($(date +%s) - $(stat -c %Y "$LATEST_LOG" 2>/dev/null || echo 0)))
        if [ "$LOG_AGE" -lt 300 ]; then  # 5分钟内更新过
            echo "   ⚠️  未检测到训练进程，但日志文件最近更新（${LOG_AGE}秒前）"
            echo "   提示: 训练可能刚启动或进程名称不匹配"
            echo "   请检查: ps aux | grep train"
        else
            echo "   ❌ 未检测到训练进程（日志已${LOG_AGE}秒未更新）"
        fi
    else
        echo "   ❌ 未检测到训练进程（未找到日志文件）"
    fi
else
    PROC_COUNT=$(echo "$TRAIN_PROC" | wc -l)
    echo "   ✅ 训练进程正在运行 (检测到${PROC_COUNT}个相关进程)"
    echo "$TRAIN_PROC" | head -3 | sed 's/^/   /'
    
    # 显示进程详细信息
    if [ "$PROC_COUNT" -gt 1 ]; then
        echo "   (可能是主进程和worker进程)"
    fi
fi
echo ""

# 检查GPU使用情况
echo "2. GPU使用情况："
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv 2>/dev/null
else
    echo "   ⚠️  nvidia-smi 不可用"
fi
echo ""

# 查找最新的日志文件
WORK_DIR="/media/ldk950413/data0/DiffBEV/runs/baseline"
LATEST_LOG=$(find "$WORK_DIR" -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

if [ -z "$LATEST_LOG" ]; then
    # 尝试其他位置
    LATEST_LOG=$(find /media/ldk950413/data0/DiffBEV/runs -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
fi

if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
    echo "3. 最新日志文件: $LATEST_LOG"
    echo "   文件大小: $(du -h "$LATEST_LOG" 2>/dev/null | cut -f1)"
    echo "   最后修改: $(stat -c %y "$LATEST_LOG" 2>/dev/null | cut -d'.' -f1)"
    echo ""
    
    echo "4. 最新日志内容（最后30行）："
    echo "   ----------------------------------------"
    tail -30 "$LATEST_LOG" 2>/dev/null | sed 's/^/   /'
    echo "   ----------------------------------------"
    echo ""
    
    # 尝试提取迭代信息
    echo "5. 训练进度统计："
    if grep -q "iter" "$LATEST_LOG" 2>/dev/null; then
        LAST_ITER=$(grep -oP "iter[=\s]+(\d+)" "$LATEST_LOG" 2>/dev/null | tail -1 | grep -oP "\d+" | head -1 || echo "")
        TOTAL_ITERS=$(grep -oP "total_iters[=\s]+(\d+)" "$LATEST_LOG" 2>/dev/null | head -1 | grep -oP "\d+" | head -1 || echo "")
        if [ -n "$LAST_ITER" ] && [ -n "$TOTAL_ITERS" ]; then
            if command -v bc &> /dev/null; then
                PROGRESS=$(echo "scale=2; $LAST_ITER * 100 / $TOTAL_ITERS" | bc 2>/dev/null)
                echo "   当前迭代: $LAST_ITER / $TOTAL_ITERS (${PROGRESS}%)"
            else
                echo "   当前迭代: $LAST_ITER / $TOTAL_ITERS"
            fi
        fi
    fi
    
    # 提取损失信息
    LAST_LOSS=$(grep -i "loss" "$LATEST_LOG" 2>/dev/null | tail -1 | grep -oP "loss[:\s=]+([\d.]+)" | grep -oP "[\d.]+" | head -1 || echo "")
    if [ -n "$LAST_LOSS" ]; then
        echo "   最新损失: $LAST_LOSS"
    fi
else
    echo "3. ⚠️  未找到日志文件（可能训练刚开始）"
    echo "   检查目录: $WORK_DIR"
    ls -lt "$WORK_DIR"/*.log 2>/dev/null | head -3 || echo "   目录中无日志文件"
fi
echo ""

echo "=== 实时监控命令 ==="
if [ -n "$LATEST_LOG" ]; then
    echo "实时查看日志: tail -f $LATEST_LOG"
fi
echo "GPU监控: watch -n 1 nvidia-smi"
echo "进程监控: watch -n 1 'ps aux | grep train.py'"
