#!/bin/bash
# 极简版训练监控（只显示关键信息）
# 使用方法: watch -n 2 bash 简洁监控训练.sh

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

clear
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}              DiffBEV 训练监控  |  $(date '+%H:%M:%S')${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# 1. 进程状态（改进的检测逻辑）
# 方法1: 检测train.py进程
PROC=$(ps aux | grep -E "python.*train\.py|train\.py" | grep -v grep | head -1)

# 方法2: 如果方法1失败，尝试更宽泛的匹配
if [ -z "$PROC" ]; then
    PROC=$(ps aux | grep -E "tools/train|train.*baseline" | grep -v grep | head -1)
fi

# 方法3: 通过日志文件的活动判断
if [ -z "$PROC" ]; then
    WORK_DIR="/media/ldk950413/data0/DiffBEV/runs/baseline"
    LATEST_LOG=$(find "$WORK_DIR" -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -n "$LATEST_LOG" ]; then
        LOG_AGE=$(($(date +%s) - $(stat -c %Y "$LATEST_LOG" 2>/dev/null || echo 0)))
        if [ "$LOG_AGE" -lt 300 ]; then  # 5分钟内更新过
            echo -e "${YELLOW}⚠️  未检测到训练进程，但日志最近更新${NC}"
            echo -e "${YELLOW}   日志: ${LOG_AGE}秒前更新${NC}"
        else
            echo -e "${YELLOW}❌ 训练未运行${NC} (日志已${LOG_AGE}秒未更新)"
            exit 0
        fi
    else
        echo -e "${YELLOW}❌ 训练未运行${NC}"
        exit 0
    fi
fi

RUNTIME=$(echo "$PROC" | awk '{print $10}')
echo -e "${GREEN}✓ 训练运行中${NC}  |  已运行: ${RUNTIME}"
echo ""

# 2. GPU
GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$GPU" ]; then
    UTIL=$(echo "$GPU" | cut -d',' -f1 | xargs)
    MEM=$(echo "$GPU" | cut -d',' -f2 | xargs)
    MEM_T=$(echo "$GPU" | cut -d',' -f3 | xargs)
    TEMP=$(echo "$GPU" | cut -d',' -f4 | xargs)
    MEM_GB=$(echo "scale=1; $MEM / 1024" | bc 2>/dev/null)
    MEM_T_GB=$(echo "scale=1; $MEM_T / 1024" | bc 2>/dev/null)
    echo -e "${BLUE}GPU:${NC} ${UTIL}%  |  显存: ${MEM_GB}GB/${MEM_T_GB}GB  |  ${TEMP}°C"
fi
echo ""

# 3. 进度（从checkpoint或日志）
WORK_DIR="/media/ldk950413/data0/DiffBEV/runs/baseline"
CKPT=$(find "$WORK_DIR" -name "iter_*.pth" -type f 2>/dev/null | sort -V | tail -1)
LOG=$(find "$WORK_DIR" -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

if [ -n "$CKPT" ]; then
    ITER=$(basename "$CKPT" | grep -oE "[0-9]+" | head -1)
    CKPT_TIME=$(stat -c %y "$CKPT" 2>/dev/null | cut -d' ' -f2 | cut -d'.' -f1)
    echo -e "${BLUE}进度:${NC} 迭代 ${GREEN}${ITER}${NC}/200000  |  Checkpoint: ${CKPT_TIME}"
elif [ -n "$LOG" ]; then
    # 尝试从日志提取
    ITER=$(grep -oE "iter[\s:=]+[0-9]+" "$LOG" 2>/dev/null | grep -oE "[0-9]+" | tail -1)
    if [ -n "$ITER" ] && [ "$ITER" != "200000" ]; then
        PROG=$(echo "scale=1; $ITER * 100 / 200000" | bc 2>/dev/null)
        echo -e "${BLUE}进度:${NC} 迭代 ${GREEN}${ITER}${NC}/200000 (${PROG}%)"
    else
        # 提取loss和lr信息
        LOSS=$(grep -oE "loss=[0-9]+\.[0-9]+" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" || echo "")
        if [ -z "$LOSS" ]; then
            LOSS=$(grep -iE "(loss|l_loss)[\s:=]+[0-9]+\.[0-9]+" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
        fi
        LR=$(grep -oE "lr[\s:=]+[0-9]+\.[0-9]+e?-?[0-9]*" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+e?-?[0-9]*" || echo "")
        
        if [ -n "$LOSS" ]; then
            OUTPUT="${BLUE}损失:${NC} ${YELLOW}${LOSS}${NC}"
            if [ -n "$LR" ]; then
                OUTPUT="${OUTPUT}  |  lr: ${LR}"
            fi
            echo -e "${OUTPUT}  |  等待进度信息..."
        else
            echo -e "${YELLOW}等待训练日志更新...${NC} (每50次迭代记录)"
        fi
    fi
else
    echo -e "${YELLOW}等待训练开始...${NC}"
fi

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
