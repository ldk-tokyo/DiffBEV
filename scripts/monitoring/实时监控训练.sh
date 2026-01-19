#!/bin/bash
# 实时训练监控脚本（自动刷新版）
# 可以直接运行此脚本，它会自动刷新（使用watch或内置循环）

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# 刷新间隔（秒）
REFRESH_INTERVAL=${1:-2}

# 检查是否应该使用watch（如果通过watch调用，不要使用内置循环）
USE_BUILTIN_LOOP=true
if [ -n "$WATCH_SOURCE" ] || [ "$0" != "$BASH_SOURCE" ]; then
    USE_BUILTIN_LOOP=false
fi

# 获取脚本所在目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORK_DIR="$PROJECT_DIR/runs/baseline"

# 监控函数
monitor_training() {
    # 确保在项目根目录
    cd "$PROJECT_DIR" 2>/dev/null || {
        echo "错误: 无法切换到项目目录: $PROJECT_DIR"
        return 1
    }
    
    # 清屏
    clear
    
    # 获取当前时间
    CURRENT_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${WHITE}                    DiffBEV 训练实时监控${NC}              ${CURRENT_TIME}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # 方法1: 检测train.py进程（使用多种模式）
    TRAIN_PROC=""
    
    # 尝试模式1: python.*train.py 或 train.py
    TRAIN_PROC=$(ps aux | grep -E "[pP]ython.*train\.py|train\.py" | grep -v grep | grep -v "grep" | head -1)
    
    # 尝试模式2: tools/train 或包含train的进程
    if [ -z "$TRAIN_PROC" ]; then
        TRAIN_PROC=$(ps aux | grep -iE "tools/train|lss_swin|baseline" | grep -v grep | grep -v "grep" | head -1)
    fi
    
    # 尝试模式3: 通过进程命令行匹配
    if [ -z "$TRAIN_PROC" ]; then
        # 使用ps -eo pid,cmd 更可靠
        TRAIN_PROC=$(ps -eo pid,cmd | grep -iE "train\.py|tools/train" | grep -v grep | grep -v "grep" | head -1)
        if [ -n "$TRAIN_PROC" ]; then
            # 如果找到，转换为ps aux格式（需要获取完整信息）
            PID_FROM_CMD=$(echo "$TRAIN_PROC" | awk '{print $1}')
            if [ -n "$PID_FROM_CMD" ]; then
                TRAIN_PROC=$(ps aux | grep "^[^ ]* *$PID_FROM_CMD " | grep -v grep | head -1)
            fi
        fi
    fi
    
    # 方法4: 通过日志文件的活动判断（如果日志最近更新过，说明训练可能在运行）
    if [ -z "$TRAIN_PROC" ]; then
        LATEST_LOG=$(find "$WORK_DIR" -name "*.log" -type f -printf '%C@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        if [ -z "$LATEST_LOG" ]; then
            LATEST_LOG=$(find "$WORK_DIR" -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        fi
        
        if [ -n "$LATEST_LOG" ]; then
            LOG_AGE=$(($(date +%s) - $(stat -c %Y "$LATEST_LOG" 2>/dev/null || echo 0)))
            if [ "$LOG_AGE" -lt 300 ]; then  # 5分钟内更新过
                # 如果日志最近更新，尝试通过日志文件打开者找进程
                # 方法: 尝试通过lsof找打开日志文件的进程
                if command -v lsof &> /dev/null; then
                    LOG_PID=$(lsof "$LATEST_LOG" 2>/dev/null | grep -i python | awk '{print $2}' | head -1)
                    if [ -n "$LOG_PID" ]; then
                        TRAIN_PROC=$(ps aux | grep "^[^ ]* *$LOG_PID " | grep -v grep | head -1)
                    fi
                fi
                
                # 如果还是没找到，尝试最后一次努力：通过日志文件打开者找进程
                if [ -z "$TRAIN_PROC" ]; then
                    # 方法1: 使用lsof找打开日志文件的进程
                    if command -v lsof &> /dev/null && [ -n "$LATEST_LOG" ]; then
                        # 找打开.log文件的所有进程
                        LOG_PIDS=$(lsof "$LATEST_LOG" 2>/dev/null | tail -n +2 | awk '{print $2}' | sort -u)
                        if [ -n "$LOG_PIDS" ]; then
                            for LOG_PID in $LOG_PIDS; do
                                # 检查这个PID是否是Python进程
                                PROC_CMD=$(ps -p "$LOG_PID" -o cmd= 2>/dev/null | head -1)
                                if echo "$PROC_CMD" | grep -qiE "train|python"; then
                                    TRAIN_PROC=$(ps aux | grep "^[^ ]* *$LOG_PID " | grep -v grep | head -1)
                                    if [ -n "$TRAIN_PROC" ]; then
                                        break  # 找到了，跳出循环
                                    fi
                                fi
                            done
                        fi
                    fi
                    
                    # 方法2: 如果在WORK_DIR下找打开文件的进程
                    if [ -z "$TRAIN_PROC" ] && command -v lsof &> /dev/null && [ -n "$WORK_DIR" ]; then
                        # 找打开目录下.log文件的进程
                        LOG_PIDS=$(lsof "$WORK_DIR"/*.log 2>/dev/null | tail -n +2 | awk '{print $2}' | sort -u)
                        if [ -n "$LOG_PIDS" ]; then
                            for LOG_PID in $LOG_PIDS; do
                                PROC_CMD=$(ps -p "$LOG_PID" -o cmd= 2>/dev/null | head -1)
                                if echo "$PROC_CMD" | grep -qiE "train|python"; then
                                    TRAIN_PROC=$(ps aux | grep "^[^ ]* *$LOG_PID " | grep -v grep | head -1)
                                    if [ -n "$TRAIN_PROC" ]; then
                                        break
                                    fi
                                fi
                            done
                        fi
                    fi
                    
                    # 方法3: 如果日志最近更新（1分钟内），尝试找所有Python进程
                    if [ -z "$TRAIN_PROC" ] && [ "$LOG_AGE" -lt 60 ]; then
                        # 找所有Python进程，看哪个打开了日志文件
                        PYTHON_PIDS=$(ps aux | grep -E "[pP]ython" | grep -v grep | awk '{print $2}')
                        if [ -n "$PYTHON_PIDS" ] && command -v lsof &> /dev/null; then
                            for PYTHON_PID in $PYTHON_PIDS; do
                                # 检查这个进程是否打开了日志文件
                                if lsof -p "$PYTHON_PID" 2>/dev/null | grep -q "$LATEST_LOG"; then
                                    TRAIN_PROC=$(ps aux | grep "^[^ ]* *$PYTHON_PID " | grep -v grep | head -1)
                                    if [ -n "$TRAIN_PROC" ]; then
                                        break
                                    fi
                                fi
                            done
                        fi
                    fi
                    
                    # 如果还是没找到，显示警告但继续
                    if [ -z "$TRAIN_PROC" ]; then
                        echo -e "${YELLOW}⚠️  未检测到训练进程，但日志文件最近更新（${LOG_AGE}秒前）${NC}"
                        echo -e "${YELLOW}   提示: 训练可能刚启动或进程名称不匹配${NC}"
                        echo -e "${YELLOW}   提示: 正在通过日志文件继续监控...${NC}"
                        echo -e "${YELLOW}   建议: 运行 'bash 诊断训练进程.sh' 查看详细信息${NC}"
                        echo ""
                        # 不退出，继续显示日志信息
                        # 设置一个标志，表示使用日志模式
                        USE_LOG_MODE=true
                    else
                        USE_LOG_MODE=false
                    fi
                else
                    USE_LOG_MODE=false
                fi
            else
                echo -e "${RED}❌ 训练进程: 未运行${NC} (日志已${LOG_AGE}秒未更新)"
                echo ""
                return
            fi
        else
            echo -e "${RED}❌ 训练进程: 未运行${NC} (未找到日志文件)"
            echo ""
            return
        fi
    else
        USE_LOG_MODE=false
    fi
    
    # 显示进程信息（如果找到了）
    if [ -n "$TRAIN_PROC" ] && [ "$USE_LOG_MODE" != "true" ]; then
        CPU_USAGE=$(echo "$TRAIN_PROC" | awk '{print $3"%"}')
        MEM_USAGE=$(echo "$TRAIN_PROC" | awk '{print $4"%"}')
        CPU_TIME=$(echo "$TRAIN_PROC" | awk '{print $10}')  # ps aux的TIME字段：CPU累计使用时间
        PID=$(echo "$TRAIN_PROC" | awk '{print $2}')
        
        # 获取进程实际运行时间（从进程启动到现在）
        # 使用ps的etime字段获取进程实际运行时间
        ELAPSED_RUNTIME=""
        if [ -n "$PID" ]; then
            ETIME=$(ps -o etime= -p "$PID" 2>/dev/null | head -1 | xargs | tr -d ' ')
            if [ -n "$ETIME" ]; then
                # 解析etime格式
                if [[ "$ETIME" =~ ^([0-9]+)-([0-9]+):([0-9]+):([0-9]+)$ ]]; then
                    # 格式: DD-HH:MM:SS
                    DAYS="${BASH_REMATCH[1]}"
                    HOURS="${BASH_REMATCH[2]}"
                    MINS="${BASH_REMATCH[3]}"
                    if [ "$DAYS" -gt 0 ]; then
                        ELAPSED_RUNTIME="${DAYS}天${HOURS}小时${MINS}分钟"
                    elif [ "$HOURS" -gt 0 ]; then
                        ELAPSED_RUNTIME="${HOURS}小时${MINS}分钟"
                    else
                        ELAPSED_RUNTIME="${MINS}分钟"
                    fi
                elif [[ "$ETIME" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
                    # 格式: HH:MM:SS
                    HOURS="${BASH_REMATCH[1]}"
                    MINS="${BASH_REMATCH[2]}"
                    if [ "$HOURS" -gt 0 ]; then
                        ELAPSED_RUNTIME="${HOURS}小时${MINS}分钟"
                    else
                        ELAPSED_RUNTIME="${MINS}分钟"
                    fi
                elif [[ "$ETIME" =~ ^([0-9]+):([0-9]+)$ ]]; then
                    # 格式: MM:SS（运行时间<1小时）
                    MINS="${BASH_REMATCH[1]}"
                    ELAPSED_RUNTIME="${MINS}分钟"
                fi
            fi
        fi
        
        # 格式化CPU时间（ps aux的TIME字段，累计CPU使用时间）
        # ps aux的TIME字段始终是MM:SS格式（分钟:秒），即使是超过60分钟也是MM:SS（如120分钟显示为120:00）
        CPU_TIME_STR=""
        if [[ "$CPU_TIME" =~ ([0-9]+):([0-9]+) ]]; then
            CPU_MINS="${BASH_REMATCH[1]}"
            CPU_SECS="${BASH_REMATCH[2]}"
            # 如果分钟数>=60，转换为小时和分钟
            if [ "$CPU_MINS" -ge 60 ] 2>/dev/null; then
                CPU_HOURS=$((CPU_MINS / 60))
                CPU_MINS_REMAINING=$((CPU_MINS % 60))
                if [ "$CPU_HOURS" -gt 0 ]; then
                    CPU_TIME_STR="${CPU_HOURS}小时${CPU_MINS_REMAINING}分钟${CPU_SECS}秒"
                else
                    CPU_TIME_STR="${CPU_MINS}分钟${CPU_SECS}秒"
                fi
            else
                # 分钟数<60，直接显示分钟和秒
                if [ "$CPU_SECS" -gt 0 ] 2>/dev/null; then
                    CPU_TIME_STR="${CPU_MINS}分钟${CPU_SECS}秒"
                else
                    CPU_TIME_STR="${CPU_MINS}分钟"
                fi
            fi
        else
            CPU_TIME_STR="$CPU_TIME"
        fi
        
        echo -e "${GREEN}✅ 训练进程: 运行中${NC}  (PID: ${PID})"
        if [ -n "$ELAPSED_RUNTIME" ]; then
            echo -e "   CPU: ${CPU_USAGE}  |  内存: ${MEM_USAGE}  |  进程运行时间: ${ELAPSED_RUNTIME}  |  CPU累计时间: ${CPU_TIME_STR}"
        else
            echo -e "   CPU: ${CPU_USAGE}  |  内存: ${MEM_USAGE}  |  CPU累计时间: ${CPU_TIME_STR}"
        fi
    fi
    echo ""
    
    # 2. GPU状态
    # 如果使用日志模式，也需要获取PID用于时间计算
    if [ "$USE_LOG_MODE" = "true" ] && [ -n "$LOG_PID" ]; then
        PID="$LOG_PID"
    elif [ -n "$TRAIN_PROC" ]; then
        PID=$(echo "$TRAIN_PROC" | awk '{print $2}')
    fi
    
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ -n "$GPU_INFO" ]; then
            GPU_UTIL=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
            GPU_MEM_USED=$(echo "$GPU_INFO" | cut -d',' -f2 | xargs)
            GPU_MEM_TOTAL=$(echo "$GPU_INFO" | cut -d',' -f3 | xargs)
            GPU_TEMP=$(echo "$GPU_INFO" | cut -d',' -f4 | xargs)
            GPU_POWER=$(echo "$GPU_INFO" | cut -d',' -f5 | xargs)
            
            # 计算显存使用百分比
            if command -v bc &> /dev/null && [ -n "$GPU_MEM_USED" ] && [ -n "$GPU_MEM_TOTAL" ]; then
                GPU_MEM_PCT=$(echo "scale=1; $GPU_MEM_USED * 100 / $GPU_MEM_TOTAL" | bc 2>/dev/null)
                GPU_MEM_USED_GB=$(echo "scale=1; $GPU_MEM_USED / 1024" | bc 2>/dev/null)
                GPU_MEM_TOTAL_GB=$(echo "scale=1; $GPU_MEM_TOTAL / 1024" | bc 2>/dev/null)
            else
                GPU_MEM_PCT="0"
                GPU_MEM_USED_GB="0"
                GPU_MEM_TOTAL_GB="0"
            fi
            
            # 根据使用率设置颜色
            if [ "${GPU_UTIL%.*}" -ge 90 ]; then
                GPU_COLOR=$GREEN
            elif [ "${GPU_UTIL%.*}" -ge 50 ]; then
                GPU_COLOR=$YELLOW
            else
                GPU_COLOR=$RED
            fi
            
            echo -e "${BLUE}🎮 GPU状态 (RTX 5090):${NC}"
            echo -e "   使用率: ${GPU_COLOR}${GPU_UTIL}%${NC}  |  显存: ${GPU_MEM_USED_GB}GB/${GPU_MEM_TOTAL_GB}GB (${GPU_MEM_PCT}%)  |  温度: ${GPU_TEMP}°C  |  功耗: ${GPU_POWER}W"
        fi
    else
        echo -e "${YELLOW}⚠️  GPU信息: 无法获取${NC}"
    fi
    echo ""
    
    # 3. 训练进度和日志
    WORK_DIR="/media/ldk950413/data0/DiffBEV/runs/baseline"
    
    # 方法1: 优先按创建时间找最新日志（每次训练启动都会创建新日志文件，格式：YYYYMMDD_HHMMSS.log）
    # 这是最准确的方法，因为每次训练启动都会创建新的日志文件
    LATEST_LOG_BY_CREATE=$(find "$WORK_DIR" -name "*.log" -type f -printf '%C@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    
    # 方法2: 如果进程在运行，尝试从进程打开的文件中找（如果lsof可用）
    if [ -n "$PID" ] && command -v lsof &> /dev/null; then
        PROCESS_LOG=$(lsof -p "$PID" 2>/dev/null | grep "\.log$" | awk '{print $NF}' | grep "$WORK_DIR" | head -1)
        if [ -n "$PROCESS_LOG" ] && [ -f "$PROCESS_LOG" ]; then
            # 如果进程打开的文件比按创建时间找到的文件新，使用进程打开的文件
            if [ -n "$LATEST_LOG_BY_CREATE" ]; then
                PROCESS_CREATE_TIME=$(stat -c %Z "$PROCESS_LOG" 2>/dev/null || echo 0)
                LATEST_CREATE_TIME=$(stat -c %Z "$LATEST_LOG_BY_CREATE" 2>/dev/null || echo 0)
                if [ "$PROCESS_CREATE_TIME" -gt "$LATEST_CREATE_TIME" ]; then
                    LATEST_LOG="$PROCESS_LOG"
                else
                    LATEST_LOG="$LATEST_LOG_BY_CREATE"
                fi
            else
                LATEST_LOG="$PROCESS_LOG"
            fi
        else
            LATEST_LOG="$LATEST_LOG_BY_CREATE"
        fi
    else
        LATEST_LOG="$LATEST_LOG_BY_CREATE"
    fi
    
    # 方法3: 如果还是没找到，按修改时间找
    if [ -z "$LATEST_LOG" ]; then
        LATEST_LOG_BY_MODIFY=$(find "$WORK_DIR" -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        LATEST_LOG="$LATEST_LOG_BY_MODIFY"
    fi
    
    # 方法4: 扩大搜索范围
    if [ -z "$LATEST_LOG" ]; then
        LATEST_LOG=$(find /media/ldk950413/data0/DiffBEV/runs -name "*.log" -type f -printf '%C@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    fi
    
    echo -e "${BLUE}📊 训练进度:${NC}"
    
    if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
        LOG_TIME=$(stat -c %y "$LATEST_LOG" 2>/dev/null | cut -d'.' -f1 | cut -d' ' -f2 || echo "")
        LOG_TIME_STAMP=$(stat -c %Y "$LATEST_LOG" 2>/dev/null || echo 0)
        LOG_CREATE_STAMP=$(stat -c %Z "$LATEST_LOG" 2>/dev/null || echo 0)
        CURRENT_TIME_STAMP=$(date +%s)
        if [ "$LOG_TIME_STAMP" -gt 0 ]; then
            LOG_AGE_MIN=$(( ($CURRENT_TIME_STAMP - $LOG_TIME_STAMP) / 60 ))
            LOG_AGE_SEC=$(( $CURRENT_TIME_STAMP - $LOG_TIME_STAMP ))
        else
            LOG_AGE_MIN=0
            LOG_AGE_SEC=0
        fi
        
        # 检查日志文件是否在最近更新（如果进程在运行但日志很久没更新，可能是输出到了其他地方）
        LOG_IS_STALE=false
        if [ "$LOG_AGE_MIN" -gt 10 ] && [ -n "$TRAIN_PROC" ]; then
            LOG_IS_STALE=true
        fi
        
        # 检查是否有多个训练进程（可能是父子进程）
        PROC_COUNT=$(ps aux | grep -iE "[pP]ython.*train\.py|train\.py|tools/train" | grep -v grep | grep -v "grep" | wc -l)
        
        # 提取迭代信息（尝试多种格式）
        LAST_ITER=""
        
        # 方法1: 从tqdm进度条或实际训练输出中查找 (iter=123 或 Training iter=123)
        LAST_ITER=$(grep -oE "iter=[0-9]+" "$LATEST_LOG" 2>/dev/null | grep -oE "[0-9]+" | tail -1 || echo "")
        
        # 方法2: 从日志记录格式中查找 (iter: 123 或 iter 123)，但排除配置中的max_iters
        if [ -z "$LAST_ITER" ] || [ "$LAST_ITER" = "$TOTAL_ITERS" ]; then
            # 尝试找到实际的训练迭代数（应该小于total_iters）
            ITER_CANDIDATES=$(grep -oE "(iter|iteration)[\s:=]+[0-9]+" "$LATEST_LOG" 2>/dev/null | grep -oE "[0-9]+" | grep -v "^${TOTAL_ITERS}$" | tail -1 || echo "")
            if [ -n "$ITER_CANDIDATES" ] && [ "$ITER_CANDIDATES" -lt "$TOTAL_ITERS" ] 2>/dev/null; then
                LAST_ITER="$ITER_CANDIDATES"
            fi
        fi
        
        # 方法3: 从checkpoint文件名中提取（但只在日志中没找到时才使用）
        if [ -z "$LAST_ITER" ] || [ "$LAST_ITER" = "$TOTAL_ITERS" ]; then
            LATEST_CKPT=$(find "$WORK_DIR" -name "iter_*.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
            if [ -n "$LATEST_CKPT" ]; then
                CKPT_ITER=$(basename "$LATEST_CKPT" | grep -oE "[0-9]+" | head -1)
                if [ -n "$CKPT_ITER" ] && [ "$CKPT_ITER" -lt "$TOTAL_ITERS" ] 2>/dev/null; then
                    LAST_ITER="$CKPT_ITER"
                fi
            fi
        fi
        
        # 如果仍然没有找到，且日志很新（5分钟内更新），说明可能刚开始训练或日志刚创建
        if [ -z "$LAST_ITER" ] && [ "$LOG_AGE_MIN" -lt 5 ]; then
            LAST_ITER="0"
        fi
        
        # 总迭代数（从配置或日志中获取）
        TOTAL_ITERS="200000"  # 默认值，从配置文件中获取
        if grep -q "total_iters\|max_iters" "$LATEST_LOG" 2>/dev/null; then
            TOTAL_ITERS_TMP=$(grep -oE "(total_iters|max_iters)[\s:=]+[0-9]+" "$LATEST_LOG" 2>/dev/null | grep -oE "[0-9]+" | head -1)
            if [ -n "$TOTAL_ITERS_TMP" ]; then
                TOTAL_ITERS="$TOTAL_ITERS_TMP"
            fi
        fi
        
        # 提取损失值（尝试多种格式）
        # 方法1: 从tqdm进度条格式提取 (Training loss=0.1234 ETA=1.5h)
        LAST_LOSS=$(grep -oE "loss=[0-9]+\.[0-9]+" "$LATEST_LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" || echo "")
        
        # 方法2: 从标准日志格式提取 (loss: 0.1234 或 loss=0.1234)
        if [ -z "$LAST_LOSS" ]; then
            LAST_LOSS=$(grep -iE "(loss|l_loss|loss_seg)[\s:=]+[0-9]+\.[0-9]+" "$LATEST_LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1 || echo "")
        fi
        
        # 方法3: 从JSON格式日志中提取
        if [ -z "$LAST_LOSS" ]; then
            LAST_LOSS=$(grep -oE "\"loss\":[0-9]+\.[0-9]+" "$LATEST_LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" || echo "")
        fi
        
        # 提取各种loss分量（如果存在）
        LOSS_SEG=$(grep -oE "(loss_seg|l_seg|seg_loss)[\s:=]+[0-9]+\.[0-9]+" "$LATEST_LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1 || echo "")
        LOSS_DEPTH=$(grep -oE "(loss_depth|depth_loss|Ldepth)[\s:=]+[0-9]+\.[0-9]+" "$LATEST_LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1 || echo "")
        LOSS_DIFF=$(grep -oE "(loss_diff|diff_loss|Ldiff)[\s:=]+[0-9]+\.[0-9]+" "$LATEST_LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1 || echo "")
        
        # 提取学习率（尝试多种格式）
        # 方法1: 从tqdm或标准日志格式提取
        LAST_LR=$(grep -oE "lr[\s:=]+[0-9]+\.[0-9]+e?-?[0-9]*" "$LATEST_LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+e?-?[0-9]*" || echo "")
        
        # 方法2: 从learning_rate字段提取
        if [ -z "$LAST_LR" ]; then
            LAST_LR=$(grep -oE "(learning_rate|lr_rate)[\s:=]+[0-9]+\.[0-9]+e?-?[0-9]*" "$LATEST_LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+e?-?[0-9]*" || echo "")
        fi
        
        # 提取评估指标（如果有）
        LAST_MIOU=$(grep -oE "mIoU[\s:=]+[0-9]+\.[0-9]+" "$LATEST_LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+\.[0-9]+" || echo "")
        
        # 从tqdm进度条中提取ETA
        ETA_INFO=$(grep -oE "ETA=[0-9]+\.[0-9]+[hm]" "$LATEST_LOG" 2>/dev/null | tail -1 || echo "")
        
        # 显示信息
        # 如果日志很新（5分钟内）且没有找到迭代数，说明可能刚开始训练
        if [ -z "$LAST_ITER" ] || [ "$LAST_ITER" = "$TOTAL_ITERS" ]; then
            if [ "$LOG_AGE_MIN" -lt 5 ]; then
                echo -e "   状态: ${YELLOW}训练正在进行中...${NC}"
                echo -e "   提示: 日志文件已创建，等待训练输出迭代信息"
                LAST_ITER=""  # 清除无效的迭代数
            elif [ "$LOG_IS_STALE" = true ]; then
                echo -e "   状态: ${YELLOW}⚠️ 训练进程运行中（${PROC_COUNT}个进程），但日志已${LOG_AGE_MIN}分钟未更新${NC}"
                echo -e "   提示: 训练可能在初始化、数据加载或等待中"
                echo -e "   提示: 如果训练刚启动，日志可能需要等待第一个checkpoint或评估点才会更新"
                if [ "$PROC_COUNT" -gt 1 ]; then
                    echo -e "   提示: 检测到${PROC_COUNT}个训练相关进程，可能是主进程和worker进程"
                fi
                LAST_ITER=""  # 清除无效的迭代数
            else
                echo -e "   状态: ${YELLOW}等待训练日志更新...${NC} (日志: ${LOG_AGE_MIN}分钟前更新)"
                LAST_ITER=""  # 清除无效的迭代数
            fi
        elif [ -n "$LAST_ITER" ] && [ "$LAST_ITER" != "0" ] 2>/dev/null; then
            # 计算进度
            if command -v bc &> /dev/null && [ "$LAST_ITER" -gt 0 ] 2>/dev/null && [ "$TOTAL_ITERS" -gt 0 ] 2>/dev/null; then
                PROGRESS=$(echo "scale=1; $LAST_ITER * 100 / $TOTAL_ITERS" | bc 2>/dev/null)
                REMAINING=$(echo "$TOTAL_ITERS - $LAST_ITER" | bc 2>/dev/null)
                PROGRESS_BAR_LEN=30
                FILLED=$(echo "scale=0; $LAST_ITER * $PROGRESS_BAR_LEN / $TOTAL_ITERS" | bc 2>/dev/null || echo "0")
                [ -z "$FILLED" ] && FILLED=0
                
                # 创建进度条
                PROGRESS_BAR=""
                for i in $(seq 1 $PROGRESS_BAR_LEN); do
                    if [ "$i" -le "$FILLED" ]; then
                        PROGRESS_BAR="${PROGRESS_BAR}█"
                    else
                        PROGRESS_BAR="${PROGRESS_BAR}░"
                    fi
                done
                
                echo -e "   迭代: ${GREEN}${LAST_ITER}${NC} / ${TOTAL_ITERS}  (${PROGRESS}%)"
                echo -e "   [${GREEN}${PROGRESS_BAR}${NC}]"
                echo -e "   剩余: ${REMAINING} 次迭代"
                
                # 计算总耗时和预计完成时间
                ELAPSED_SECONDS=0
                PROC_START_TIME=""
                
                # 方法1: 使用ps的etime（最可靠）
                if [ -n "$PID" ]; then
                    ETIME=$(ps -o etime= -p "$PID" 2>/dev/null | head -1 | xargs | tr -d ' ')
                    if [ -n "$ETIME" ]; then
                        # 解析etime格式
                        if [[ "$ETIME" =~ ^([0-9]+)-([0-9]+):([0-9]+):([0-9]+)$ ]]; then
                            # 格式: DD-HH:MM:SS
                            DAYS="${BASH_REMATCH[1]}"
                            HOURS="${BASH_REMATCH[2]}"
                            MINS="${BASH_REMATCH[3]}"
                            SECS="${BASH_REMATCH[4]}"
                            ELAPSED_SECONDS=$((DAYS * 86400 + HOURS * 3600 + MINS * 60 + SECS))
                        elif [[ "$ETIME" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
                            # 格式: HH:MM:SS
                            HOURS="${BASH_REMATCH[1]}"
                            MINS="${BASH_REMATCH[2]}"
                            SECS="${BASH_REMATCH[3]}"
                            ELAPSED_SECONDS=$((HOURS * 3600 + MINS * 60 + SECS))
                        elif [[ "$ETIME" =~ ^([0-9]+):([0-9]+)$ ]]; then
                            # 格式可能是 HH:MM 或 MM:SS
                            # ps的etime格式规则：
                            # - 如果运行时间<1小时：格式为 MM:SS
                            # - 如果运行时间>=1小时但<1天：格式为 HH:MM:SS
                            # - 如果运行时间>=1天：格式为 DD-HH:MM:SS
                            # 所以如果是 XX:XX 格式（只有两个冒号分隔的数字），应该是 MM:SS
                            FIRST_NUM="${BASH_REMATCH[1]}"
                            SECOND_NUM="${BASH_REMATCH[2]}"
                            
                            # 判断逻辑：
                            # 1. 如果第一个数字>=60，绝对是MM:SS（分钟数不能>=60）
                            # 2. 如果第二个数字>=60，绝对是MM:SS（秒数不能>=60）
                            # 3. 如果第一个数字<60，且第二个数字<60，很可能是MM:SS（因为HH:MM格式在etime中不常见）
                            # 实际上，ps etime的XX:XX格式始终是MM:SS
                            if [ "$FIRST_NUM" -ge 60 ] 2>/dev/null || [ "$SECOND_NUM" -ge 60 ] 2>/dev/null; then
                                # 绝对错误，应该是MM:SS但数据有问题，按MM:SS处理
                                MINS="$FIRST_NUM"
                                SECS="$SECOND_NUM"
                                ELAPSED_SECONDS=$((MINS * 60 + SECS))
                            else
                                # 两个数字都<60，按MM:SS处理（ps etime的XX:XX格式就是MM:SS）
                                MINS="$FIRST_NUM"
                                SECS="$SECOND_NUM"
                                ELAPSED_SECONDS=$((MINS * 60 + SECS))
                            fi
                        fi
                    fi
                fi
                
                # 方法2: 如果etime获取失败，尝试使用进程启动时间
                if [ "$ELAPSED_SECONDS" -eq 0 ] && [ -n "$PID" ]; then
                    # 尝试使用/proc/PID（Linux系统）
                    if [ -f "/proc/$PID" ]; then
                        PROC_START_TIME=$(stat -c %Z "/proc/$PID" 2>/dev/null || echo "")
                        if [ -n "$PROC_START_TIME" ] && [ "$PROC_START_TIME" -gt 0 ] 2>/dev/null; then
                            CURRENT_TIME_STAMP=$(date +%s)
                            ELAPSED_SECONDS=$((CURRENT_TIME_STAMP - PROC_START_TIME))
                        fi
                    fi
                    
                    # 尝试使用ps的lstart
                    if [ "$ELAPSED_SECONDS" -eq 0 ]; then
                        PS_START=$(ps -o lstart= -p "$PID" 2>/dev/null | head -1 | xargs)
                        if [ -n "$PS_START" ]; then
                            PROC_START_TIME=$(date -d "$PS_START" +%s 2>/dev/null || echo "")
                            if [ -n "$PROC_START_TIME" ] && [ "$PROC_START_TIME" -gt 0 ] 2>/dev/null; then
                                CURRENT_TIME_STAMP=$(date +%s)
                                ELAPSED_SECONDS=$((CURRENT_TIME_STAMP - PROC_START_TIME))
                            fi
                        fi
                    fi
                fi
                
                # 方法3: 如果都失败，使用日志文件创建时间
                if [ "$ELAPSED_SECONDS" -eq 0 ] && [ -n "$LATEST_LOG" ] && [ "$LOG_CREATE_STAMP" -gt 0 ] 2>/dev/null; then
                    CURRENT_TIME_STAMP=$(date +%s)
                    ELAPSED_SECONDS=$((CURRENT_TIME_STAMP - LOG_CREATE_STAMP))
                fi
                
                # 计算总耗时和预计完成时间
                if [ "$ELAPSED_SECONDS" -gt 0 ] 2>/dev/null; then
                    CURRENT_TIME_STAMP=$(date +%s)
                    
                    # 格式化总耗时
                    ELAPSED_HOURS=$((ELAPSED_SECONDS / 3600))
                    ELAPSED_MINS=$(((ELAPSED_SECONDS % 3600) / 60))
                    ELAPSED_SECS=$((ELAPSED_SECONDS % 60))
                    
                    if [ "$ELAPSED_HOURS" -gt 0 ]; then
                        ELAPSED_STR="${ELAPSED_HOURS}小时${ELAPSED_MINS}分钟"
                    elif [ "$ELAPSED_MINS" -gt 0 ]; then
                        ELAPSED_STR="${ELAPSED_MINS}分钟${ELAPSED_SECS}秒"
                    else
                        ELAPSED_STR="${ELAPSED_SECONDS}秒"
                    fi
                    
                    # 计算迭代速度（迭代/秒）
                    # 优先使用最近一段时间的速度（更准确），如果无法计算则使用平均速度
                    ITER_PER_SEC=0
                    RECENT_SPEED=0
                    
                    if [ -n "$LATEST_LOG" ] && [ "$LAST_ITER" -gt 0 ] 2>/dev/null; then
                        # 尝试从日志中提取最近的迭代记录，计算最近的速度
                        # 获取最近N条迭代记录（例如最近10条）
                        RECENT_ITERS=$(grep -oE "iter=[0-9]+" "$LATEST_LOG" 2>/dev/null | grep -oE "[0-9]+" | tail -10)
                        if [ -n "$RECENT_ITERS" ]; then
                            # 计算最近迭代的时间范围
                            # 方法：从日志中提取最近几条迭代记录的时间戳
                            FIRST_RECENT_ITER=$(echo "$RECENT_ITERS" | head -1)
                            LAST_RECENT_ITER=$(echo "$RECENT_ITERS" | tail -1)
                            
                            # 获取第一条和最后一条迭代记录的时间戳
                            FIRST_ITER_TIME=$(grep "iter=$FIRST_RECENT_ITER" "$LATEST_LOG" 2>/dev/null | head -1 | grep -oE "[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}" | head -1)
                            LAST_ITER_TIME=$(grep "iter=$LAST_RECENT_ITER" "$LATEST_LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}" | head -1)
                            
                            if [ -n "$FIRST_ITER_TIME" ] && [ -n "$LAST_ITER_TIME" ] && [ "$FIRST_ITER_TIME" != "$LAST_ITER_TIME" ]; then
                                # 计算时间差（秒）
                                FIRST_TIMESTAMP=$(date -d "$FIRST_ITER_TIME" +%s 2>/dev/null || echo "0")
                                LAST_TIMESTAMP=$(date -d "$LAST_ITER_TIME" +%s 2>/dev/null || echo "0")
                                
                                if [ "$FIRST_TIMESTAMP" -gt 0 ] && [ "$LAST_TIMESTAMP" -gt 0 ] && [ "$FIRST_TIMESTAMP" -lt "$LAST_TIMESTAMP" ]; then
                                    RECENT_TIME_SPAN=$((LAST_TIMESTAMP - FIRST_TIMESTAMP))
                                    RECENT_ITER_SPAN=$((LAST_RECENT_ITER - FIRST_RECENT_ITER))
                                    
                                    if [ "$RECENT_TIME_SPAN" -gt 0 ] && [ "$RECENT_ITER_SPAN" -gt 0 ]; then
                                        # 计算最近的速度（迭代/秒）
                                        RECENT_SPEED=$(echo "scale=4; $RECENT_ITER_SPAN / $RECENT_TIME_SPAN" | bc 2>/dev/null || echo "0")
                                    fi
                                fi
                            fi
                        fi
                    fi
                    
                    # 如果无法计算最近速度，或最近速度明显异常（太小），使用平均速度
                    if [ "$ELAPSED_SECONDS" -gt 0 ] && [ "$LAST_ITER" -gt 0 ] 2>/dev/null; then
                        AVG_ITER_PER_SEC=$(echo "scale=4; $LAST_ITER / $ELAPSED_SECONDS" | bc 2>/dev/null || echo "0")
                        
                        # 如果最近速度有效且合理（不为0且不太小），使用最近速度
                        # 否则使用平均速度
                        if [ "$(echo "$RECENT_SPEED > 0" | bc 2>/dev/null || echo "0")" = "1" ] && \
                           [ "$(echo "$RECENT_SPEED >= $AVG_ITER_PER_SEC * 0.5" | bc 2>/dev/null || echo "0")" = "1" ]; then
                            ITER_PER_SEC="$RECENT_SPEED"
                            SPEED_TYPE="最近速度"
                        else
                            ITER_PER_SEC="$AVG_ITER_PER_SEC"
                            SPEED_TYPE="平均速度"
                        fi
                    else
                        ITER_PER_SEC=0
                    fi
                    
                    # 计算剩余时间（秒）
                    REMAINING_SECONDS=0
                    if [ "$(echo "$ITER_PER_SEC > 0" | bc 2>/dev/null || echo "0")" = "1" ] && [ "$REMAINING" -gt 0 ] 2>/dev/null; then
                        REMAINING_SECONDS=$(echo "scale=0; $REMAINING / $ITER_PER_SEC" | bc 2>/dev/null || echo "0")
                    fi
                    
                    # 格式化剩余时间并显示
                    if [ "$REMAINING_SECONDS" -gt 0 ] 2>/dev/null; then
                        REMAINING_HOURS=$((REMAINING_SECONDS / 3600))
                        REMAINING_MINS=$(((REMAINING_SECONDS % 3600) / 60))
                        
                        if [ "$REMAINING_HOURS" -gt 0 ]; then
                            REMAINING_TIME_STR="${REMAINING_HOURS}小时${REMAINING_MINS}分钟"
                        elif [ "$REMAINING_MINS" -gt 0 ]; then
                            REMAINING_TIME_STR="${REMAINING_MINS}分钟"
                        else
                            REMAINING_TIME_STR="${REMAINING_SECONDS}秒"
                        fi
                        
                        # 计算预计完成时间
                        ETA_TIMESTAMP=$((CURRENT_TIME_STAMP + REMAINING_SECONDS))
                        ETA_DATE=$(date -d "@$ETA_TIMESTAMP" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date '+%Y-%m-%d %H:%M:%S')
                        
                        # 计算总耗时（已运行时间 + 剩余时间）
                        TOTAL_TRAINING_SECONDS=$((ELAPSED_SECONDS + REMAINING_SECONDS))
                        TOTAL_HOURS=$((TOTAL_TRAINING_SECONDS / 3600))
                        TOTAL_MINS=$(((TOTAL_TRAINING_SECONDS % 3600) / 60))
                        
                        if [ "$TOTAL_HOURS" -gt 0 ]; then
                            TOTAL_TIME_STR="${TOTAL_HOURS}小时${TOTAL_MINS}分钟"
                        elif [ "$TOTAL_MINS" -gt 0 ]; then
                            TOTAL_TIME_STR="${TOTAL_MINS}分钟"
                        else
                            TOTAL_TIME_STR="${TOTAL_TRAINING_SECONDS}秒"
                        fi
                        
                        # 显示时间信息
                        echo -e "${BLUE}⏱️  训练时间:${NC}"
                        echo -e "   已运行时间: ${CYAN}${ELAPSED_STR}${NC}"
                        echo -e "   剩余时间: ${CYAN}${REMAINING_TIME_STR}${NC} (基于${SPEED_TYPE})"
                        echo -e "   总耗时: ${GREEN}${TOTAL_TIME_STR}${NC} (从开始到完成预计总时间)"
                        echo -e "   预计完成: ${GREEN}${ETA_DATE}${NC}"
                        echo -e "   速度: ${YELLOW}$(echo "scale=2; $ITER_PER_SEC * 60" | bc 2>/dev/null)${NC} 迭代/分钟 (${SPEED_TYPE})"
                        # 如果使用了平均速度，显示提示
                        if [ "$SPEED_TYPE" = "平均速度" ] && [ "$ELAPSED_SECONDS" -gt 3600 ]; then
                            echo -e "   ${YELLOW}提示: 平均速度可能受初始化阶段影响，实际训练速度可能更快${NC}"
                        fi
                    else
                        echo -e "${BLUE}⏱️  训练时间:${NC} 已运行时间: ${CYAN}${ELAPSED_STR}${NC} (无法计算剩余时间)"
                    fi
                else
                    echo -e "${BLUE}⏱️  训练时间:${NC} 已运行时间: ${CYAN}${ELAPSED_STR}${NC} (无法计算剩余时间)"
                fi
            else
                echo -e "   迭代: ${GREEN}${LAST_ITER}${NC} / ${TOTAL_ITERS}"
            fi
        else
            # 如果日志很新（5分钟内），说明可能刚开始训练
            if [ "$LOG_AGE_MIN" -lt 5 ]; then
                echo -e "   状态: ${YELLOW}初始化中或刚开始训练...${NC}"
                echo -e "   提示: 日志每50次迭代记录一次，请稍候"
            elif [ "$LOG_IS_STALE" = true ]; then
                echo -e "   状态: ${YELLOW}⚠️ 训练进程运行中（${PROC_COUNT}个进程），但日志已${LOG_AGE_MIN}分钟未更新${NC}"
                echo -e "   提示: 训练可能在初始化、数据加载或等待中"
                echo -e "   提示: 如果训练刚启动，日志可能需要等待第一个checkpoint或评估点才会更新"
                if [ "$PROC_COUNT" -gt 1 ]; then
                    echo -e "   提示: 检测到${PROC_COUNT}个训练相关进程，可能是主进程和worker进程"
                fi
            else
                echo -e "   状态: ${YELLOW}等待训练日志更新...${NC} (日志: ${LOG_AGE_MIN}分钟前更新)"
            fi
        fi
        
        # 显示训练指标
        if [ -n "$LAST_LOSS" ] || [ -n "$LOSS_SEG" ] || [ -n "$LOSS_DEPTH" ]; then
            echo -e "${BLUE}📈 训练指标:${NC}"
            if [ -n "$LAST_LOSS" ]; then
                echo -e "   总损失: ${YELLOW}${LAST_LOSS}${NC}"
            fi
            if [ -n "$LOSS_SEG" ]; then
                echo -e "   分割损失: ${YELLOW}${LOSS_SEG}${NC}"
            fi
            if [ -n "$LOSS_DEPTH" ]; then
                echo -e "   深度损失: ${YELLOW}${LOSS_DEPTH}${NC}"
            fi
            if [ -n "$LOSS_DIFF" ]; then
                echo -e "   扩散损失: ${YELLOW}${LOSS_DIFF}${NC}"
            fi
        fi
        
        if [ -n "$LAST_LR" ]; then
            echo -e "${BLUE}⚙️  学习率:${NC} ${CYAN}${LAST_LR}${NC}"
        fi
        
        if [ -n "$LAST_MIOU" ]; then
            echo -e "${BLUE}🎯 评估指标:${NC} mIoU=${GREEN}${LAST_MIOU}${NC}"
        fi
        
        if [ -n "$ETA_INFO" ]; then
            echo -e "${BLUE}⏱️  ${ETA_INFO}${NC}"
        fi
        
        echo -e "   日志文件: ${LATEST_LOG##*/}  (${LOG_AGE_MIN}分钟前更新)"
    else
        echo -e "   ${YELLOW}⚠️  日志文件未找到${NC}"
    fi
    echo ""
    
    # 4. Checkpoint信息
    LATEST_CKPT=$(find "$WORK_DIR" -name "*.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -n "$LATEST_CKPT" ]; then
        CKPT_TIME=$(stat -c %y "$LATEST_CKPT" 2>/dev/null | cut -d'.' -f1 | cut -d' ' -f2 || echo "")
        CKPT_SIZE=$(du -h "$LATEST_CKPT" 2>/dev/null | cut -f1 || echo "")
        CKPT_NAME=$(basename "$LATEST_CKPT")
        
        # 从checkpoint文件名提取迭代数
        CKPT_ITER=$(echo "$CKPT_NAME" | grep -oE "[0-9]+" | head -1 || echo "")
        
        if [ -n "$CKPT_ITER" ]; then
            echo -e "${BLUE}💾 最新Checkpoint:${NC} ${CKPT_NAME}  (迭代: ${CKPT_ITER}, ${CKPT_SIZE}, ${CKPT_TIME})"
        else
            echo -e "${BLUE}💾 最新Checkpoint:${NC} ${CKPT_NAME}  (${CKPT_SIZE}, ${CKPT_TIME})"
        fi
    else
        echo -e "${YELLOW}💾 Checkpoint: 尚未保存${NC} (每5000次迭代保存一次)"
    fi
    
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
    
    # 显示刷新提示（只在内置循环模式下显示）
    if [ "$USE_BUILTIN_LOOP" = true ]; then
        echo -e "${WHITE}按 Ctrl+C 停止监控 | 刷新间隔: ${REFRESH_INTERVAL}秒${NC}"
    fi
}

# 主程序
if [ "$USE_BUILTIN_LOOP" = true ] && [ -t 0 ]; then
    # 使用内置循环（如果是在交互式终端中）
    while true; do
        monitor_training
        sleep "$REFRESH_INTERVAL"
    done
else
    # 单次执行（如果通过watch调用或非交互式终端）
    monitor_training
fi
