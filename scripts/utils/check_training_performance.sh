#!/bin/bash
# 检查训练性能瓶颈的脚本

echo "=========================================="
echo "训练性能诊断工具"
echo "=========================================="

# 1. 检查CPU核心数
echo ""
echo "1. CPU信息："
CPU_CORES=$(nproc)
echo "   CPU核心数: $CPU_CORES"
echo "   建议 workers_per_gpu: $((CPU_CORES / 2)) - $((CPU_CORES / 4))"

# 2. 检查GPU利用率
echo ""
echo "2. GPU状态："
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | \
    awk -F', ' '{printf "   GPU %s (%s): 使用率=%s%%, 显存=%s/%s MB\n", $1, $2, $3, $4, $5}'

# 3. 检查内存和Swap
echo ""
echo "3. 内存状态："
free -h | grep -E "Mem|Swap" | awk '{printf "   %s: %s\n", $1, $2}'
SWAP_USED=$(free | grep Swap | awk '{print $3}')
if [ -n "$SWAP_USED" ] && [ "$SWAP_USED" -gt 0 ] 2>/dev/null; then
    echo "   ⚠️  警告：Swap被使用，可能导致训练变慢！"
else
    echo "   ✓ Swap未使用"
fi

# 4. 检查训练进程
echo ""
echo "4. 训练进程状态："
TRAIN_PID=$(pgrep -f "python.*train.py" | head -1)
if [ -n "$TRAIN_PID" ]; then
    echo "   训练进程PID: $TRAIN_PID"
    echo "   CPU使用率: $(ps -p $TRAIN_PID -o %cpu --no-headers)%"
    echo "   内存使用: $(ps -p $TRAIN_PID -o rss --no-headers | awk '{printf "%.1f MB", $1/1024}')"
    
    # 检查进程的线程数（worker数量）
    THREAD_COUNT=$(ps -T -p $TRAIN_PID | wc -l)
    echo "   线程数（包括workers）: $((THREAD_COUNT - 1))"
else
    echo "   ⚠️  未找到训练进程"
fi

# 5. 检查数据目录IO
echo ""
echo "5. 数据目录IO状态："
DATA_DIR="/media/ldk950413/data0/nuScenes"
if [ -d "$DATA_DIR" ]; then
    echo "   数据目录: $DATA_DIR"
    echo "   目录大小: $(du -sh $DATA_DIR 2>/dev/null | cut -f1)"
    echo "   文件数量: $(find $DATA_DIR/img_dir/train -type f 2>/dev/null | wc -l) (train)"
else
    echo "   ⚠️  数据目录不存在: $DATA_DIR"
fi

# 6. 检查最近的训练日志
echo ""
echo "6. 最近的训练速度："
LATEST_LOG=$(find runs -name "*.log" -type f -printf '%C@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
    echo "   日志文件: $LATEST_LOG"
    echo "   最近10次迭代的平均速度:"
    tail -100 "$LATEST_LOG" 2>/dev/null | grep -oE "[0-9]+\.[0-9]+iter/s" | tail -10 | \
        awk '{sum+=$1; count++} END {if(count>0) printf "   %.2f iter/s\n", sum/count; else print "   无法提取速度信息"}'
    
    # 检查速度波动
    echo "   速度波动分析:"
    tail -100 "$LATEST_LOG" 2>/dev/null | grep -oE "[0-9]+\.[0-9]+iter/s" | tail -20 | \
        awk '{
            speeds[NR]=$1
            sum+=$1
            count++
        }
        END {
            if(count>0) {
                avg=sum/count
                min=speeds[1]
                max=speeds[1]
                for(i=1;i<=count;i++) {
                    if(speeds[i]<min) min=speeds[i]
                    if(speeds[i]>max) max=speeds[i]
                }
                printf "   平均: %.2f iter/s, 最小: %.2f iter/s, 最大: %.2f iter/s\n", avg, min, max
                if((max-min)/avg > 0.5) {
                    printf "   ⚠️  速度波动较大（>50%%），可能存在性能瓶颈\n"
                }
            }
        }'
else
    echo "   ⚠️  未找到训练日志"
fi

echo ""
echo "=========================================="
echo "诊断完成"
echo "=========================================="
echo ""
echo "优化建议："
echo "1. 如果workers_per_gpu < CPU核心数/4，考虑增加workers"
echo "2. 如果Swap被使用，考虑减少batch_size或workers"
echo "3. 如果GPU利用率 < 80%，可能是数据加载瓶颈"
echo "4. 如果速度波动 > 50%，检查checkpoint保存和评估配置"
echo ""
