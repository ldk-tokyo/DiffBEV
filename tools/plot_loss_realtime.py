#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时绘制训练loss曲线
监控metrics.csv文件的变化，自动更新loss曲线图
"""
import os
import sys
import argparse
import time
import shutil
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
from pathlib import Path


def load_metrics(csv_path):
    """加载metrics.csv文件
    处理混合格式的CSV文件：
    - 训练数据行：step,loss,lr (3个字段)
    - 评估数据行：step,其他评估指标... (18个字段)
    只读取训练数据行用于绘制loss曲线
    """
    if not os.path.exists(csv_path):
        return None
    
    try:
        # 先读取文件内容，手动处理格式问题
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 找到训练数据的表头行（step,loss,lr格式）
        header_line_idx = None
        train_header = None
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            # 查找训练数据的表头（包含step,loss,lr）
            if 'step' in line_lower and 'loss' in line_lower and 'lr' in line_lower:
                fields = line.strip().split(',')
                if len(fields) == 3:  # 训练数据表头应该是3个字段
                    header_line_idx = i
                    train_header = [f.strip() for f in fields]
                    break
        
        if header_line_idx is None or train_header is None:
            print("WARNING: Training data header (step,loss,lr) not found")
            return None
        
        # 读取训练数据行（允许有多余列，只取前3列）
        data_lines = []
        for line in lines[header_line_idx + 1:]:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            
            # 分割字段
            fields = line.split(',')
            fields = [f.strip() for f in fields]
            
            # 允许行包含多列（评估会扩展CSV列），只取前3列作为训练数据
            if len(fields) >= 3:
                # 检查前3列是否都是数字（step, loss, lr）
                try:
                    float(fields[0])
                    float(fields[1])
                    float(fields[2])
                    data_lines.append(fields[:3])
                except ValueError:
                    # 如果不是数字，可能是另一个表头行，跳过
                    continue
        
        if not data_lines:
            return None
        
        # 创建DataFrame
        df = pd.DataFrame(data_lines, columns=train_header)
        
        # 确保step列是数值类型
        if 'step' in df.columns:
            df['step'] = pd.to_numeric(df['step'], errors='coerce')
            df = df.dropna(subset=['step'])
            df = df.sort_values('step')
            df = df.reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"WARNING: Failed to read CSV file: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_loss_curves(df, output_path, title="Training Loss Curves", max_iter=None):
    """绘制loss曲线"""
    if df is None or len(df) == 0:
        print("WARNING: No data to plot")
        return False
    
    # 清除之前的图形，确保每次都是全新绘制
    plt.close('all')
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 获取step列
    steps = df['step'].values if 'step' in df.columns else range(len(df))
    
    # 确定x轴范围（自动缩放）
    if len(steps) > 0:
        min_step = int(steps.min())
        max_step = int(steps.max())
        
        if max_iter is not None:
            # 如果指定了最大迭代次数，使用它作为上限，但不小于当前数据最大值
            x_max = max(max_iter, max_step + 1000)
        else:
            # 自动缩放：根据数据范围调整，留适当边距
            # 计算数据范围
            data_range = max_step - min_step
            if data_range == 0:
                # 如果只有一个数据点，设置一个合理的范围
                x_max = max_step + 100
            else:
                # 添加5%的边距，但至少1000个单位
                margin = max(int(data_range * 0.05), 1000)
                x_max = max_step + margin
        
        # x轴最小值，留一点左边距
        x_min = max(0, min_step - max(int((max_step - min_step) * 0.01), 50))
    else:
        # 如果没有数据，使用默认值
        x_min = 0
        x_max = max_iter if max_iter is not None else 20000
    
    # 1. 总loss曲线
    ax1 = axes[0, 0]
    if 'loss' in df.columns:
        loss_values = pd.to_numeric(df['loss'], errors='coerce')
        ax1.plot(steps, loss_values, 'b-', linewidth=2, label='Total Loss')
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Total Loss', fontsize=14, fontweight='bold')
        ax1.set_xlim(x_min, x_max)  # 设置x轴范围
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # 2. 各项loss曲线（Lwce, Ldepth, Ldiff）
    ax2 = axes[0, 1]
    loss_components = []
    if 'Lwce' in df.columns:
        lwce = pd.to_numeric(df['Lwce'], errors='coerce')
        if not lwce.isna().all():  # 检查是否有有效数据
            ax2.plot(steps, lwce, 'g-', linewidth=2, label='Lwce', alpha=0.8)
            loss_components.append('Lwce')
    if 'Ldepth' in df.columns:
        ldepth = pd.to_numeric(df['Ldepth'], errors='coerce')
        if not ldepth.isna().all():  # 检查是否有有效数据
            ax2.plot(steps, ldepth, 'r-', linewidth=2, label='Ldepth', alpha=0.8)
            loss_components.append('Ldepth')
    if 'Ldiff' in df.columns:
        ldiff = pd.to_numeric(df['Ldiff'], errors='coerce')
        if not ldiff.isna().all():  # 检查是否有有效数据
            ax2.plot(steps, ldiff, 'm-', linewidth=2, label='Ldiff', alpha=0.8)
            loss_components.append('Ldiff')
    
    if loss_components:
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Loss Components', fontsize=14, fontweight='bold')
        ax2.set_xlim(x_min, x_max)  # 设置x轴范围
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        # 使用对数刻度（如果loss值范围很大）
        if len(loss_components) > 0:
            try:
                ax2.set_yscale('log')
            except:
                pass
    else:
        # 如果没有Loss Components数据，显示提示信息（使用英文避免字体问题）
        ax2.axis('off')
        info_text = [
            "Loss Components",
            "=" * 40,
            "",
            "No loss component data",
            "found in CSV file.",
            "",
            "CSV columns:",
            f"  {', '.join(df.columns.tolist())}",
            "",
            "Possible reasons:",
            "1. Baseline config only logs total loss",
            "2. Training code doesn't log Lwce/Ldepth/Ldiff",
            "",
            "To enable Loss Components:",
            "Check log_training_losses() call",
            "in training code"
        ]
        ax2.text(0.5, 0.5, '\n'.join(info_text),
                transform=ax2.transAxes,
                fontsize=10,
                verticalalignment='center',
                horizontalalignment='center',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 3. 学习率曲线
    ax3 = axes[1, 0]
    if 'lr' in df.columns:
        lr_values = pd.to_numeric(df['lr'], errors='coerce')
        ax3.plot(steps, lr_values, 'orange', linewidth=2, label='Learning Rate')
        ax3.set_xlabel('Iteration', fontsize=12)
        ax3.set_ylabel('Learning Rate', fontsize=12)
        ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax3.set_xlim(x_min, x_max)  # 设置x轴范围
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        # 使用对数刻度
        try:
            ax3.set_yscale('log')
        except:
            pass
    
    # 4. Loss统计信息（最近N个iter的平均值）
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 计算统计信息（使用英文避免字体问题）
    stats_text = []
    stats_text.append("Loss Statistics")
    stats_text.append("=" * 40)
    
    if 'loss' in df.columns:
        loss_values = pd.to_numeric(df['loss'], errors='coerce')
        if len(loss_values) > 0:
            stats_text.append(f"Total Loss:")
            stats_text.append(f"  Current: {loss_values.iloc[-1]:.6f}")
            stats_text.append(f"  Mean: {loss_values.mean():.6f}")
            stats_text.append(f"  Min: {loss_values.min():.6f}")
            stats_text.append(f"  Max: {loss_values.max():.6f}")
            
            # 最近100个iter的平均值
            if len(loss_values) >= 100:
                recent_mean = loss_values.iloc[-100:].mean()
                stats_text.append(f"  Recent 100 iter mean: {recent_mean:.6f}")
    
    stats_text.append("")
    
    # 各项loss的统计
    for comp in ['Lwce', 'Ldepth', 'Ldiff']:
        if comp in df.columns:
            comp_values = pd.to_numeric(df[comp], errors='coerce')
            if len(comp_values) > 0:
                stats_text.append(f"{comp}:")
                stats_text.append(f"  Current: {comp_values.iloc[-1]:.6f}")
                stats_text.append(f"  Mean: {comp_values.mean():.6f}")
    
    stats_text.append("")
    stats_text.append(f"Iterations: {len(df)}")
    if 'step' in df.columns and len(df) > 0:
        stats_text.append(f"Current Step: {df['step'].iloc[-1]}")
    
    ax4.text(0.1, 0.9, '\n'.join(stats_text), 
             transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存图片（使用临时文件名，然后重命名，确保原子性写入）
    # 注意：临时文件必须使用正确的图片扩展名，否则matplotlib无法识别格式
    temp_path = str(output_path).replace('.png', '.tmp.png').replace('.jpg', '.tmp.jpg')
    if temp_path == str(output_path):
        # 如果没有扩展名，添加.tmp.png
        temp_path = str(output_path) + '.tmp.png'
    plt.savefig(temp_path, dpi=150, bbox_inches='tight', facecolor='white', format='png')
    plt.close('all')  # 关闭所有图形，释放内存
    
    # 原子性重命名，确保图片文件完整
    if os.path.exists(output_path):
        os.remove(output_path)  # 删除旧文件
    shutil.move(temp_path, output_path)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='实时绘制训练loss曲线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 监控指定目录的metrics.csv
  python tools/plot_loss_realtime.py --work-dir runs/baseline
  
  # 指定输出图片路径和刷新间隔
  python tools/plot_loss_realtime.py --work-dir runs/baseline --output loss_curves.png --interval 5
  
  # 后台运行（推荐）
  nohup python tools/plot_loss_realtime.py --work-dir runs/baseline > plot_loss.log 2>&1 &
        """
    )
    
    parser.add_argument(
        '--work-dir',
        type=str,
        required=True,
        help='训练输出目录（包含metrics.csv）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='loss_curves.png',
        help='输出图片文件名（默认: loss_curves.png）'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=10,
        help='刷新间隔（秒，默认: 10）'
    )
    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='图表标题（默认: 使用work-dir名称）'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='只绘制一次，不循环更新'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=None,
        help='最大迭代次数（可选，用于设置x轴上限；默认: 根据数据自动缩放）'
    )
    
    args = parser.parse_args()
    
    # 检查work_dir是否存在
    work_dir = Path(args.work_dir)
    if not work_dir.exists():
        print(f"❌ 错误: 目录不存在: {work_dir}")
        sys.exit(1)
    
    csv_path = work_dir / 'metrics.csv'
    output_path = work_dir / args.output
    
    # 设置标题
    title = args.title if args.title else f"Training Loss - {work_dir.name}"
    
    print(f"Real-time Loss Curve Plotting Tool")
    print(f"   Monitoring: {csv_path}")
    print(f"   Output: {output_path}")
    print(f"   Refresh interval: {args.interval} seconds")
    print(f"   Mode: {'Single plot' if args.once else 'Continuous update'}")
    print()
    
    last_mtime = 0
    last_size = 0
    last_row_count = 0
    last_train_count = 0
    
    try:
        while True:
            # 检查文件是否更新（使用多种方法检测）
            if csv_path.exists():
                current_stat = csv_path.stat()
                current_mtime = current_stat.st_mtime
                current_size = current_stat.st_size
                
                # 计算当前行数和训练数据行数（更精确的检查）
                try:
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        current_row_count = len([l for l in lines if l.strip()])
                        # 计算训练数据行数（至少3个字段且前3个为数字）
                        train_data_count = 0
                        for line in lines:
                            fields = line.strip().split(',')
                            if len(fields) >= 3:
                                try:
                                    float(fields[0])
                                    float(fields[1])
                                    float(fields[2])
                                    train_data_count += 1
                                except ValueError:
                                    pass
                except:
                    current_row_count = 0
                    train_data_count = 0
                
                # 检测文件是否更新（文件大小、修改时间、总行数或训练数据行数变化）
                file_changed = (
                    current_mtime > last_mtime or 
                    current_size != last_size or 
                    current_row_count != last_row_count or
                    train_data_count != last_train_count or
                    args.once
                )
                
                if file_changed:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Updating plot...", end=' ')
                    print(f"(train_data: {train_data_count}, total: {current_row_count})", end=' ')
                    
                    # 加载数据并绘制
                    df = load_metrics(csv_path)
                    if df is not None and len(df) > 0:
                        # 显示数据信息用于调试
                        current_step = int(df['step'].iloc[-1]) if 'step' in df.columns and len(df) > 0 else 0
                        min_step = int(df['step'].iloc[0]) if 'step' in df.columns and len(df) > 0 else 0
                        max_loss = float(df['loss'].iloc[-1]) if 'loss' in df.columns and len(df) > 0 else 0.0
                        
                        success = plot_loss_curves(df, output_path, title, max_iter=args.max_iter)
                        if success:
                            print(f"OK ({len(df)} points, step={min_step}-{current_step}, latest_loss={max_loss:.6f})")
                        else:
                            print("FAILED")
                    else:
                        print("NO DATA")
                    
                    last_mtime = current_mtime
                    last_size = current_size
                    last_row_count = current_row_count
                    last_train_count = train_data_count
                else:
                    # 显示等待信息，包含当前进度
                    try:
                        df_check = load_metrics(csv_path)
                        if df_check is not None and len(df_check) > 0:
                            current_step = int(df_check['step'].iloc[-1]) if 'step' in df_check.columns else 0
                            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Waiting... (step={current_step}, {len(df_check)} rows)", end='\r')
                        else:
                            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Waiting for update...", end='\r')
                    except:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Waiting for update...", end='\r')
            else:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Waiting for metrics.csv...", end='\r')
            
            if args.once:
                break
            
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n\nStopped monitoring")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
