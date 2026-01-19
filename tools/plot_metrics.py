#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""
训练指标对比可视化工具

功能：
- 自动读取多个runs目录下的metrics.csv
- 生成对比曲线：mIoU、NDS、mAP、三项loss（Lwce、Ldepth、Ldiff）、learning rate
- 输出PNG到reports目录
- 自动生成summary.md，列出每个实验的best指标与对应checkpoint路径

示例：
    python tools/plot_metrics.py \
        runs/baseline runs/diffbev_default \
        --output-dir reports/comparison
"""
import argparse
import os
import os.path as osp
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot training metrics from multiple experiments')
    
    parser.add_argument(
        'runs_dirs',
        type=str,
        nargs='+',
        help='Paths to runs directories containing metrics.csv')
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports',
        help='Output directory for plots and summary (default: reports)')
    
    parser.add_argument(
        '--experiment-names',
        type=str,
        nargs='+',
        help='Names for experiments (default: directory names)')
    
    parser.add_argument(
        '--fig-size',
        type=float,
        nargs=2,
        default=[12, 6],
        help='Figure size in inches (width, height) (default: 12 6)')
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='Figure DPI (default: 150)')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.experiment_names and len(args.experiment_names) != len(args.runs_dirs):
        raise ValueError(
            f'Number of experiment names ({len(args.experiment_names)}) '
            f'must match number of runs directories ({len(args.runs_dirs)})')
    
    return args


def find_metrics_csv(runs_dir: str) -> Optional[str]:
    """在runs目录下查找metrics.csv文件"""
    metrics_path = osp.join(runs_dir, 'metrics.csv')
    if osp.exists(metrics_path):
        return metrics_path
    
    # 尝试在子目录中查找
    for root, dirs, files in os.walk(runs_dir):
        if 'metrics.csv' in files:
            return osp.join(root, 'metrics.csv')
    
    return None


def find_checkpoints(runs_dir: str) -> Dict[str, str]:
    """查找checkpoint文件
    
    Returns:
        dict: {checkpoint_name: checkpoint_path}
    """
    checkpoints = {}
    
    # 查找常见的checkpoint文件名
    common_names = ['best_mIoU.pth', 'best_mIoU_iter_*.pth', 
                    'latest.pth', 'epoch_*.pth', 'iter_*.pth']
    
    for pattern in common_names:
        matches = glob.glob(osp.join(runs_dir, pattern))
        for match in matches:
            name = osp.basename(match)
            checkpoints[name] = match
    
    # 也查找子目录
    for root, dirs, files in os.walk(runs_dir):
        for file in files:
            if file.endswith('.pth'):
                full_path = osp.join(root, file)
                checkpoints[file] = full_path
    
    return checkpoints


def load_metrics(metrics_path: str) -> Optional[pd.DataFrame]:
    """加载metrics.csv文件"""
    try:
        df = pd.read_csv(metrics_path)
        return df
    except Exception as e:
        print(f"Warning: Failed to load {metrics_path}: {e}")
        return None


def get_best_metrics(df: pd.DataFrame) -> Dict[str, Tuple[float, int]]:
    """获取各项指标的最佳值及其对应的step
    
    Returns:
        dict: {metric_name: (best_value, step)}
    """
    best_metrics = {}
    
    # 评估指标（越高越好）
    eval_metrics = ['mIoU', 'NDS', 'mAP']
    for metric in eval_metrics:
        if metric in df.columns:
            # 过滤NaN值
            valid_df = df[df[metric].notna()]
            if len(valid_df) > 0:
                idx_max = valid_df[metric].idxmax()
                best_value = valid_df.loc[idx_max, metric]
                best_step = valid_df.loc[idx_max, 'step'] if 'step' in df.columns else idx_max
                best_metrics[metric] = (best_value, int(best_step))
    
    # Loss指标（越低越好）- 但我们记录的是训练过程中的loss，通常关注最后的值或平均值
    loss_metrics = ['Lwce', 'Ldepth', 'Ldiff']
    for metric in loss_metrics:
        if metric in df.columns:
            valid_df = df[df[metric].notna()]
            if len(valid_df) > 0:
                # 取最后几个epoch的平均值作为稳定值
                last_n = min(100, len(valid_df))
                last_values = valid_df[metric].tail(last_n)
                avg_value = last_values.mean()
                final_step = valid_df['step'].iloc[-1] if 'step' in df.columns else len(valid_df) - 1
                best_metrics[f'{metric}_avg'] = (avg_value, int(final_step))
    
    return best_metrics


def plot_metric_comparison(metrics_dict: Dict[str, pd.DataFrame],
                          metric_name: str,
                          experiment_names: List[str],
                          output_path: str,
                          fig_size: Tuple[float, float] = (12, 6),
                          dpi: int = 150,
                          ylabel: Optional[str] = None,
                          higher_is_better: bool = True):
    """绘制单个指标的对比曲线"""
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_dict)))
    
    for idx, (exp_name, df) in enumerate(metrics_dict.items()):
        if metric_name not in df.columns:
            continue
        
        # 获取有效数据
        valid_df = df[df[metric_name].notna()].copy()
        if len(valid_df) == 0:
            continue
        
        # 使用step作为x轴，如果没有step则使用index
        if 'step' in valid_df.columns:
            x = valid_df['step']
        elif 'iter' in valid_df.columns:
            x = valid_df['iter']
        elif 'epoch' in valid_df.columns:
            x = valid_df['epoch']
        else:
            x = valid_df.index
        
        y = valid_df[metric_name]
        
        # 绘制曲线
        ax.plot(x, y, label=exp_name, color=colors[idx], linewidth=2, alpha=0.8)
        
        # 标记最佳值
        if higher_is_better:
            best_idx = y.idxmax()
            best_x = x.loc[best_idx]
            best_y = y.loc[best_idx]
        else:
            best_idx = y.idxmin()
            best_x = x.loc[best_idx]
            best_y = y.loc[best_idx]
        
        ax.scatter(best_x, best_y, color=colors[idx], s=100, zorder=5,
                  marker='*', edgecolors='black', linewidths=1)
    
    ax.set_xlabel('Step/Iter/Epoch', fontsize=12)
    ax.set_ylabel(ylabel or metric_name, fontsize=12)
    ax.set_title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def generate_summary(experiments_data: List[Dict],
                    output_path: str):
    """生成summary.md文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 实验指标总结\n\n")
        f.write("本文档自动生成，汇总了各个实验的最佳指标和checkpoint路径。\n\n")
        f.write("---\n\n")
        
        for exp_data in experiments_data:
            exp_name = exp_data['name']
            best_metrics = exp_data['best_metrics']
            checkpoints = exp_data['checkpoints']
            runs_dir = exp_data['runs_dir']
            
            f.write(f"## {exp_name}\n\n")
            f.write(f"**实验目录**: `{runs_dir}`\n\n")
            
            # 最佳指标表格
            f.write("### 最佳指标\n\n")
            f.write("| 指标 | 最佳值 | Step |\n")
            f.write("|------|--------|------|\n")
            
            # 评估指标
            eval_metrics = ['mIoU', 'NDS', 'mAP']
            for metric in eval_metrics:
                if metric in best_metrics:
                    value, step = best_metrics[metric]
                    f.write(f"| {metric} | {value:.4f} | {step} |\n")
            
            # Loss指标
            loss_metrics = ['Lwce_avg', 'Ldepth_avg', 'Ldiff_avg']
            for metric in loss_metrics:
                if metric in best_metrics:
                    value, step = best_metrics[metric]
                    f.write(f"| {metric} | {value:.6f} | {step} |\n")
            
            f.write("\n")
            
            # Checkpoint信息
            if checkpoints:
                f.write("### Checkpoints\n\n")
                for ckpt_name, ckpt_path in checkpoints.items():
                    f.write(f"- **{ckpt_name}**: `{ckpt_path}`\n")
                f.write("\n")
            
            f.write("---\n\n")


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = osp.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("Training Metrics Comparison Tool")
    print(f"{'='*80}\n")
    
    # 确定实验名称
    if args.experiment_names:
        experiment_names = args.experiment_names
    else:
        experiment_names = [osp.basename(osp.abspath(d)) for d in args.runs_dirs]
    
    # 加载所有metrics
    metrics_dict = {}
    experiments_data = []
    
    for runs_dir, exp_name in zip(args.runs_dirs, experiment_names):
        print(f"Loading metrics from: {runs_dir} ({exp_name})")
        
        # 查找metrics.csv
        metrics_path = find_metrics_csv(runs_dir)
        if metrics_path is None:
            print(f"  ⚠️  No metrics.csv found in {runs_dir}")
            continue
        
        # 加载metrics
        df = load_metrics(metrics_path)
        if df is None or len(df) == 0:
            print(f"  ⚠️  Failed to load or empty metrics.csv")
            continue
        
        print(f"  ✓ Loaded {len(df)} records")
        metrics_dict[exp_name] = df
        
        # 获取最佳指标
        best_metrics = get_best_metrics(df)
        
        # 查找checkpoints
        checkpoints = find_checkpoints(runs_dir)
        
        experiments_data.append({
            'name': exp_name,
            'runs_dir': runs_dir,
            'best_metrics': best_metrics,
            'checkpoints': checkpoints
        })
        
        print(f"  ✓ Found {len(checkpoints)} checkpoints")
        print()
    
    if len(metrics_dict) == 0:
        print("❌ No metrics data found. Exiting.")
        return
    
    print(f"{'='*80}\n")
    print("Generating plots...\n")
    
    # 绘制各项指标
    metrics_to_plot = {
        'mIoU': {'ylabel': 'mIoU', 'higher_is_better': True},
        'NDS': {'ylabel': 'NDS', 'higher_is_better': True},
        'mAP': {'ylabel': 'mAP', 'higher_is_better': True},
        'Lwce': {'ylabel': 'Lwce (Weighted Cross-Entropy Loss)', 'higher_is_better': False},
        'Ldepth': {'ylabel': 'Ldepth (Depth Loss)', 'higher_is_better': False},
        'Ldiff': {'ylabel': 'Ldiff (Diffusion Loss)', 'higher_is_better': False},
        'lr': {'ylabel': 'Learning Rate', 'higher_is_better': False},
    }
    
    for metric_name, plot_config in metrics_to_plot.items():
        # 检查是否有任何实验包含此指标
        has_metric = any(metric_name in df.columns for df in metrics_dict.values())
        if not has_metric:
            print(f"  ⚠️  Skipping {metric_name} (not found in any experiment)")
            continue
        
        output_path = osp.join(plots_dir, f'{metric_name}_comparison.png')
        plot_metric_comparison(
            metrics_dict,
            metric_name,
            experiment_names,
            output_path,
            fig_size=tuple(args.fig_size),
            dpi=args.dpi,
            ylabel=plot_config['ylabel'],
            higher_is_better=plot_config['higher_is_better']
        )
        print(f"  ✓ Generated: {output_path}")
    
    # 生成summary.md
    summary_path = osp.join(args.output_dir, 'summary.md')
    generate_summary(experiments_data, summary_path)
    print(f"\n  ✓ Generated: {summary_path}")
    
    print(f"\n{'='*80}")
    print(f"✓ All plots and summary generated in: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
