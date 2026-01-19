#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""
nuScenes BEV分割结果可视化脚本

功能：
- 输入：baseline checkpoint、DiffBEV checkpoint + 若干样本索引
- 输出：GT、baseline预测、DiffBEV预测的BEV分割结果对比图（三列）
- 使用统一色表与阈值规则
- 结果保存至work_dir/vis目录

示例：
    python tools/vis_nuscenes_bev.py \\
        configs/baseline/lss_swin_nuscenes.py \\
        runs/baseline/best_mIoU.pth \\
        --diffbev-config configs/diffbev/diffbev_lss_swin_nuscenes.py \\
        --diffbev-checkpoint runs/diffbev_default/best_mIoU.pth \\
        --indices 0 10 20 30 \\
        --work-dir runs/vis_comparison
"""
import argparse
import os
import os.path as osp
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mmengine import Config as mmcv_Config
from mmcv import Config

from mmseg.apis import init_segmentor
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.datasets.nuscenes import visualize_map_mask, NuscenesDataset
from mmseg.models.utils.vis_util import nuscenes_palette, nuscenes_class


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize nuScenes BEV segmentation results')
    
    # 必需参数
    parser.add_argument('baseline_config', help='Baseline config file path')
    parser.add_argument('baseline_checkpoint', help='Baseline checkpoint file path')
    
    # DiffBEV参数（可选）
    parser.add_argument('--diffbev-config', help='DiffBEV config file path')
    parser.add_argument('--diffbev-checkpoint', help='DiffBEV checkpoint file path')
    
    # 数据参数
    parser.add_argument(
        '--indices',
        type=int,
        nargs='+',
        default=[0, 10, 20, 30],
        help='Sample indices to visualize (default: 0 10 20 30)')
    
    # 输出参数
    parser.add_argument(
        '--work-dir',
        type=str,
        default='work_dirs/vis',
        help='Directory to save visualization results')
    
    # 其他参数
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device used for inference')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for binary classification (default: 0.5)')
    parser.add_argument(
        '--fig-size',
        type=int,
        nargs=2,
        default=[24, 8],
        help='Figure size (width, height) in inches (default: 24 8)')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.diffbev_config and not args.diffbev_checkpoint:
        raise ValueError('--diffbev-checkpoint is required when --diffbev-config is provided')
    if args.diffbev_checkpoint and not args.diffbev_config:
        raise ValueError('--diffbev-config is required when --diffbev-checkpoint is provided')
    
    return args


def load_model(config_path, checkpoint_path, device='cuda:0'):
    """加载模型"""
    print(f"Loading model from {config_path}...")
    print(f"  Checkpoint: {checkpoint_path}")
    
    model = init_segmentor(config_path, checkpoint_path, device=device)
    model.eval()
    
    return model


def get_sample_from_dataset(dataset, index):
    """从数据集中获取单个样本"""
    if index >= len(dataset):
        raise IndexError(f"Index {index} out of range (dataset size: {len(dataset)})")
    
    sample = dataset[index]
    return sample


def inference_sample(model, sample, threshold=0.5):
    """
    对单个样本进行推理
    
    Args:
        model: 模型
        sample: 数据样本（dict）
        threshold: 二值化阈值
        
    Returns:
        pred_mask: 预测的BEV分割mask，shape为 (num_classes, H, W)
    """
    # 准备输入数据
    img = sample['img']
    img_metas = sample['img_metas']
    
    # 确保img是tensor
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img)
    
    # 确保img有batch维度
    if img.dim() == 3:
        img = img.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
    
    # 确保img在正确的设备上
    device = next(model.parameters()).device
    img = img.to(device)
    
    # 处理img_metas - 模型期望img_metas是列表的列表
    if isinstance(img_metas, list):
        # 如果已经是列表
        if len(img_metas) > 0 and isinstance(img_metas[0], dict):
            # 元素是dict，包装成列表的列表
            img_metas_batch = [img_metas]
        else:
            # 元素不是dict，可能需要进一步处理
            img_metas_batch = [img_metas] if isinstance(img_metas[0], dict) else [[img_metas]]
    elif isinstance(img_metas, dict):
        # 单个dict，包装成列表的列表
        img_metas_batch = [[img_metas]]
    else:
        img_metas_batch = [[img_metas]]
    
    # 推理
    # model的__call__方法会调用forward_test，期望img_metas是列表的列表
    with torch.no_grad():
        result = model(return_loss=False, img=img, img_metas=img_metas_batch, rescale=False)
    
    # 处理结果
    # result通常是list，第一个元素是预测结果
    if isinstance(result, list):
        result = result[0]
    
    # 获取预测的logits
    # 对于BEV分割，结果通常是logits tensor
    if isinstance(result, torch.Tensor):
        pred_logits = result.cpu().numpy()
    elif isinstance(result, np.ndarray):
        pred_logits = result
    else:
        raise ValueError(f"Unexpected result type: {type(result)}")
    
    # pred_logits的shape可能是 (num_classes, H, W) 或 (H, W, num_classes)
    if pred_logits.ndim == 2:
        # (H, W) - 可能是单类别，需要转换为(num_classes, H, W)
        # 但nuScenes是多类别，所以这种情况不应该发生
        pred_logits = pred_logits[np.newaxis, :, :]
    elif pred_logits.ndim == 3:
        # 检查是否是 (H, W, num_classes)
        if pred_logits.shape[2] <= 20:  # 假设num_classes <= 20
            # 可能是 (H, W, num_classes)，需要转换为 (num_classes, H, W)
            pred_logits = pred_logits.transpose(2, 0, 1)
    
    # 确保是 (num_classes, H, W)
    if pred_logits.ndim != 3:
        raise ValueError(f"Unexpected pred_logits shape: {pred_logits.shape}")
    
    num_classes, h, w = pred_logits.shape
    
    # 应用sigmoid并二值化
    pred_logits = pred_logits.astype(np.float32)
    pred_probs = 1 / (1 + np.exp(-np.clip(pred_logits, -500, 500)))  # 防止溢出
    
    # 二值化
    pred_mask = (pred_probs > threshold).astype(bool)
    
    return pred_mask


def visualize_bev_mask(mask, palette=None):
    """
    可视化BEV分割mask
    
    Args:
        mask: BEV分割mask，shape为 (num_classes, H, W) 或 (1, num_classes, H, W)
        palette: 调色板（可选）
        
    Returns:
        vis_img: 可视化图像，shape为 (H, W, 3)
    """
    # 确保mask是3D: (num_classes, H, W)
    if mask.ndim == 4:
        mask = mask[0]  # 去除batch维度
    
    # 使用统一的visualize_map_mask函数
    vis_img = visualize_map_mask(mask)
    
    return vis_img


def create_comparison_figure(gt_mask, baseline_mask, diffbev_mask, 
                             sample_idx, threshold=0.5, fig_size=(24, 8)):
    """
    创建对比图（三列：GT、Baseline、DiffBEV）
    
    Args:
        gt_mask: GT mask，shape为 (num_classes, H, W) 或 (1, num_classes, H, W)
        baseline_mask: Baseline预测mask
        diffbev_mask: DiffBEV预测mask（可选）
        sample_idx: 样本索引
        threshold: 阈值
        fig_size: 图像大小
        
    Returns:
        fig: matplotlib figure对象
    """
    # 可视化各个mask
    gt_vis = visualize_bev_mask(gt_mask)
    baseline_vis = visualize_bev_mask(baseline_mask)
    
    # 创建figure
    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(1, 3, figure=fig, wspace=0.1, hspace=0.1)
    
    # GT
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(gt_vis)
    ax1.set_title('Ground Truth', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # Baseline
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(baseline_vis)
    ax2.set_title('Baseline Prediction', fontsize=16, fontweight='bold')
    ax2.axis('off')
    
    # DiffBEV
    if diffbev_mask is not None:
        diffbev_vis = visualize_bev_mask(diffbev_mask)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(diffbev_vis)
        ax3.set_title('DiffBEV Prediction', fontsize=16, fontweight='bold')
        ax3.axis('off')
    else:
        # 如果没有DiffBEV，显示空白或baseline
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.text(0.5, 0.5, 'DiffBEV\nNot Available', 
                ha='center', va='center', fontsize=14)
        ax3.set_title('DiffBEV Prediction', fontsize=16, fontweight='bold')
        ax3.axis('off')
    
    # 添加总标题
    fig.suptitle(f'Sample {sample_idx} - BEV Segmentation Comparison (threshold={threshold})',
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    return fig


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.work_dir, exist_ok=True)
    vis_dir = osp.join(args.work_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("nuScenes BEV Segmentation Visualization")
    print(f"{'='*80}\n")
    print(f"Baseline Config: {args.baseline_config}")
    print(f"Baseline Checkpoint: {args.baseline_checkpoint}")
    if args.diffbev_config:
        print(f"DiffBEV Config: {args.diffbev_config}")
        print(f"DiffBEV Checkpoint: {args.diffbev_checkpoint}")
    print(f"Sample Indices: {args.indices}")
    print(f"Output Directory: {args.work_dir}")
    print(f"Threshold: {args.threshold}\n")
    
    # 加载配置
    baseline_cfg = Config.fromfile(args.baseline_config)
    
    # 加载数据集
    print("Loading dataset...")
    test_dataset = build_dataset(baseline_cfg.data.test)
    print(f"Dataset size: {len(test_dataset)}\n")
    
    # 加载模型
    baseline_model = load_model(
        args.baseline_config,
        args.baseline_checkpoint,
        device=args.device
    )
    print()
    
    diffbev_model = None
    if args.diffbev_config and args.diffbev_checkpoint:
        diffbev_model = load_model(
            args.diffbev_config,
            args.diffbev_checkpoint,
            device=args.device
        )
        print()
    
    # 处理每个样本
    for idx in args.indices:
        print(f"Processing sample {idx}...")
        
        try:
            # 获取样本
            sample = get_sample_from_dataset(test_dataset, idx)
            
            # 获取GT mask
            gt_semantic_seg = sample['gt_semantic_seg']
            if isinstance(gt_semantic_seg, torch.Tensor):
                gt_mask = gt_semantic_seg.cpu().numpy()
            else:
                gt_mask = np.array(gt_semantic_seg)
            
            # 确保GT mask是 (num_classes, H, W) 格式
            # nuScenes的GT格式是 (1, num_classes+1, H, W) 或 (num_classes+1, H, W)
            # 最后一个通道是valid mask，需要去掉
            if gt_mask.ndim == 4:
                # (1, num_classes+1, H, W) -> (num_classes+1, H, W) -> (num_classes, H, W)
                gt_mask = gt_mask[0]
                if gt_mask.shape[0] > len(NuscenesDataset.CLASSES):
                    gt_mask = gt_mask[:-1]  # 去掉mask通道
            elif gt_mask.ndim == 3:
                # (num_classes+1, H, W) -> (num_classes, H, W)
                if gt_mask.shape[0] > len(NuscenesDataset.CLASSES):
                    gt_mask = gt_mask[:-1]  # 去掉mask通道
            else:
                raise ValueError(f"Unexpected GT mask shape: {gt_mask.shape}")
            
            # GT已经是二值化的，直接使用
            gt_mask = gt_mask.astype(bool)
            
            # Baseline推理
            baseline_mask = inference_sample(
                baseline_model, sample, threshold=args.threshold
            )
            
            # DiffBEV推理（如果有）
            diffbev_mask = None
            if diffbev_model is not None:
                diffbev_mask = inference_sample(
                    diffbev_model, sample, threshold=args.threshold
                )
            
            # 创建对比图
            fig = create_comparison_figure(
                gt_mask, baseline_mask, diffbev_mask,
                sample_idx=idx,
                threshold=args.threshold,
                fig_size=args.fig_size
            )
            
            # 保存图像
            output_path = osp.join(vis_dir, f'sample_{idx:04d}.png')
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  ✓ Saved to {output_path}")
            
        except Exception as e:
            print(f"  ✗ Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"Visualization complete! Results saved to: {vis_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
