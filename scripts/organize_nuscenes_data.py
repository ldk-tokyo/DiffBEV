#!/usr/bin/env python
"""
nuScenes数据组织脚本

将 make_nuscenes_labels.py 生成的标注文件和原始图像文件
组织到 img_dir 和 ann_bev_dir 目录结构，并按 train/val 划分。
"""

import os
import os.path as osp
import shutil
from tqdm import tqdm
from nuscenes import NuScenes

# 导入 scene 划分信息
import sys
sys.path.insert(0, osp.join(osp.dirname(__file__), '..', 'mono-semantic-maps', 'src'))
from data.nuscenes.splits import TRAIN_SCENES, VAL_SCENES
from data.nuscenes.utils import CAMERA_NAMES, iterate_samples


def organize_nuscenes_data(dataroot, label_root, output_root):
    """
    组织nuScenes数据到img_dir和ann_bev_dir目录
    
    Args:
        dataroot: nuScenes数据集根目录 (包含v1.0-trainval等)
        label_root: BEV标注文件目录 (map-labels-v1.2)
        output_root: 输出根目录 (通常是dataroot)
    """
    
    # 创建输出目录
    img_dir_train = osp.join(output_root, 'img_dir', 'train')
    img_dir_val = osp.join(output_root, 'img_dir', 'val')
    ann_dir_train = osp.join(output_root, 'ann_bev_dir', 'train')
    ann_dir_val = osp.join(output_root, 'ann_bev_dir', 'val')
    
    os.makedirs(img_dir_train, exist_ok=True)
    os.makedirs(img_dir_val, exist_ok=True)
    os.makedirs(ann_dir_train, exist_ok=True)
    os.makedirs(ann_dir_val, exist_ok=True)
    
    print(f"输出目录:")
    print(f"  img_dir/train: {img_dir_train}")
    print(f"  img_dir/val: {img_dir_val}")
    print(f"  ann_bev_dir/train: {ann_dir_train}")
    print(f"  ann_bev_dir/val: {ann_dir_val}")
    print()
    
    # 加载nuScenes数据集
    print("加载nuScenes数据集...")
    nuscenes = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)
    print(f"✓ 数据集加载完成 (共 {len(nuscenes.scene)} 个场景)")
    print()
    
    # 统计信息
    train_count = 0
    val_count = 0
    missing_label_count = 0
    missing_image_count = 0
    
    # 遍历所有场景
    print("开始组织数据...")
    all_scenes = list(nuscenes.scene)
    
    for scene in tqdm(all_scenes, desc="处理场景"):
        scene_name = scene['name']
        
        # 确定是训练集还是验证集
        if scene_name in TRAIN_SCENES:
            split = 'train'
            train_count += 1
        elif scene_name in VAL_SCENES:
            split = 'val'
            val_count += 1
        else:
            # 跳过不在训练或验证集中的场景
            continue
        
        # 设置输出目录
        img_dir = img_dir_train if split == 'train' else img_dir_val
        ann_dir = ann_dir_train if split == 'train' else ann_dir_val
        
        # 遍历该场景的所有样本
        for sample in iterate_samples(nuscenes, scene['first_sample_token']):
            
            # 遍历所有相机
            for camera in CAMERA_NAMES:
                cam_token = sample['data'][camera]
                
                # 源文件路径
                # 标注文件: label_root/<cam_token>.png
                label_src = osp.join(label_root, f'{cam_token}.png')
                
                # 图像文件: 从nuScenes原始数据中获取
                try:
                    img_src = nuscenes.get_sample_data_path(cam_token)
                except:
                    missing_image_count += 1
                    continue
                
                # 目标文件路径（使用相同的文件名）
                filename = f'{cam_token}.png'
                img_dst = osp.join(img_dir, filename)
                label_dst = osp.join(ann_dir, filename)
                
                # 复制图像文件
                if osp.exists(img_src):
                    if not osp.exists(img_dst):
                        shutil.copy2(img_src, img_dst)
                else:
                    missing_image_count += 1
                
                # 复制标注文件
                if osp.exists(label_src):
                    if not osp.exists(label_dst):
                        shutil.copy2(label_src, label_dst)
                else:
                    missing_label_count += 1
    
    print()
    print("="*80)
    print("数据组织完成！")
    print("="*80)
    print(f"训练场景数: {train_count}")
    print(f"验证场景数: {val_count}")
    print(f"缺失标注文件数: {missing_label_count}")
    print(f"缺失图像文件数: {missing_image_count}")
    print()
    print(f"训练图像数量: {len(os.listdir(img_dir_train))}")
    print(f"验证图像数量: {len(os.listdir(img_dir_val))}")
    print(f"训练标注数量: {len(os.listdir(ann_dir_train))}")
    print(f"验证标注数量: {len(os.listdir(ann_dir_val))}")
    print("="*80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='组织nuScenes数据到img_dir和ann_bev_dir')
    parser.add_argument('--dataroot', type=str, 
                       default='/media/ldk950413/data0/nuScenes',
                       help='nuScenes数据集根目录')
    parser.add_argument('--label-root', type=str,
                       default='/media/ldk950413/data0/nuScenes/nuscenes/map-labels-v1.2',
                       help='BEV标注文件根目录')
    parser.add_argument('--output-root', type=str,
                       default='/media/ldk950413/data0/nuScenes',
                       help='输出根目录（默认为dataroot）')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not osp.exists(args.dataroot):
        print(f"错误: 数据集根目录不存在: {args.dataroot}")
        sys.exit(1)
    
    if not osp.exists(args.label_root):
        print(f"错误: 标注文件目录不存在: {args.label_root}")
        print(f"请先运行 make_nuscenes_labels.py 生成标注文件")
        sys.exit(1)
    
    # 执行组织
    organize_nuscenes_data(args.dataroot, args.label_root, args.output_root)
