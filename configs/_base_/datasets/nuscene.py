# dataset settings
dataset_type = 'NuscenesDataset'
# nuScenes数据集路径 - 统一配置为固定路径
# 数据集应按照以下结构组织：
# data_root/
#   ├── img_dir/
#   │   ├── train/
#   │   └── val/
#   ├── ann_bev_dir/
#   │   ├── train/
#   │   ├── val/
#   │   ├── train_depth/
#   │   └── val_depth/
#   ├── samples/          # nuScenes原始samples目录（v1.0-trainval）
#   ├── sweeps/           # nuScenes原始sweeps目录（v1.0-trainval）
#   └── v1.0-trainval/    # nuScenes v1.0-trainval版本目录
data_root = "/media/ldk950413/data0/nuScenes"  # 注意：路径使用大写S（nuScenes）
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False, with_calib=True, imdecode_backend='pyramid'),
    dict(type='Resize', img_scale=(800, 600), resize_gt=False, keep_ratio=False),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg', 'calib')),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False, with_calib=True, imdecode_backend='pyramid'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False, resize_gt=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg', 'gt_semantic_seg', 'calib')),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=20,  # 充分利用32核CPU（使用20个workers，约62.5%，留出核心给系统和GPU通信）
    pin_memory=True,    # 启用pin_memory，加速CPU到GPU的数据传输
    persistent_workers=True,  # 保持worker进程，避免重复创建的开销
    prefetch_factor=4,  # PyTorch 2.0+: 每个worker预取4个batch，减少等待时间
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/train',
        ann_dir='ann_bev_dir/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_bev_dir/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_bev_dir/val',
        pipeline=test_pipeline))
