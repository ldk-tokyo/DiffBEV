# dataset settings - 优化版本：平衡显存利用率和训练速度
# 优化策略：
# 1. 适中的batch size（6，而不是8）
# 2. 增加数据加载workers和prefetch
# 3. 优化数据管道
dataset_type = 'NuscenesDataset'
data_root = "/media/ldk950413/data0/nuScenes"  # 注意：实际路径会被统一为小写的nuscenes
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
# 优化配置：平衡显存和速度，充分利用32核CPU
data = dict(
    samples_per_gpu=6,  # 适中的batch size（比4大，比8小），平衡显存和速度
    workers_per_gpu=20,  # 充分利用32核CPU（使用20个workers，约62.5%，留出核心给系统和GPU通信）
    pin_memory=True,  # 启用pin_memory，加速CPU到GPU传输
    persistent_workers=True,  # 保持worker进程，避免重复创建
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
