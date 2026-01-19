# dataset settings - 高显存利用率版本
# 继承基础配置，增加batch size以提高显存利用率
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
# 高显存利用率配置：增加batch size，充分利用32核CPU
data = dict(
    samples_per_gpu=8,  # 从4增加到8，提高显存利用率（RTX 5090有32GB显存）
    workers_per_gpu=20,  # 充分利用32核CPU（使用20个workers，约62.5%）
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,  # PyTorch 2.0+: 每个worker预取4个batch
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
