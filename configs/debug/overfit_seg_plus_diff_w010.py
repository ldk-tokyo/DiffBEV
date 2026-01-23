_base_ = ['../diffbev/diffbev_lss_swin_nuscenes.py']

# Seg + diff (no depth)
model = dict(
    transformer=dict(
        depthsup=False,
        outdepth=False,
    ),
    decode_head=dict(
        use_diffusion=True,
        loss_diff_weight=0.10,
        loss_depth_weight=0.0,
    ),
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations',
         reduce_zero_label=False,
         with_calib=True,
         with_depth=False,
         depth_dir='ann_bev_dir/train_depth',
         depth_suffix='.npz',
         depth_scale=256.0,
         imdecode_backend='pyramid'),
    dict(type='OneHotToSegLabel',
         num_classes=14,
         ignore_index=255,
         class_names=['drivable_area', 'ped_crossing', 'walkway', 'carpark',
                      'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
                      'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'],
         priority=[9, 11, 10, 4, 5, 6, 7, 8, 12, 13, 1, 2, 3, 0]),
    dict(type='Resize', img_scale=(800, 600), resize_gt=False, keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'],
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg', 'calib')),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    persistent_workers=False,
    shuffle=False,
    train=dict(
        type='RepeatDataset',
        times=6000,
        dataset=dict(
            type='SingleSampleDataset',
            index=34029,
            dataset=dict(
                type='NuscenesDataset',
                data_root='/media/ldk950413/data0/nuScenes',
                img_dir='img_dir/val',
                ann_dir='ann_bev_dir/val',
                reduce_zero_label=False,
                pipeline=None,
            ),
        ),
    ),
)
data['train']['dataset']['dataset']['pipeline'] = train_pipeline

optimizer = dict(type='AdamW', lr=1e-3, weight_decay=0.0)
lr_config = dict(policy='Fixed')

runner = dict(type='IterBasedRunner', max_iters=1000)
log_config = dict(interval=50)
checkpoint_config = dict(by_epoch=False, interval=200, max_keep_ckpts=200)
evaluation = None

custom_hooks = [
    dict(type='OverfitDebugHook', interval=50),
    dict(type='CheckpointHook', by_epoch=False, interval=200, priority='LOW', max_keep_ckpts=200),
]

work_dir = '/media/ldk950413/data0/DiffBEV/runs/overfit_seg_plus_diff_w010'
