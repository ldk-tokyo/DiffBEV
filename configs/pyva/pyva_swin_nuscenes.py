# DiffBEV 论文复现配置 - nuScenes v1.0-trainval
# 严格按照复现规范配置
_base_ = [
    '../_base_/models/pyva_swin.py',
    '../_base_/datasets/nuscene.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_200k_nuscenes.py'  # 使用符合规范的训练计划
]

# 模型配置
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='new_pyva_BEVSegmentor',
    # 请替换为实际的Swin Transformer预训练权重路径
    pretrained="YOUR_PATH/swin_tiny_patch4_window7_224.pth",
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(3,),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.0,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg,
        pretrain_style='official',
        output_missing_index_as_none=True),
    transformer=dict(type='v4_Pyva_transformer', size=(32, 32), back='swin'),
    decode_head=dict(
        type='PyramidHead',
        num_classes=14,  # nuScenes BEV语义分割：14个类别
        align_corners=True),
    train_cfg=dict(),
    test_cfg=dict(mode='whole', output_type='iou', positive_thred=0.5))

# 数据增强和预处理配置（严格按照论文复现规范：800x600分辨率）
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False, with_calib=True, imdecode_backend='pyramid'),
    dict(type='Resize', img_scale=(800, 600), resize_gt=False, keep_ratio=False),  # 论文规范：800x600
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'], 
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                   'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg', 'calib')),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False, with_calib=True, imdecode_backend='pyramid'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 600),  # 论文规范：800x600
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False, resize_gt=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], 
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'img_shape', 'pad_shape', 'scale_factor', 'flip',
                           'flip_direction', 'img_norm_cfg', 'gt_semantic_seg', 'calib')),
        ])
]

# 数据加载配置（严格按照论文复现规范：batch size=4 per GPU）
data = dict(
    samples_per_gpu=4,  # 论文规范：batch size=4 per GPU
    workers_per_gpu=2,
    train=dict(
        type='NuscenesDataset',
        data_root="/media/ldk950413/data0/nuScenes",  # 数据集路径（统一配置，注意大小写）
        img_dir='img_dir/train',
        ann_dir='ann_bev_dir/train',
        pipeline=train_pipeline),
    val=dict(
        type='NuscenesDataset',
        data_root="/media/ldk950413/data0/nuScenes",  # 数据集路径（统一配置，注意大小写）
        img_dir='img_dir/val',
        ann_dir='ann_bev_dir/val',
        pipeline=test_pipeline),
    test=dict(
        type='NuscenesDataset',
        data_root="/media/ldk950413/data0/nuScenes",  # 数据集路径（统一配置，注意大小写）
        img_dir='img_dir/val',
        ann_dir='ann_bev_dir/val',
        pipeline=test_pipeline))

# 优化器和学习率配置已在 _base_ 的 schedule_200k_nuscenes.py 中定义
# 这里确保配置正确（会覆盖base配置中的optimizer和lr_config）
# 注意：optimizer和lr_config已经在schedule文件中配置为符合规范的值

# 评估配置
evaluation = dict(interval=20000, metric='mIoU', save_best='mIoU')
