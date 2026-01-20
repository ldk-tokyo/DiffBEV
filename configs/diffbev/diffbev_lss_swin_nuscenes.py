# DiffBEV配置 - LSS Transformer with Diffusion for nuScenes
# 启用diffusion模块，使用FS-BEV条件输入和Cross-Attention融合
# 其他训练设置保持与论文规范一致

_base_ = [
    '../_base_/datasets/nuscene.py',
    '../_base_/default_runtime_bf16.py',  # 启用BF16混合精度训练（数值稳定性更好，不需要loss scaling）
    '../_base_/schedules/schedule_200k_nuscenes.py'  # 使用符合规范的训练计划
]

# 模型配置 - 使用LSS Transformer + Diffusion模块
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='BEVSegmentor',  # 使用BEVSegmentor（支持transformer和calib，可配置diffusion）
    # Swin Transformer预训练权重路径
    # 选项1: 使用预训练权重（推荐，需要先下载）
    # 下载地址: https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    # 下载命令: wget -P /media/ldk950413/data0/DiffBEV/pretrained/ https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    # 选项2: 从头训练（设置为None）
    pretrained=None,  # 暂时设置为None，允许从头训练；下载权重后可改为: "/media/ldk950413/data0/DiffBEV/pretrained/swin_tiny_patch4_window7_224.pth"
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
    # 使用LSS Transformer（视图变换模块）
    # BEVSegmentor支持transformer参数，会自动从img_metas提取calib信息
    transformer=dict(
        type='TransformerLiftSplatShoot',
        use_high_res=False,
        downsample=32,
        in_channels=768,
        bev_feature_channels=64,
        ogfH=600,
        ogfW=800,
        outdepth=True,  # 启用深度输出（diffusion需要）
        depthsup=True,  # 启用深度监督（diffusion需要）
        grid_conf=dict(
            dbound=[1, 50, 1],
            xbound=[-25, 25, 0.5],
            zbound=[1, 50, 0.5],
            ybound=[-10, 10, 20]
        )
    ),
    # 使用DiffusionHead（支持diffusion模块）
    # 配置：FS-BEV条件输入 + Cross-Attention融合 + 论文默认xt编码
    decode_head=dict(
        type='DiffusionHead',  # 使用DiffusionHead（支持diffusion模块）
        num_classes=14,  # nuScenes BEV语义分割：14个类别
        align_corners=True,
        # outdepth=True 时会在BEV特征中拼接1个深度通道（64+1=65）
        in_channels=65,
        # Diffusion模块配置
        use_diffusion=True,  # 启用diffusion模块
        # 条件输入配置
        condition_type='FS-BEV',  # Full-Scale BEV条件（默认）
        # 融合方式配置
        fusion_type='Cross-Attention',  # 跨注意力融合（默认）
        # xt编码方式配置
        xt_encoding='default',  # 论文默认实现（通常是Conv编码）
        # 损失函数权重（按照论文规范：Lseg = Lwce + 10*Ldepth + 1*Ldiff）
        loss_depth_weight=10.0,  # Ldepth权重
        loss_diff_weight=1.0,    # Ldiff权重
        # Diffusion其他参数
        diffusion_steps=1000,    # 扩散步数（论文默认）
        noise_schedule='linear', # 噪声调度（论文默认）
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole', output_type='iou', positive_thred=0.5))

# 数据增强和预处理配置（严格按照论文复现规范：800x600分辨率）
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations',
         reduce_zero_label=False,
         with_calib=True,
         with_depth=True,
         depth_dir='ann_bev_dir/train_depth',
         depth_suffix='.npz',
         depth_scale=256.0,
         imdecode_backend='pyramid'),
    dict(type='Resize', img_scale=(800, 600), resize_gt=False, keep_ratio=False),  # 论文规范：800x600
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'gt_depth', 'gt_depth_mask'], 
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
        reduce_zero_label=False,
        pipeline=train_pipeline),
    val=dict(
        type='NuscenesDataset',
        data_root="/media/ldk950413/data0/nuScenes",  # 数据集路径（统一配置，注意大小写）
        img_dir='img_dir/val',
        ann_dir='ann_bev_dir/val',
        reduce_zero_label=False,
        pipeline=test_pipeline),
    test=dict(
        type='NuscenesDataset',
        data_root="/media/ldk950413/data0/nuScenes",  # 数据集路径（统一配置，注意大小写）
        img_dir='img_dir/val',
        ann_dir='ann_bev_dir/val',
        reduce_zero_label=False,
        pipeline=test_pipeline))

# 优化器和学习率配置已在 _base_ 的 schedule_200k_nuscenes.py 中定义
# 这里确保配置正确（会覆盖base配置中的optimizer和lr_config）
# 注意：optimizer和lr_config已经在schedule文件中配置为符合规范的值

# 评估配置
# 注意：efficient_test=True 可以显著减少评估时的内存使用（将结果保存为临时文件而不是全部加载到内存）
evaluation = dict(
    interval=20000, 
    metric='mIoU', 
    save_best='mIoU',
    efficient_test=True  # 启用高效评估模式，减少内存使用（特别是对于大型验证集）
)
