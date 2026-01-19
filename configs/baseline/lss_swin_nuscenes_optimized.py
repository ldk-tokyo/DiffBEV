# Baseline配置 - 优化版本：平衡显存利用率和训练速度
# 优化策略：
# 1. Batch Size=6（平衡显存和速度）
# 2. FP16混合精度训练（充分利用Tensor Cores）
# 3. 增加数据加载workers和prefetch
# 4. 预期显存使用: 15-18GB（利用率45-55%）
# 5. 预期训练速度: 比batch size=8快，比batch size=4稍慢但吞吐量更高

_base_ = [
    '../_base_/datasets/nuscene_optimized.py',  # 使用优化版本的数据配置
    '../_base_/default_runtime_fp16.py',  # 启用FP16
    '../_base_/schedules/schedule_200k_nuscenes_bs6.py'  # 使用batch size=6的学习率配置
]

# 模型配置（与baseline相同）
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='BEVSegmentor',
    pretrained=None,  # 暂时设置为None，允许从头训练
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
    # 使用LSS Transformer（baseline方法，无diffusion）
    transformer=dict(
        type='TransformerLiftSplatShoot',
        use_high_res=False,
        downsample=32,
        in_channels=768,
        bev_feature_channels=64,
        ogfH=600,
        ogfW=800,
        outdepth=False,  # 关闭深度输出（baseline不需要）
        depthsup=False,  # 关闭深度监督（baseline不需要）
        grid_conf=dict(
            dbound=[1, 50, 1],
            xbound=[-25, 25, 0.5],
            zbound=[1, 50, 0.5],
            ybound=[-10, 10, 20]
        )
    ),
    # 使用PyramidHead（只包含Lwce损失，无Ldepth和Ldiff）
    decode_head=dict(
        type='PyramidHead',
        num_classes=14,  # nuScenes BEV语义分割：14个类别
        align_corners=True),
    train_cfg=dict(),
    test_cfg=dict(mode='whole', output_type='iou', positive_thred=0.5))
