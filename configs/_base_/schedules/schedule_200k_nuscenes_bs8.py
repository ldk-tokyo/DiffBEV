# 复现规范：200k iterations, AdamW, lr=4e-4 (batch size=8时线性缩放), weight_decay=0.01, warmup 1500 iter
# 优化器配置（Batch Size=8时的学习率调整）
optimizer = dict(
    type='AdamW',
    lr=4e-4,  # 从2e-4增加到4e-4（batch size从4增加到8，学习率线性缩放）
    betas=(0.9, 0.999),
    weight_decay=0.01)  # 论文规范：0.01
optimizer_config = dict()

# 学习率策略（严格按照论文复现规范）
lr_config = dict(
    policy='poly',
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
    warmup='linear',
    warmup_iters=1500,  # 论文规范：warmup 1500 iter
    warmup_ratio=1e-6)

# 训练配置（严格按照论文复现规范）
runner = dict(type='IterBasedRunner', max_iters=200000)  # 论文规范：200k iterations
checkpoint_config = dict(
    by_epoch=False, 
    interval=5000,
    max_keep_ckpts=3  # 只保留最近3个checkpoint，节省磁盘空间和IO
)
evaluation = dict(
    interval=20000, 
    metric='mIoU',
    efficient_test=True,  # 启用efficient_test模式，节省内存和加速评估
    pre_eval=True  # 使用pre_eval模式，避免内存峰值
)
