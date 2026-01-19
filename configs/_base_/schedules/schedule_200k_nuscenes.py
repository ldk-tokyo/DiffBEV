# 复现规范：200k iterations, AdamW, lr=2e-4, weight_decay=0.01, warmup 1500 iter
# 优化器配置（严格按照论文复现规范）
optimizer = dict(
    type='AdamW',
    lr=2e-4,  # 论文规范：2e-4
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
checkpoint_config = dict(by_epoch=False, interval=5000)
evaluation = dict(interval=20000, metric='mIoU')  # 每20k iterations评估一次
