# yapf:disable
# 混合精度训练（FP16）配置
# 启用FP16可以：
# 1. 减少显存使用约50%
# 2. 提高训练速度约1.5-2倍
# 3. 允许使用更大的batch size
fp16 = dict(loss_scale=1024.0)  # 增加loss_scale以避免FP16梯度下溢（从512增加到1024）

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
