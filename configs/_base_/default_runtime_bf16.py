# yapf:disable
# 混合精度训练（BF16）配置
# 启用BF16可以：
# 1. 减少显存使用约50%（与FP16类似）
# 2. 提高训练速度约1.5-2倍（充分利用Tensor Cores）
# 3. 数值稳定性更好：BF16的数值范围与FP32相同，不容易出现梯度下溢
# 4. 不需要loss scaling（BF16数值范围大，通常不需要GradScaler）
# 注意：需要GPU支持BF16（如RTX 5090、A100等）

bf16 = dict()  # BF16通常不需要loss_scale，因为数值范围与FP32相同

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
