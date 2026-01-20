"""
训练指标记录Hook
用于在使用mmcv Runner时，将训练loss/学习率写入metrics.csv与TensorBoard。
"""
import torch

# 尝试导入Hook基类（兼容MMCV 1.x和2.x）
try:
    from mmcv.runner import Hook
    HookBase = Hook
except ImportError:
    try:
        from mmengine.hooks import Hook as HookBase
    except ImportError:
        # 如果都失败，创建一个最小Hook基类
        class HookBase:  # pragma: no cover - fallback
            def __init__(self):
                pass
            priority = 50


class MetricsLoggerHook(HookBase):
    """训练阶段写入metrics.csv的Hook（与metrics_logger配合）。"""

    def __init__(self, interval=50):
        super().__init__()
        self.interval = interval
        self.priority = 50

    def after_train_iter(self, runner):
        # 仅在指定间隔记录
        if self.interval and (runner.iter + 1) % self.interval != 0:
            return

        # 获取log_vars
        log_vars = None
        if hasattr(runner, 'outputs') and isinstance(runner.outputs, dict):
            log_vars = runner.outputs.get('log_vars', None)
        if log_vars is None:
            if hasattr(runner, 'log_buffer'):
                if hasattr(runner.log_buffer, 'output'):
                    log_vars = runner.log_buffer.output
                elif isinstance(runner.log_buffer, dict):
                    log_vars = runner.log_buffer

        if not log_vars:
            return

        # 确保metrics_logger可用
        from mmseg.utils.metrics_logger import MetricsLogger
        if not hasattr(runner, 'metrics_logger') or runner.metrics_logger is None:
            runner.metrics_logger = MetricsLogger(
                work_dir=runner.work_dir,
                csv_filename='metrics.csv',
                mode='train'
            )

        def _get_value(key):
            if key in log_vars:
                value = log_vars[key]
                if isinstance(value, torch.Tensor):
                    return value.item()
                if isinstance(value, (int, float)):
                    return value
            return None

        # 训练损失拆分
        Lwce = _get_value('loss_seg') or _get_value('loss_decode.loss_seg') or _get_value('loss_decode')
        Ldepth = _get_value('loss_depth') or _get_value('loss_decode.loss_depth')
        Ldiff = _get_value('loss_diff') or _get_value('loss_diffusion') or _get_value('loss_decode.loss_diff')
        loss_total = _get_value('loss')

        # 学习率
        learning_rate = None
        if hasattr(runner, 'optimizer') and runner.optimizer is not None:
            try:
                learning_rate = runner.optimizer.param_groups[0].get('lr', None)
            except Exception:
                learning_rate = None
        if learning_rate is None:
            learning_rate = _get_value('lr') or _get_value('learning_rate')

        runner.metrics_logger.log_training_losses(
            Lwce=Lwce,
            Ldepth=Ldepth,
            Ldiff=Ldiff,
            loss_total=loss_total,
            learning_rate=learning_rate,
            step=runner.iter + 1,
            prefix='train',
            mode='train'
        )
        runner.metrics_logger.flush()
