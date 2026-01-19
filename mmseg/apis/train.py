# Copyright (c) OpenMMLab. All rights reserved.
import os
import random
import warnings

import numpy as np
import torch
# 尝试从 mmcv.parallel 导入（mmcv 1.x），如果失败则尝试其他位置
try:
    from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
except ImportError:
    try:
        from mmengine.parallel import MMDataParallel, MMDistributedDataParallel
    except ImportError:
        # 如果都失败，使用 PyTorch 原生的 DataParallel
        from torch.nn.parallel import DataParallel as MMDataParallel
        from torch.nn.parallel import DistributedDataParallel as MMDistributedDataParallel
# 尝试从 mmcv.runner 导入（mmcv 1.x），如果失败则使用兼容实现
try:
    from mmcv.runner import build_optimizer, build_runner
except ImportError:
    # 使用兼容层
    from mmseg.utils.mmcv_compat import build_optimizer, build_runner
    warnings.warn("使用MMCV 2.x兼容层的build_optimizer和build_runner。某些功能可能需要调整。")

from mmseg.core import DistEvalHook, EvalHook
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]

    # put model on gpus (or CPU if CUDA not available or forced)
    use_cuda = torch.cuda.is_available() and len(cfg.gpu_ids) > 0 and cfg.gpu_ids[0] >= 0
    
    # 检查是否强制使用CPU（通过环境变量或CUDA不可用）
    if os.environ.get('FORCE_CPU', '').lower() in ('1', 'true', 'yes'):
        use_cuda = False
        logger.warning('FORCE_CPU环境变量已设置，强制使用CPU模式')
    
    if not use_cuda:
        logger.warning('使用CPU模式训练（可能很慢）')
        # CPU模式：直接使用模型，不包装DataParallel
        # 注意：CPU模式不支持DataParallel，需要直接使用模型
        pass  # 保持模型在CPU上
    elif distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    # 注册Loss结构自检Hook
    try:
        from mmseg.core.hooks.loss_check_hook import LossCheckHook
        
        # 从配置中获取权重值
        lambda_depth = 10.0
        lambda_diff = 1.0
        
        if hasattr(cfg, 'model') and cfg.model is not None:
            decode_head_cfg = cfg.model.get('decode_head', {})
            if isinstance(decode_head_cfg, dict):
                lambda_depth = decode_head_cfg.get('loss_depth_weight', lambda_depth)
                lambda_diff = decode_head_cfg.get('loss_diff_weight', lambda_diff)
        
        loss_check_hook = LossCheckHook(
            check_interval=10,
            monitor_iters=100,
            lambda_depth=lambda_depth,
            lambda_diff=lambda_diff
        )
        runner.register_hook(loss_check_hook, priority='HIGHEST')
        logger.info("✅ Loss结构自检Hook已注册")
    except Exception as e:
        logger.warning(f"⚠️  无法注册Loss结构自检Hook: {e}")

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        # 评估时减少workers以减少内存使用
        eval_workers = min(cfg.data.workers_per_gpu, 1)  # 评估时最多使用1个worker
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=eval_workers,  # 评估时使用更少的workers
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
