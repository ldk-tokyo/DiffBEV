# Copyright (c) OpenMMLab. All rights reserved.
"""
MMCV 2.x 兼容层：提供 MMCV 1.x 的 build_optimizer 和 build_runner 函数
"""
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
# 尝试导入build_from_cfg（兼容MMCV 1.x和2.x）
try:
    from mmcv.utils import build_from_cfg
except ImportError:
    try:
        from mmengine.registry import build_from_cfg
    except ImportError:
        try:
            from mmengine.utils import build_from_cfg
        except ImportError:
            # 如果都失败，提供一个简化实现
            def build_from_cfg(cfg, registry, default_args=None):
                """从配置构建对象（简化实现）"""
                if isinstance(cfg, str):
                    return registry.get(cfg)
                elif isinstance(cfg, dict):
                    obj_type = cfg.pop('type')
                    if default_args:
                        cfg.update(default_args)
                    obj_cls = registry.get(obj_type)
                    return obj_cls(**cfg)
                else:
                    raise TypeError(f'cfg must be a str or dict, but got {type(cfg)}')
from mmengine.registry import Registry

# 尝试创建OPTIMIZERS registry（如果不存在）
try:
    from mmcv.runner import OPTIMIZERS
except ImportError:
    try:
        # 尝试从mmengine创建
        OPTIMIZERS = Registry('optimizer')
    except:
        OPTIMIZERS = Registry('optimizer', scope='mmseg')


def build_optimizer(model, optimizer_cfg):
    """构建优化器（兼容MMCV 1.x接口）
    
    Args:
        model (nn.Module): 模型
        optimizer_cfg (dict): 优化器配置
        
    Returns:
        Optimizer: PyTorch优化器
    """
    if isinstance(optimizer_cfg, dict):
        optimizer_cfg = optimizer_cfg.copy()
        
        # 处理paramwise_cfg（参数分组配置）
        paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
        
        # 获取优化器类型和参数
        optimizer_type = optimizer_cfg.pop('type', 'AdamW')
        lr = optimizer_cfg.get('lr', 2e-4)
        weight_decay = optimizer_cfg.get('weight_decay', 0.01)
        
        # 如果指定了paramwise_cfg，需要分组参数
        if paramwise_cfg is not None:
            params = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                
                # 检查是否匹配paramwise_cfg的条件
                lr_mult = 1.0
                decay_mult = 1.0
                
                if 'custom_keys' in paramwise_cfg:
                    for key_pattern, key_cfg in paramwise_cfg['custom_keys'].items():
                        if key_pattern in name:
                            lr_mult = key_cfg.get('lr_mult', 1.0)
                            decay_mult = key_cfg.get('decay_mult', 1.0)
                            break
                
                params.append({
                    'params': [param],
                    'lr': lr * lr_mult,
                    'weight_decay': weight_decay * decay_mult
                })
        else:
            # 没有paramwise_cfg，使用所有参数
            params = [p for p in model.parameters() if p.requires_grad]
        
        # 构建优化器（注意：从optimizer_cfg中移除已经提取的参数，避免重复传递）
        optimizer_args = optimizer_cfg.copy()
        optimizer_args.pop('lr', None)  # 移除lr，避免重复
        optimizer_args.pop('weight_decay', None)  # 移除weight_decay，避免重复
        
        if optimizer_type == 'AdamW':
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay, **optimizer_args)
        elif optimizer_type == 'Adam':
            return optim.Adam(params, lr=lr, weight_decay=weight_decay, **optimizer_args)
        elif optimizer_type == 'SGD':
            momentum = optimizer_args.pop('momentum', 0.9)
            return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, **optimizer_args)
        else:
            # 尝试从registry构建
            try:
                if optimizer_type in OPTIMIZERS:
                    optimizer_class = OPTIMIZERS.get(optimizer_type)
                    return optimizer_class(params, lr=lr, **optimizer_cfg)
                else:
                    # 尝试直接使用PyTorch的优化器
                    if hasattr(optim, optimizer_type):
                        optimizer_class = getattr(optim, optimizer_type)
                        return optimizer_class(params, lr=lr, **optimizer_cfg)
                    else:
                        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
            except Exception as e:
                warnings.warn(f"无法从registry构建优化器 {optimizer_type}: {e}，使用AdamW作为默认值")
                return optim.AdamW(params, lr=lr, weight_decay=weight_decay, **optimizer_cfg)
    else:
        # 如果optimizer_cfg已经是优化器实例，直接返回
        return optimizer_cfg


def build_runner(runner_cfg, default_args=None):
    """构建runner（兼容MMCV 1.x接口）
    
    Args:
        runner_cfg (dict): Runner配置
        default_args (dict, optional): 默认参数
        
    Returns:
        Runner: MMCV Runner实例
    """
    if default_args is None:
        default_args = {}
    
    if isinstance(runner_cfg, dict):
        runner_type = runner_cfg.get('type', 'IterBasedRunner')
        
        # 尝试从mmcv.runner导入Runner类
        try:
            from mmcv.runner import IterBasedRunner, EpochBasedRunner, RUNNERS
            if runner_type == 'IterBasedRunner':
                RunnerClass = IterBasedRunner
            elif runner_type == 'EpochBasedRunner':
                RunnerClass = EpochBasedRunner
            else:
                RunnerClass = RUNNERS.get(runner_type)
                if RunnerClass is None:
                    raise ValueError(f"Unknown runner type: {runner_type}")
            
            # 合并配置
            runner_args = {**default_args, **runner_cfg}
            runner_args.pop('type', None)
            
            return RunnerClass(**runner_args)
        except ImportError:
            # 如果mmcv.runner不存在，创建一个兼容MMCV 1.x接口的Runner包装类
            # 合并配置
            runner_args = {**default_args, **runner_cfg}
            runner_type = runner_args.pop('type', 'IterBasedRunner')
            
            # 提取MMCV 1.x特有的参数
            model = runner_args.pop('model', None)
            optimizer = runner_args.pop('optimizer', None)
            work_dir = runner_args.pop('work_dir', None)
            logger = runner_args.pop('logger', None)
            meta = runner_args.pop('meta', None)
            batch_processor = runner_args.pop('batch_processor', None)
            
            # 创建一个兼容MMCV 1.x接口的Runner类
            from mmseg.utils.runner_compat import MMCVRunnerCompat
            runner = MMCVRunnerCompat(
                model=model,
                optimizer=optimizer,
                work_dir=work_dir,
                logger=logger,
                meta=meta,
                batch_processor=batch_processor,
                runner_type=runner_type,
                max_iters=runner_args.pop('max_iters', None),
                max_epochs=runner_args.pop('max_epochs', None)
            )
            
            warnings.warn(
                "使用兼容MMCV 1.x接口的Runner包装类。"
                "某些高级功能可能无法完全支持。"
            )
            
            return runner
    else:
        # 如果runner_cfg已经是Runner实例，直接返回
        return runner_cfg
