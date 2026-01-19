# Copyright (c) OpenMMLab. All rights reserved.
from .collect_env import collect_env
from .logger import get_root_logger
from .compat import check_pytorch_version, check_mmcv_version, check_environment

__all__ = ['get_root_logger', 'collect_env', 'check_pytorch_version', 'check_mmcv_version', 'check_environment', 'register_all_modules']


def register_all_modules(init_default_scope=True):
    """
    注册所有模块到注册表
    
    Args:
        init_default_scope (bool): 是否初始化默认scope。默认为True。
    
    注意：此函数是为了兼容 mmengine 0.10+ 的自动导入机制。
    在 mmseg 中，模块通过显式导入（如 from .models import *）来注册，
    因此此函数实际上不需要做任何事情。
    """
    # mmseg 的模块已经通过显式导入注册，这里不需要额外操作
    # 但为了兼容 mmengine 的自动导入机制，我们需要提供这个函数
    pass
