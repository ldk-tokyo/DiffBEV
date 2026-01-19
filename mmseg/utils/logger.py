# Copyright (c) OpenMMLab. All rights reserved.
import logging

# 尝试从mmcv.utils导入，如果失败则从mmengine.utils导入
try:
    from mmcv.utils import get_logger
except ImportError:
    try:
        from mmengine.utils import get_logger
    except ImportError:
        # 如果都失败，可能需要导入logging并创建简化版本
        import logging
        def get_logger(name, log_file=None, log_level=logging.INFO):
            logger = logging.getLogger(name)
            logger.setLevel(log_level)
            if log_file and not logger.handlers:
                handler = logging.FileHandler(log_file)
                handler.setLevel(log_level)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
            return logger


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmseg".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """

    logger = get_logger(name='mmseg', log_file=log_file, log_level=log_level)

    return logger
