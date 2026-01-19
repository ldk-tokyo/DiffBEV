# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import collect_env as collect_base_env
# 尝试从mmcv.utils导入，如果失败则从mmengine.utils导入
try:
    from mmcv.utils import get_git_hash
except ImportError:
    try:
        from mmengine.utils import get_git_hash
    except ImportError:
        # 如果都失败，提供一个简化实现
        def get_git_hash():
            return 'unknown'

import mmseg


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMSegmentation'] = f'{mmseg.__version__}+{get_git_hash()[:7]}'

    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print('{}: {}'.format(name, val))
