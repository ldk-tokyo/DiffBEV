# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmengine import Config as mmcv_Config
import mmcv
from packaging.version import parse

from .version import __version__, version_info

# MMCV版本要求
# 优先支持MMCV 2.x（用于PyTorch 2.x和RTX 5090等新GPU）
# 如果使用旧GPU和PyTorch 1.x，可以设置环境变量使用MMCV 1.x
import os
USE_MMCV2 = os.environ.get('DIFFBEV_USE_MMCV1', '0') != '1'  # 默认使用MMCV 2.x

if USE_MMCV2:
    # MMCV 2.x版本要求（推荐，支持PyTorch 2.x和新GPU）
    MMCV_MIN = '2.0.0'
    MMCV_MAX = '3.0.0'
else:
    # MMCV 1.x版本要求（仅用于旧GPU和PyTorch 1.x）
    MMCV_MIN = '1.3.13'
    MMCV_MAX = '1.4.0'

# 兼容性：允许通过环境变量强制使用MMCV 2.x（向后兼容）
ALLOW_MMCV2 = os.environ.get('DIFFBEV_ALLOW_PYTORCH2', '0') == '1' or USE_MMCV2

# 在导入时检查PyTorch版本兼容性
def _check_pytorch_version():
    """检查PyTorch版本是否兼容（在模块导入时自动执行）"""
    import os
    
    # 默认使用PyTorch 2.x（推荐配置，支持新GPU架构）
    # 如果需要使用PyTorch 1.x，设置环境变量 DIFFBEV_USE_PYTORCH1=1
    use_pytorch1 = os.environ.get('DIFFBEV_USE_PYTORCH1', '0') == '1'
    
    try:
        import torch
        pytorch_version = torch.__version__
        # 处理版本号格式，如 "1.9.1+cu111" 或 "2.5.0"
        version_parts = pytorch_version.split('+')[0].split('.')
        pytorch_major = int(version_parts[0])
        
        if pytorch_major >= 2:
            # PyTorch 2.x（推荐配置）
            if use_pytorch1:
                # 如果要求使用PyTorch 1.x，但检测到PyTorch 2.x
                error_msg = (
                    f"\n{'='*80}\n"
                    f"⚠️  PyTorch版本配置冲突！\n"
                    f"{'='*80}\n"
                    f"当前PyTorch版本: {pytorch_version} (2.x)\n"
                    f"但环境变量 DIFFBEV_USE_PYTORCH1=1 已设置，要求使用PyTorch 1.x\n\n"
                    f"解决方案:\n"
                    f"1. 使用PyTorch 2.x（推荐，支持新GPU架构）:\n"
                    f"   unset DIFFBEV_USE_PYTORCH1\n"
                    f"   然后继续使用\n\n"
                    f"2. 降级到PyTorch 1.x（仅用于旧配置）:\n"
                    f"   conda install pytorch==1.9.1 torchvision==0.10.1 cudatoolkit=11.1 -c pytorch\n"
                    f"   然后继续使用\n"
                    f"{'='*80}\n"
                )
                warnings.warn(error_msg)
                # 不抛出异常，允许继续（向后兼容）
            else:
                # PyTorch 2.x + MMCV 2.x（推荐配置）
                # 静默使用，不需要警告
                pass
        else:
            # PyTorch 1.x
            if not use_pytorch1:
                # 如果要求使用PyTorch 2.x（默认），但检测到PyTorch 1.x
                warning_msg = (
                    f"\n{'='*80}\n"
                    f"ℹ️  信息: 检测到PyTorch 1.x，但项目默认使用PyTorch 2.x\n"
                    f"{'='*80}\n"
                    f"当前PyTorch版本: {pytorch_version} (1.x)\n"
                    f"项目推荐版本: PyTorch >= 2.0.0（支持新GPU架构）\n\n"
                    f"建议:\n"
                    f"1. 升级到PyTorch 2.x（推荐）:\n"
                    f"   conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia\n"
                    f"   或访问 https://pytorch.org/get-started/locally/\n\n"
                    f"2. 如果必须使用PyTorch 1.x，设置环境变量:\n"
                    f"   export DIFFBEV_USE_PYTORCH1=1\n"
                    f"   然后继续使用\n"
                    f"{'='*80}\n"
                )
                warnings.warn(warning_msg)
                # 不抛出异常，允许继续（向后兼容）
            else:
                # PyTorch 1.x + MMCV 1.x（兼容配置）
                # 静默使用，不需要警告
                pass
    except ImportError:
        # PyTorch可能还未安装，这里不报错，让后续代码处理
        pass

# 在模块导入时执行检查（尽早发现不兼容问题）
_check_pytorch_version()

# 为mmcv添加is_list_of的全局兼容实现（在MMCV 2.x中已被移除）
import mmcv
import os
if not hasattr(mmcv, 'is_list_of'):
    try:
        from mmengine.utils import is_list_of as _is_list_of
        mmcv.is_list_of = _is_list_of
    except ImportError:
        # 如果都失败，提供简化实现
        def _is_list_of(seq, type_or_types):
            """检查序列是否为指定类型的列表"""
            if not isinstance(seq, list):
                return False
            if isinstance(type_or_types, tuple):
                return all(isinstance(x, t) for x in seq for t in type_or_types)
            return all(isinstance(x, type_or_types) for x in seq)
        mmcv.is_list_of = _is_list_of

# 为mmcv添加scandir的全局兼容实现（在MMCV 2.x中已被移除）
if not hasattr(mmcv, 'scandir'):
    try:
        from mmcv.fileio import scandir as _scandir
        mmcv.scandir = _scandir
    except ImportError:
        try:
            from mmengine.fileio import scandir as _scandir
            mmcv.scandir = _scandir
        except ImportError:
            # 如果都失败，提供简化实现
            def _scandir(dir_path, suffix=None, recursive=False):
                """扫描目录并返回文件列表"""
                if not os.path.isdir(dir_path):
                    return []
                
                results = []
                if recursive:
                    for root, dirs, files in os.walk(dir_path):
                        for file in files:
                            if suffix is None or file.endswith(suffix):
                                rel_path = os.path.relpath(os.path.join(root, file), dir_path)
                                results.append(rel_path)
                else:
                    for file in os.listdir(dir_path):
                        file_path = os.path.join(dir_path, file)
                        if os.path.isfile(file_path):
                            if suffix is None or file.endswith(suffix):
                                results.append(file)
                
                return sorted(results)
            mmcv.scandir = _scandir

# 为mmcv添加FileClient的全局兼容实现（在MMCV 2.x中已被移除）
if not hasattr(mmcv, 'FileClient'):
    try:
        from mmcv.fileio import FileClient as _FileClient
        mmcv.FileClient = _FileClient
    except ImportError:
        try:
            from mmengine.fileio import FileClient as _FileClient
            mmcv.FileClient = _FileClient
        except ImportError:
            # 如果都失败，提供一个简化的FileClient实现
            class _FileClient(object):
                """简化的FileClient实现（兼容MMCV 1.x接口）"""
                def __init__(self, backend='disk', **kwargs):
                    self.backend = backend
                    self.kwargs = kwargs
                
                def get(self, filepath):
                    """读取文件内容"""
                    if self.backend == 'disk':
                        with open(filepath, 'rb') as f:
                            return f.read()
                    else:
                        raise NotImplementedError(f"FileClient backend '{self.backend}' not supported in compatibility layer")
                
                def put(self, obj, filepath):
                    """写入文件内容"""
                    if self.backend == 'disk':
                        with open(filepath, 'wb') as f:
                            f.write(obj)
                    else:
                        raise NotImplementedError(f"FileClient backend '{self.backend}' not supported in compatibility layer")
                
                def exists(self, filepath):
                    """检查文件是否存在"""
                    if self.backend == 'disk':
                        return os.path.exists(filepath)
                    else:
                        raise NotImplementedError(f"FileClient backend '{self.backend}' not supported in compatibility layer")
                
                def isdir(self, filepath):
                    """检查是否为目录"""
                    if self.backend == 'disk':
                        return os.path.isdir(filepath)
                    else:
                        raise NotImplementedError(f"FileClient backend '{self.backend}' not supported in compatibility layer")
                
                def isfile(self, filepath):
                    """检查是否为文件"""
                    if self.backend == 'disk':
                        return os.path.isfile(filepath)
                    else:
                        raise NotImplementedError(f"FileClient backend '{self.backend}' not supported in compatibility layer")
                
                def list_dir_or_file(self, dir_path, list_dir=True, list_file=True, suffix=None, recursive=False):
                    """列出目录或文件"""
                    if self.backend == 'disk':
                        return mmcv.scandir(dir_path, suffix=suffix, recursive=recursive)
                    else:
                        raise NotImplementedError(f"FileClient backend '{self.backend}' not supported in compatibility layer")
            
            mmcv.FileClient = _FileClient


def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions. For pre-release
    versions: alpha < beta < rc.

    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Default: 4.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    version = parse(version_str)
    assert version.release, f'failed to parse version {version_str}'
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {'a': -3, 'b': -2, 'rc': -1}
        val = -4
        # version.pre can be None
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(f'unknown prerelease version {version.pre[0]}, '
                              'version checking may go wrong')
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])
    else:
        release.extend([0, 0])
    return tuple(release)


mmcv_min_version = digit_version(MMCV_MIN)
mmcv_max_version = digit_version(MMCV_MAX)
mmcv_version = digit_version(mmcv.__version__)

# 检查MMCV版本
if not (mmcv_min_version <= mmcv_version <= mmcv_max_version):
    # 如果是MMCV 2.x但配置要求1.x，给出提示
    if USE_MMCV2 and mmcv_version[0] < 2:
        error_msg = (
            f'\n{"="*80}\n'
            f'❌ MMCV版本不兼容！\n'
            f'{"="*80}\n'
            f'当前MMCV版本: {mmcv.__version__} (1.x)\n'
            f'项目要求: MMCV >= {MMCV_MIN}, <= {MMCV_MAX} (2.x)\n\n'
            f'请升级MMCV:\n'
            f'  pip install --upgrade mmcv>=2.0.0\n\n'
            f'如果需要使用MMCV 1.x（用于旧GPU/PyTorch 1.x），请设置:\n'
            f'  export DIFFBEV_USE_MMCV1=1\n'
            f'{"="*80}\n'
        )
        raise AssertionError(error_msg)
    elif not USE_MMCV2 and mmcv_version[0] >= 2:
        # 如果是MMCV 2.x但配置要求1.x，给出提示
        error_msg = (
            f'\n{"="*80}\n'
            f'⚠️  MMCV版本配置冲突！\n'
            f'{"="*80}\n'
            f'当前MMCV版本: {mmcv.__version__} (2.x)\n'
            f'但环境变量 DIFFBEV_USE_MMCV1=1 已设置，要求使用MMCV 1.x\n\n'
            f'解决方案:\n'
            f'1. 使用MMCV 2.x（推荐，支持PyTorch 2.x和新GPU）:\n'
            f'   unset DIFFBEV_USE_MMCV1\n'
            f'   然后继续使用\n\n'
            f'2. 降级到MMCV 1.x（仅用于旧GPU/PyTorch 1.x）:\n'
            f'   pip install mmcv-full==1.4.0\n'
            f'{"="*80}\n'
        )
        raise AssertionError(error_msg)
    else:
        # 版本不匹配
        error_msg = (
            f'\n{"="*80}\n'
            f'❌ MMCV版本不兼容！\n'
            f'{"="*80}\n'
            f'当前MMCV版本: {mmcv.__version__}\n'
            f'要求版本: mmcv>={MMCV_MIN}, <={MMCV_MAX}\n\n'
            f'请安装兼容的MMCV版本。\n'
            f'{"="*80}\n'
        )
        raise AssertionError(error_msg)

# 成功检测到兼容版本，输出版本信息（如果是MMCV 2.x）
if USE_MMCV2 and mmcv_version[0] >= 2:
    # MMCV 2.x已成功加载，这是推荐配置
    # 在DEBUG模式下输出版本信息
    import logging
    logger = logging.getLogger(__name__)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f'✓ MMCV {mmcv.__version__} (2.x) loaded successfully')
elif not USE_MMCV2 and mmcv_version[0] < 2:
    # MMCV 1.x（兼容旧配置）
    pass

__all__ = ['__version__', 'version_info', 'digit_version']
