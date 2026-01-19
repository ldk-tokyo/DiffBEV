"""
兼容性检查工具模块

本模块提供版本兼容性检查功能，确保项目在正确的环境中运行。
"""
import warnings
import sys


def check_pytorch_version(raise_error=True, allow_pytorch1=None):
    """
    检查PyTorch版本是否兼容
    
    Args:
        raise_error (bool): 如果为True，不兼容时抛出异常；如果为False，仅返回结果
        allow_pytorch1 (bool, optional): 如果为None，自动检测；如果为True/False，强制使用该配置
        
    Returns:
        bool: 如果版本兼容返回True，否则返回False
        
    Raises:
        RuntimeError: 如果版本不兼容且raise_error=True
    """
    try:
        import torch
    except ImportError:
        error_msg = "PyTorch未安装，请先安装PyTorch >= 2.0.0 (推荐) 或 1.9.1 (兼容)"
        if raise_error:
            raise RuntimeError(error_msg)
        return False
    
    import os
    pytorch_version = torch.__version__
    # 处理版本号格式，如 "1.9.1+cu111" 或 "2.5.0"
    version_parts = pytorch_version.split('+')[0].split('.')
    pytorch_major = int(version_parts[0])
    
    # 自动检测是否使用PyTorch 1.x（默认使用PyTorch 2.x）
    if allow_pytorch1 is None:
        allow_pytorch1 = os.environ.get('DIFFBEV_USE_PYTORCH1', '0') == '1'
    
    if pytorch_major >= 2:
        # PyTorch 2.x（推荐配置）
        if allow_pytorch1:
            # 如果要求使用PyTorch 1.x，但检测到PyTorch 2.x
            error_msg = (
                f"检测到PyTorch {pytorch_version} (2.x)，但DIFFBEV_USE_PYTORCH1=1已设置。\n"
                f"如需使用PyTorch 2.x（推荐），请取消设置该环境变量。"
            )
            if raise_error:
                warnings.warn(error_msg)  # 仅警告，不抛出异常（向后兼容）
            return True
        else:
            # PyTorch 2.x（推荐配置）
            return True
    else:
        # PyTorch 1.x（兼容配置）
        if not allow_pytorch1:
            # 如果要求使用PyTorch 2.x（默认），但检测到PyTorch 1.x
            warning_msg = (
                f"检测到PyTorch {pytorch_version} (1.x)，但项目推荐使用PyTorch 2.x。\n"
                f"建议升级到PyTorch 2.x以获得更好的性能和新GPU支持。\n"
                f"如需继续使用PyTorch 1.x，请设置: export DIFFBEV_USE_PYTORCH1=1"
            )
            if raise_error:
                warnings.warn(warning_msg)  # 仅警告，不抛出异常（向后兼容）
            return True
        else:
            # PyTorch 1.x（兼容配置）
            return True


def check_mmcv_version(raise_error=False, allow_mmcv2=None):
    """
    检查MMCV版本是否兼容
    
    Args:
        raise_error (bool): 如果为True，不兼容时抛出异常；如果为False，仅警告
        allow_mmcv2 (bool, optional): 如果为None，自动检测；如果为True/False，强制使用该配置
        
    Returns:
        bool: 如果版本兼容返回True，否则返回False
    """
    try:
        import mmcv
        import os
        mmcv_version = mmcv.__version__
        version_parts = mmcv_version.split('.')
        mmcv_major = int(version_parts[0])
        mmcv_minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        
        # 自动检测是否使用MMCV 2.x（默认支持2.x）
        if allow_mmcv2 is None:
            allow_mmcv2 = os.environ.get('DIFFBEV_USE_MMCV1', '0') != '1'
        
        # 检查是否在推荐范围内
        if mmcv_major == 1 and 3 <= mmcv_minor < 4:
            # mmcv 1.3.x - 旧版本（PyTorch 1.x）
            if allow_mmcv2:
                # 如果要求使用2.x，但安装的是1.x
                error_msg = (
                    f"检测到MMCV {mmcv_version} (1.x)，但项目要求MMCV 2.x。\n"
                    f"请升级: pip install --upgrade mmcv>=2.0.0"
                )
                if raise_error:
                    raise RuntimeError(error_msg)
                else:
                    warnings.warn(error_msg)
                return False
            return True
        elif mmcv_major >= 2:
            # mmcv 2.x - 新版本（PyTorch 2.x，推荐）
            if not allow_mmcv2:
                # 如果要求使用1.x，但安装的是2.x
                error_msg = (
                    f"检测到MMCV {mmcv_version} (2.x)，但DIFFBEV_USE_MMCV1=1已设置。\n"
                    f"如需使用MMCV 2.x，请取消设置该环境变量。"
                )
                if raise_error:
                    raise RuntimeError(error_msg)
                else:
                    warnings.warn(error_msg)
                return False
            # 检查是否安装了mmengine（必需）
            try:
                import mmengine
                return True
            except ImportError:
                error_msg = (
                    f"MMCV {mmcv_version} (2.x) 已安装，但缺少必需的 mmengine 模块\n"
                    f"请运行: pip install mmengine>=0.10.0"
                )
                if raise_error:
                    raise RuntimeError(error_msg)
                else:
                    warnings.warn(error_msg)
                return False
        else:
            # 不兼容的版本
            warning_msg = (
                f"MMCV版本 {mmcv_version} 不在支持范围内\n"
                f"推荐版本: mmcv>=2.0.0 (PyTorch 2.x，推荐) 或 mmcv-full==1.4.0 (PyTorch 1.x)"
            )
            if raise_error:
                raise RuntimeError(warning_msg)
            else:
                warnings.warn(warning_msg)
            return False
    except ImportError:
        error_msg = "MMCV未安装，请安装: pip install mmcv>=2.0.0"
        if raise_error:
            raise RuntimeError(error_msg)
        else:
            warnings.warn(error_msg)
        return False


def check_environment(raise_on_pytorch_error=True, raise_on_mmcv_error=False):
    """
    完整的环境兼容性检查
    
    Args:
        raise_on_pytorch_error (bool): PyTorch版本不兼容时是否抛出异常
        raise_on_mmcv_error (bool): MMCV版本不兼容时是否抛出异常
        
    Returns:
        dict: 检查结果字典，包含各组件状态
    """
    results = {
        'pytorch_ok': False,
        'mmcv_ok': False,
        'pytorch_version': None,
        'mmcv_version': None,
    }
    
    # 检查PyTorch
    try:
        import torch
        results['pytorch_version'] = torch.__version__
        results['pytorch_ok'] = check_pytorch_version(raise_error=raise_on_pytorch_error)
    except Exception as e:
        if raise_on_pytorch_error:
            raise
        results['pytorch_ok'] = False
    
    # 检查MMCV
    try:
        import mmcv
        results['mmcv_version'] = mmcv.__version__
        results['mmcv_ok'] = check_mmcv_version(raise_error=raise_on_mmcv_error)
    except Exception as e:
        if raise_on_mmcv_error:
            raise
        results['mmcv_ok'] = False
    
    return results


def check_gpu_compatibility():
    """
    检查GPU兼容性，检测新架构GPU（如RTX 5090）
    
    Returns:
        tuple: (is_new_gpu, gpu_name, gpu_arch)
            - is_new_gpu (bool): 是否为新架构GPU
            - gpu_name (str): GPU名称
            - gpu_arch (str): GPU架构信息（如sm_120）
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False, None, None
        
        # 获取GPU名称
        gpu_name = torch.cuda.get_device_name(0)
        
        # 获取计算能力
        compute_capability = torch.cuda.get_device_capability(0)
        compute_capability_str = f"sm_{compute_capability[0]}{compute_capability[1]}"
        
        # 检查是否为新架构（sm_100+，如RTX 5090的sm_120）
        is_new_gpu = compute_capability[0] >= 10
        
        return is_new_gpu, gpu_name, compute_capability_str
    except Exception:
        # 如果无法检测，返回默认值
        return False, None, None
