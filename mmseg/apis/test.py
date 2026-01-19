# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile

from mmengine import Config as mmcv_Config
import mmcv
import numpy as np
import torch
# 尝试导入collect_results_cpu, collect_results_gpu（兼容MMCV 1.x和2.x）
try:
    from mmcv.engine import collect_results_cpu, collect_results_gpu
except ImportError:
    try:
        from mmengine.runner import collect_results_cpu, collect_results_gpu
    except ImportError:
        # 如果都失败，提供简化实现
        import warnings
        warnings.warn("使用collect_results的兼容实现。某些功能可能不完整。")
        def collect_results_cpu(result_part, size, tmpdir=None):
            """在CPU上收集结果（简化实现）"""
            return result_part
        def collect_results_gpu(result_part, size):
            """在GPU上收集结果（简化实现）"""
            return result_part
from mmcv.image import tensor2imgs
# 尝试从 mmengine.dist 导入（mmengine 0.10+）
try:
    from mmengine.dist import get_dist_info
except ImportError:
    try:
        from mmcv.runner import get_dist_info
    except ImportError:
        from mmengine.runner import get_dist_info
# 尝试导入ProgressBar（兼容MMCV 1.x和2.x）
try:
    from mmcv.utils import ProgressBar
except ImportError:
    try:
        from mmengine.utils import ProgressBar
    except ImportError:
        from tqdm import tqdm
        # 创建ProgressBar的兼容包装
        class ProgressBar:
            def __init__(self, total):
                self.pbar = tqdm(total=total)
            def update(self):
                self.pbar.update(1)

# 兼容函数：创建目录（如果不存在）
def mkdir_or_exist(dir_path):
    """创建目录（如果不存在），兼容MMCV 1.x的mkdir_or_exist"""
    if dir_path and not osp.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def np2tmp(array, temp_file_name=None, tmpdir=None):
    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5,
                    tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))
    # 如果指定了tmpdir，使用它；否则使用默认的.efficient_test
    if efficient_test:
        if tmpdir is None:
            tmpdir = '.efficient_test'
        mkdir_or_exist(tmpdir)
    for i, data in enumerate(data_loader):
        # 解包DataContainer（如果存在），因为模型期望的是解包后的数据
        unwrapped_data = {}
        for key, value in data.items():
            # 检查是否是DataContainer
            if hasattr(value, 'data') and hasattr(value, 'stack') and hasattr(value, 'padding_value'):
                # 如果是DataContainer，提取data
                if value.stack:
                    # 如果需要stack，data应该是tensor或已经stack好的数据
                    unwrapped_data[key] = value.data
                else:
                    # 如果不需要stack，data是列表
                    unwrapped_data[key] = value.data
            elif isinstance(value, list):
                # 如果value是列表，检查列表中是否包含DataContainer
                unwrapped_list = []
                for item in value:
                    if hasattr(item, 'data') and hasattr(item, 'stack') and hasattr(item, 'padding_value'):
                        # 列表项是DataContainer
                        if item.stack:
                            unwrapped_list.append(item.data)
                        else:
                            unwrapped_list.append(item.data)
                    else:
                        unwrapped_list.append(item)
                unwrapped_data[key] = unwrapped_list
            else:
                unwrapped_data[key] = value
        
        with torch.no_grad():
            result = model(return_loss=False, **unwrapped_data)

        if show or out_dir:
            # 使用unwrapped_data而不是原始的data
            img_tensor = unwrapped_data.get('img', data.get('img', []))
            if isinstance(img_tensor, (list, tuple)) and len(img_tensor) > 0:
                img_tensor = img_tensor[0]
            
            img_metas = unwrapped_data.get('img_metas', data.get('img_metas', []))
            # 处理img_metas可能是列表的列表的情况
            if isinstance(img_metas, list) and len(img_metas) > 0:
                if isinstance(img_metas[0], list):
                    img_metas = img_metas[0]
                elif hasattr(img_metas[0], 'data'):
                    img_metas = img_metas[0].data
                if len(img_metas) > 0 and isinstance(img_metas[0], list):
                    img_metas = img_metas[0]
            
            if not img_metas or not isinstance(img_metas[0], dict):
                continue  # 跳过如果无法处理
                
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file,
                    opacity=opacity)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_, tmpdir=tmpdir) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result, tmpdir=tmpdir)
            results.append(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = ProgressBar(len(dataset))
    # 如果指定了efficient_test_tmpdir，使用它；否则使用默认的.efficient_test
    if efficient_test:
        if efficient_test_tmpdir is None:
            efficient_test_tmpdir = '.efficient_test'
        mkdir_or_exist(efficient_test_tmpdir)
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_, tmpdir=tmpdir) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result, tmpdir=tmpdir)
            results.append(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results
