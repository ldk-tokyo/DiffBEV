# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform
import random
from functools import partial
import pdb
import numpy as np
import torch
# 尝试从 mmcv.parallel 导入（mmcv 1.x），如果失败则尝试其他位置
try:
    from mmcv.parallel import collate
    _collate_supports_samples_per_gpu = True
except ImportError:
    try:
        from mmengine.parallel import collate
        _collate_supports_samples_per_gpu = True
    except ImportError:
        # 如果都失败，创建一个兼容的collate函数，支持samples_per_gpu参数和DataContainer
        from torch.utils.data.dataloader import default_collate as _default_collate
        _collate_supports_samples_per_gpu = False
        
        # 尝试导入DataContainer以检查batch中是否包含它
        try:
            from mmcv.parallel import DataContainer as _DataContainer
        except ImportError:
            try:
                from mmengine.parallel import DataContainer as _DataContainer
            except ImportError:
                try:
                    from mmcv.utils import DataContainer as _DataContainer
                except ImportError:
                    try:
                        from mmengine.utils import DataContainer as _DataContainer
                    except ImportError:
                        # 如果DataContainer不存在，使用简化版本
                        _DataContainer = None
        
        def _is_datacontainer(value):
            """检查值是否是DataContainer类型"""
            if value is None:
                return False
            return (hasattr(value, 'data') and 
                    hasattr(value, 'stack') and 
                    hasattr(value, 'padding_value'))
        
        def collate(batch, samples_per_gpu=1):
            """兼容的collate函数，支持samples_per_gpu参数和DataContainer"""
            if not batch:
                return batch
            
            # 检查batch中的元素是否是字典，以及是否包含DataContainer
            if isinstance(batch[0], dict):
                # 检查是否包含DataContainer
                has_datacontainer = False
                for item in batch:
                    for key, value in item.items():
                        if _is_datacontainer(value):
                            has_datacontainer = True
                            break
                    if has_datacontainer:
                        break
                
                if has_datacontainer:
                    # 使用MMCV/MMEngine风格的collate处理DataContainer
                    result = {}
                    for key in batch[0].keys():
                        values = [item[key] for item in batch]
                        
                        # 检查是否是DataContainer
                        if len(values) > 0 and _is_datacontainer(values[0]):
                            # 处理DataContainer
                            dc = values[0]
                            if dc.stack:
                                # 如果需要stack，提取data并处理
                                data_list = [dc.data for dc in values]
                                # 对于data_list，使用default_collate处理（因为data已经是实际数据）
                                try:
                                    stacked = _default_collate(data_list)
                                except (TypeError, RuntimeError) as e:
                                    # 如果default_collate失败（可能是因为data_list包含不支持的类型），
                                    # 尝试手动处理
                                    if len(data_list) > 0:
                                        if isinstance(data_list[0], torch.Tensor):
                                            stacked = torch.stack(data_list, dim=0)
                                        elif isinstance(data_list[0], np.ndarray):
                                            stacked = np.stack(data_list, axis=0)
                                            stacked = torch.from_numpy(stacked)
                                        elif isinstance(data_list[0], (list, tuple)):
                                            # 对于列表/元组，尝试递归处理
                                            try:
                                                stacked = _default_collate(data_list)
                                            except (TypeError, RuntimeError):
                                                stacked = data_list
                                        else:
                                            stacked = data_list
                                    else:
                                        stacked = data_list
                                
                                # 创建新的DataContainer
                                result[key] = type(dc)(
                                    stacked,
                                    stack=dc.stack,
                                    padding_value=dc.padding_value,
                                    cpu_only=dc.cpu_only
                                )
                            else:
                                # 如果不需要stack，直接返回列表
                                result[key] = type(dc)(
                                    [dc.data for dc in values],
                                    stack=dc.stack,
                                    padding_value=dc.padding_value,
                                    cpu_only=dc.cpu_only
                                )
                        else:
                            # 不是DataContainer，使用default_collate
                            # 但如果values中仍然包含DataContainer，需要递归处理
                            if len(values) > 0 and isinstance(values[0], (list, tuple)):
                                # 检查列表中是否包含DataContainer
                                nested_has_dc = False
                                for v in values:
                                    if isinstance(v, (list, tuple)):
                                        for item in v:
                                            if _is_datacontainer(item):
                                                nested_has_dc = True
                                                break
                                    if nested_has_dc:
                                        break
                                
                                if nested_has_dc:
                                    # 递归处理嵌套的DataContainer
                                    # 对于列表的列表，转置后递归处理
                                    if all(isinstance(v, (list, tuple)) and len(v) == len(values[0]) for v in values):
                                        transposed = [[v[i] for v in values] for i in range(len(values[0]))]
                                        # 对于每个转置后的列表，使用我们的collate函数递归处理
                                        processed = []
                                        for trans_item in transposed:
                                            if isinstance(trans_item[0], dict):
                                                # 字典类型，使用collate递归处理
                                                processed.append(collate(trans_item, samples_per_gpu))
                                            elif _is_datacontainer(trans_item[0]):
                                                # DataContainer类型，需要特殊处理
                                                dc = trans_item[0]
                                                if dc.stack:
                                                    data_list = [dc.data for dc in trans_item]
                                                    try:
                                                        stacked = _default_collate(data_list)
                                                    except (TypeError, RuntimeError):
                                                        stacked = data_list
                                                    processed.append(type(dc)(
                                                        stacked,
                                                        stack=dc.stack,
                                                        padding_value=dc.padding_value,
                                                        cpu_only=dc.cpu_only
                                                    ))
                                                else:
                                                    processed.append(type(dc)(
                                                        [dc.data for dc in trans_item],
                                                        stack=dc.stack,
                                                        padding_value=dc.padding_value,
                                                        cpu_only=dc.cpu_only
                                                    ))
                                            else:
                                                # 普通类型，使用default_collate
                                                try:
                                                    processed.append(_default_collate(trans_item))
                                                except (TypeError, RuntimeError):
                                                    processed.append(trans_item)
                                        result[key] = processed
                                    else:
                                        result[key] = values
                                else:
                                    # 普通列表，使用default_collate
                                    try:
                                        result[key] = _default_collate(values)
                                    except (TypeError, RuntimeError):
                                        result[key] = values
                            else:
                                try:
                                    result[key] = _default_collate(values)
                                except (TypeError, RuntimeError):
                                    result[key] = values
                    return result
                else:
                    # 没有检测到DataContainer，但为了安全起见，先尝试default_collate
                    # 如果失败并提示DataContainer错误，则用我们的方法处理
                    try:
                        return _default_collate(batch)
                    except TypeError as e:
                        if 'DataContainer' in str(e) or 'mmseg.datasets.pipelines.formatting.DataContainer' in str(e):
                            # 实际上有DataContainer，但我们的检测漏掉了，重新用完整检测逻辑处理
                            # 使用更严格的递归检测
                            def _deep_check_datacontainer(obj, depth=0):
                                """递归检查是否包含DataContainer（最多递归3层）"""
                                if depth > 3:
                                    return False
                                if _is_datacontainer(obj):
                                    return True
                                if isinstance(obj, dict):
                                    return any(_deep_check_datacontainer(v, depth+1) for v in obj.values())
                                elif isinstance(obj, (list, tuple)):
                                    return any(_deep_check_datacontainer(item, depth+1) for item in obj)
                                return False
                            
                            # 重新检查是否有DataContainer
                            if _deep_check_datacontainer(batch):
                                # 有DataContainer，按有DataContainer的路径处理
                                result = {}
                                for key in batch[0].keys():
                                    values = [item[key] for item in batch]
                                    if len(values) > 0 and _is_datacontainer(values[0]):
                                        dc = values[0]
                                        if dc.stack:
                                            data_list = [dc.data for dc in values]
                                            try:
                                                stacked = _default_collate(data_list)
                                            except (TypeError, RuntimeError):
                                                stacked = data_list
                                            result[key] = type(dc)(stacked, stack=dc.stack, 
                                                                   padding_value=dc.padding_value, cpu_only=dc.cpu_only)
                                        else:
                                            result[key] = type(dc)([dc.data for dc in values], 
                                                                   stack=dc.stack, padding_value=dc.padding_value, 
                                                                   cpu_only=dc.cpu_only)
                                    else:
                                        # 尝试递归collate
                                        try:
                                            result[key] = collate(values, samples_per_gpu) if isinstance(values[0], dict) else _default_collate(values)
                                        except (TypeError, RuntimeError):
                                            result[key] = values
                                return result
                            else:
                                # 确实没有DataContainer，重新抛出原始错误
                                raise
                        else:
                            # 其他类型的TypeError，直接抛出
                            raise
            else:
                # 不是字典，检查是否包含DataContainer
                if len(batch) > 0 and _is_datacontainer(batch[0]):
                    dc = batch[0]
                    if dc.stack:
                        data_list = [dc.data for dc in batch]
                        try:
                            stacked = _default_collate(data_list)
                        except (TypeError, RuntimeError):
                            stacked = data_list
                        return type(dc)(
                            stacked,
                            stack=dc.stack,
                            padding_value=dc.padding_value,
                            cpu_only=dc.cpu_only
                        )
                    else:
                        return type(dc)(
                            [dc.data for dc in batch],
                            stack=dc.stack,
                            padding_value=dc.padding_value,
                            cpu_only=dc.cpu_only
                        )
                else:
                    # 不是字典也不是DataContainer，尝试default_collate，如果失败并提到DataContainer则用递归方法
                    try:
                        return _default_collate(batch)
                    except TypeError as e:
                        if 'DataContainer' in str(e) or 'mmseg.datasets.pipelines.formatting.DataContainer' in str(e):
                            # 有DataContainer但检测漏掉了，需要递归检查
                            def _deep_check_datacontainer(obj, depth=0):
                                if depth > 3:
                                    return False
                                if _is_datacontainer(obj):
                                    return True
                                if isinstance(obj, (list, tuple)):
                                    return any(_deep_check_datacontainer(item, depth+1) for item in obj)
                                return False
                            
                            if len(batch) > 0 and _deep_check_datacontainer(batch[0]):
                                # 递归处理包含DataContainer的列表
                                # 这种情况比较复杂，先尝试简单的处理
                                if all(_is_datacontainer(item) for item in batch):
                                    dc = batch[0]
                                    if dc.stack:
                                        return type(dc)(_default_collate([item.data for item in batch]), 
                                                       stack=dc.stack, padding_value=dc.padding_value, cpu_only=dc.cpu_only)
                                    else:
                                        return type(dc)([item.data for item in batch], 
                                                       stack=dc.stack, padding_value=dc.padding_value, cpu_only=dc.cpu_only)
                            raise  # 如果无法处理，重新抛出错误
                        else:
                            raise  # 其他错误直接抛出
# 尝试从 mmengine.dist 导入（mmengine 0.10+）
try:
    from mmengine.dist import get_dist_info
except ImportError:
    try:
        from mmcv.runner import get_dist_info
    except ImportError:
        from mmengine.runner import get_dist_info
from mmengine.registry import Registry, build_from_cfg
# 尝试从 mmengine.utils 导入 digit_version（mmengine 0.10+）
try:
    from mmengine.utils import digit_version
except ImportError:
    try:
        from mmcv.utils import digit_version
    except ImportError:
        # 如果都失败，使用本地实现
        def digit_version(version_str):
            """将版本字符串转换为数字元组"""
            version_digits = []
            for x in version_str.split('.'):
                if x.isdigit():
                    version_digits.append(int(x))
                elif x.find('rc') != -1:
                    patch_version = x.split('rc')
                    version_digits.append(int(patch_version[0]))
                    version_digits.append(int(patch_version[1]) - 10)
            return tuple(version_digits)
from torch.utils.data import DataLoader, DistributedSampler

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def _concat_dataset(cfg, default_args=None):
    """Build :obj:`ConcatDataset by."""
    from .dataset_wrappers import ConcatDataset
    img_dir = cfg['img_dir']
    ann_dir = cfg.get('ann_dir', None)
    split = cfg.get('split', None)
    # pop 'separate_eval' since it is not a valid key for common datasets.
    separate_eval = cfg.pop('separate_eval', True)
    num_img_dir = len(img_dir) if isinstance(img_dir, (list, tuple)) else 1
    if ann_dir is not None:
        num_ann_dir = len(ann_dir) if isinstance(ann_dir, (list, tuple)) else 1
    else:
        num_ann_dir = 0
    if split is not None:
        num_split = len(split) if isinstance(split, (list, tuple)) else 1
    else:
        num_split = 0
    if num_img_dir > 1:
        assert num_img_dir == num_ann_dir or num_ann_dir == 0
        assert num_img_dir == num_split or num_split == 0
    else:
        assert num_split == num_ann_dir or num_ann_dir <= 1
    num_dset = max(num_split, num_img_dir)

    datasets = []
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        if isinstance(img_dir, (list, tuple)):
            data_cfg['img_dir'] = img_dir[i]
        if isinstance(ann_dir, (list, tuple)):
            data_cfg['ann_dir'] = ann_dir[i]
        if isinstance(split, (list, tuple)):
            data_cfg['split'] = split[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets, separate_eval)


def build_dataset(cfg, default_args=None):
    """Build datasets."""
    from .dataset_wrappers import ConcatDataset, RepeatDataset
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif isinstance(cfg.get('img_dir'), (list, tuple)) or isinstance(
            cfg.get('split', None), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)
    # type of dataset: <class 'mmseg.datasets.nuscenes.NuscenesDataset'>
    # dataset.__dict__ are like followings:
    # {'filename': '4880ac912e944d6da91aeee4e89d4deb.png', 'ann': {'seg_map': '4880ac912e944d6da91aeee4e89d4deb.png'}},
    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     drop_last=False,
                     pin_memory=True,
                     persistent_workers=True,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int | None): Seed to be used. Default: None.
        drop_last (bool): Whether to drop the last incomplete batch in epoch.
            Default: False
        pin_memory (bool): Whether to use pin_memory in DataLoader.
            Default: True
        persistent_workers (bool): If True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            This allows to maintain the workers Dataset instances alive.
            The argument also has effect in PyTorch>=1.7.0.
            Default: True
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        sampler = DistributedSampler(
            dataset, world_size, rank, shuffle=shuffle)
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    if digit_version(torch.__version__) >= digit_version('1.8.0'):
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
            pin_memory=pin_memory,
            shuffle=shuffle,
            worker_init_fn=init_fn,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            **kwargs)
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
            pin_memory=pin_memory,
            shuffle=shuffle,
            worker_init_fn=init_fn,
            drop_last=drop_last,
            **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
