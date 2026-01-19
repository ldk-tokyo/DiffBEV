# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import sys
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
import os.path as osp
import time
import warnings
import torch.distributed as dist
import mmcv
import torch

# 尝试导入revert_sync_batchnorm（兼容MMCV 1.x和2.x）
try:
    from mmcv.cnn.utils import revert_sync_batchnorm
except ImportError:
    try:
        # MMCV 2.x可能在mmengine.model中
        from mmengine.model import revert_sync_batchnorm
    except ImportError:
        # 如果都失败，使用PyTorch内置实现
        def revert_sync_batchnorm(module):
            """
            将SyncBatchNorm转换为BatchNorm2d（兼容性函数）
            
            Args:
                module: PyTorch模型或模块
                
            Returns:
                转换后的模块
            """
            module_output = module
            if isinstance(module, torch.nn.SyncBatchNorm):
                module_output = torch.nn.BatchNorm2d(
                    module.num_features,
                    module.eps,
                    module.momentum,
                    module.affine,
                    module.track_running_stats
                )
                if module.affine:
                    module_output.weight.data = module.weight.data.clone().detach()
                    module_output.bias.data = module.bias.data.clone().detach()
                    module_output.weight.requires_grad = module.weight.requires_grad
                    module_output.bias.requires_grad = module.bias.requires_grad
                module_output.running_mean = module.running_mean
                module_output.running_var = module.running_var
                module_output.num_batches_tracked = module.num_batches_tracked
            
            for name, child in module.named_children():
                module_output.add_module(name, revert_sync_batchnorm(child))
            
            return module_output

# 尝试从mmengine导入，如果失败则从mmcv导入（兼容性处理）
try:
    from mmengine import Config, DictAction
    from mmengine.utils import get_git_hash
except ImportError:
    # 如果mmengine不可用，使用mmcv（旧版本）
    from mmcv.utils import Config, DictAction, get_git_hash

# 尝试导入 get_dist_info 和 init_dist（兼容不同版本）
try:
    # 首先尝试从 mmengine.dist（mmengine 0.10+）
    from mmengine.dist import get_dist_info, init_dist
except ImportError:
    try:
        # 尝试从 mmcv.runner（mmcv 1.x）
        from mmcv.runner import get_dist_info, init_dist
    except ImportError:
        try:
            # 尝试从 mmengine.runner（某些 mmengine 版本）
            from mmengine.runner import get_dist_info, init_dist
        except ImportError:
            # 如果都失败，使用 torch.distributed 的替代实现
            def get_dist_info():
                if dist.is_initialized():
                    rank = dist.get_rank()
                    world_size = dist.get_world_size()
                else:
                    rank = 0
                    world_size = 1
                return rank, world_size
            
            def init_dist(launcher, backend='nccl', **kwargs):
                if launcher == 'pytorch':
                    dist.init_process_group(backend=backend, **kwargs)
                elif launcher == 'slurm':
                    # SLURM 初始化
                    rank = int(os.environ['SLURM_PROCID'])
                    world_size = int(os.environ['SLURM_NPROCS'])
                    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, **kwargs)
                else:
                    raise ValueError(f'Unknown launcher type: {launcher}')

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger
from mmseg.utils.compat import check_pytorch_version, check_mmcv_version
import pdb


def check_environment_compatibility():
    """
    检查环境兼容性，特别是PyTorch版本
    
    该函数在训练启动时检查：
    1. PyTorch版本（默认使用 >= 2.0，推荐配置；可通过环境变量使用1.x）
    2. MMCV版本（默认使用 >= 2.0，推荐配置；可通过环境变量使用1.x）
    3. GPU兼容性（检测新架构GPU如RTX 5090）
    4. 版本兼容性
    
    使用mmseg.utils.compat模块进行检查。
    
    Raises:
        RuntimeError: 如果检测到不兼容的PyTorch版本
    """
    import os
    
    # 检查GPU
    from mmseg.utils.compat import check_gpu_compatibility
    is_new_gpu, gpu_name, gpu_arch = check_gpu_compatibility()
    
    if gpu_name:
        print(f"\n检测到GPU: {gpu_name}")
        if is_new_gpu:
            print(f"ℹ️  新架构GPU检测（支持PyTorch 2.x）")
            if gpu_arch:
                print(f"   架构信息: {gpu_arch}")
    
    # 使用统一的兼容性检查函数
    # 默认使用PyTorch 2.x（推荐配置），函数会自动检测环境变量
    check_pytorch_version(raise_error=True)
    check_mmcv_version(raise_error=False)
    
    # 输出环境信息
    import torch
    pytorch_version = torch.__version__
    print("\n" + "="*80)
    print("环境兼容性检查通过")
    print("="*80)
    print(f"PyTorch版本: {pytorch_version} ✓")
    try:
        import mmcv
        print(f"MMCV版本: {mmcv.__version__} ✓")
    except ImportError:
        print("MMCV: 将在后续检查中验证")
    print("="*80 + "\n")


def check_nuscenes_dataset_path(cfg):
    """
    检查nuScenes数据集路径和相关目录是否存在
    
    Args:
        cfg: 配置对象
        
    Raises:
        AssertionError: 如果数据集路径或必要目录不存在
    """
    # 统一的数据集路径（检查大小写变体）
    # 首先尝试小写路径，如果不存在则尝试大写S路径
    nuscenes_lower = "/media/ldk950413/data0/nuscenes"
    nuscenes_upper = "/media/ldk950413/data0/nuScenes"
    
    if osp.exists(nuscenes_lower):
        NUSCENES_DATA_ROOT = nuscenes_lower
    elif osp.exists(nuscenes_upper):
        NUSCENES_DATA_ROOT = nuscenes_upper
        print(f"\n注意: 检测到数据集路径使用大写S: {NUSCENES_DATA_ROOT}")
        print(f"将自动更新配置中的data_root路径\n")
    else:
        NUSCENES_DATA_ROOT = nuscenes_lower  # 默认使用小写路径（用于错误提示）
    
    # 检查数据集根目录是否存在
    if not osp.exists(NUSCENES_DATA_ROOT):
        raise AssertionError(
            f"nuScenes数据集根目录不存在: {NUSCENES_DATA_ROOT}\n"
            f"请确保数据集已正确下载并放置在指定路径。"
        )
    
    # 检查配置文件中的data_root是否与统一路径一致
    if hasattr(cfg, 'data') and 'train' in cfg.data:
        train_data_root = cfg.data.train.get('data_root', None)
        if train_data_root and train_data_root.rstrip('/') != NUSCENES_DATA_ROOT.rstrip('/'):
            print(f"警告: 配置文件中的data_root ({train_data_root}) 与统一路径 ({NUSCENES_DATA_ROOT}) 不一致")
            print(f"将使用统一路径: {NUSCENES_DATA_ROOT}")
            # 更新配置为统一路径
            if 'train' in cfg.data:
                cfg.data.train['data_root'] = NUSCENES_DATA_ROOT
            if 'val' in cfg.data:
                cfg.data.val['data_root'] = NUSCENES_DATA_ROOT
            if 'test' in cfg.data:
                cfg.data.test['data_root'] = NUSCENES_DATA_ROOT
    
    # 检查必要的nuScenes目录结构
    required_dirs = []
    required_files = []
    
    # 检查samples目录（nuScenes原始数据）
    samples_dir = osp.join(NUSCENES_DATA_ROOT, 'samples')
    if osp.exists(samples_dir):
        required_dirs.append(('samples', samples_dir, True))
    else:
        required_dirs.append(('samples', samples_dir, False))
    
    # 检查sweeps目录（nuScenes原始数据）
    sweeps_dir = osp.join(NUSCENES_DATA_ROOT, 'sweeps')
    if osp.exists(sweeps_dir):
        required_dirs.append(('sweeps', sweeps_dir, True))
    else:
        required_dirs.append(('sweeps', sweeps_dir, False))
    
    # 检查v1.0-trainval目录（nuScenes版本目录）
    v1_trainval_dir = osp.join(NUSCENES_DATA_ROOT, 'v1.0-trainval')
    if osp.exists(v1_trainval_dir):
        required_dirs.append(('v1.0-trainval', v1_trainval_dir, True))
    else:
        required_dirs.append(('v1.0-trainval', v1_trainval_dir, False))
    
    # 检查BEV处理后的数据目录
    img_dir_train = osp.join(NUSCENES_DATA_ROOT, 'img_dir', 'train')
    ann_dir_train = osp.join(NUSCENES_DATA_ROOT, 'ann_bev_dir', 'train')
    img_dir_val = osp.join(NUSCENES_DATA_ROOT, 'img_dir', 'val')
    ann_dir_val = osp.join(NUSCENES_DATA_ROOT, 'ann_bev_dir', 'val')
    
    required_dirs.extend([
        ('img_dir/train', img_dir_train, osp.exists(img_dir_train)),
        ('ann_bev_dir/train', ann_dir_train, osp.exists(ann_dir_train)),
        ('img_dir/val', img_dir_val, osp.exists(img_dir_val)),
        ('ann_bev_dir/val', ann_dir_val, osp.exists(ann_dir_val)),
    ])
    
    # 检查calib.json
    calib_file = osp.join(NUSCENES_DATA_ROOT, 'calib.json')
    required_files.append(('calib.json', calib_file, osp.exists(calib_file)))
    
    # 收集缺失的目录和文件
    missing_dirs = [(name, path) for name, path, exists in required_dirs if not exists]
    missing_files = [(name, path) for name, path, exists in required_files if not exists]
    
    # 对于关键的nuScenes原始数据目录（samples、sweeps、v1.0-trainval），必须存在
    critical_missing = [(name, path) for name, path, exists in required_dirs[:3] if not exists]
    
    if critical_missing:
        error_msg = "\n" + "="*80 + "\n"
        error_msg += "错误: nuScenes数据集关键目录缺失！\n"
        error_msg += "="*80 + "\n"
        error_msg += f"数据集根目录: {NUSCENES_DATA_ROOT}\n"
        error_msg += "\n缺失的关键目录:\n"
        for name, path in critical_missing:
            error_msg += f"  - {name}: {path}\n"
        error_msg += "\n请确保nuScenes v1.0-trainval数据集已正确下载并解压。\n"
        error_msg += "="*80 + "\n"
        raise AssertionError(error_msg)
    
    # 对于其他目录，给出警告但继续执行
    if missing_dirs[3:] or missing_files:
        warning_msg = "\n" + "-"*80 + "\n"
        warning_msg += "警告: 部分数据目录或文件缺失\n"
        warning_msg += "-"*80 + "\n"
        if missing_dirs[3:]:
            warning_msg += "\n缺失的目录:\n"
            for name, path in missing_dirs[3:]:
                warning_msg += f"  - {name}: {path}\n"
        if missing_files:
            warning_msg += "\n缺失的文件:\n"
            for name, path in missing_files:
                warning_msg += f"  - {name}: {path}\n"
        warning_msg += "\n这些目录/文件可能需要在训练前生成（如BEV标注处理）。\n"
        warning_msg += "-"*80 + "\n"
        print(warning_msg)
    
    # 输出成功信息
    print("\n" + "="*80)
    print("nuScenes数据集路径检查通过")
    print("="*80)
    print(f"数据集根目录: {NUSCENES_DATA_ROOT}")
    print("\n存在的关键目录:")
    for name, path, exists in required_dirs[:3]:
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")
    print("="*80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from',help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from',help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    
    # 首先检查环境兼容性（PyTorch版本等）
    check_environment_compatibility()
    
    # 检查nuScenes数据集路径（在配置加载后立即检查，必须执行）
    # 检查配置中是否使用了NuscenesDataset类型
    is_nuscenes_config = False
    if 'nuscene' in args.config.lower() or 'nuscenes' in args.config.lower():
        is_nuscenes_config = True
    elif hasattr(cfg, 'data') and cfg.data:
        # 检查数据配置中是否包含NuscenesDataset
        train_type = cfg.data.get('train', {}).get('type', '')
        if 'NuscenesDataset' in str(train_type) or 'Nuscenes' in str(train_type):
            is_nuscenes_config = True
    
    # 如果是nuScenes配置，必须执行路径检查（使用assert确保失败时退出）
    if is_nuscenes_config:
        try:
            check_nuscenes_dataset_path(cfg)
        except AssertionError as e:
            print(str(e))
            sys.exit(1)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # gpu_ids is used to calculate iter when resuming checkpoint
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    # 尝试导入mkdir_or_exist（兼容MMCV 1.x和2.x）
    try:
        mkdir_or_exist = mmcv.mkdir_or_exist
    except AttributeError:
        try:
            from mmengine.fileio import mkdir_or_exist
        except ImportError:
            try:
                from mmcv.fileio import mkdir_or_exist
            except ImportError:
                # 如果都失败，提供简化实现
                import os
                def mkdir_or_exist(dir_path):
                    """创建目录（如果不存在）"""
                    os.makedirs(dir_path, exist_ok=True)
    mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # SyncBN is not support for DP
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        model = revert_sync_batchnorm(model)
        
    # print(model) model is the arch of the network, including backbone,neck and decode_head
    logger.info(model)
    datasets = [build_dataset(cfg.data.train)]
    # datasets is a list, dataset = [<mmseg.datasets.nuscenes.NuscenesDataset object at 0x7f87ab274190>]
    # datasets[0].CLASSES and datasets[0].PALETTE are the same with configs
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    # passing checkpoint meta for saving best checkpoint
    meta.update(cfg.checkpoint_config.meta)
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
