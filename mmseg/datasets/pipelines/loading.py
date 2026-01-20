# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from cv2 import decomposeProjectionMatrix
import cv2
from mmengine import Config as mmcv_Config
import mmcv
import numpy as np
import json
import torch
import torchvision
from PIL import Image

from ..builder import PIPELINES
import os
# from tools import heatmap_vis

@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        # using cv2 to read image, so the image format is H,W,C
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

def decode_binary_labels(labels,nclass):
    bits = torch.pow(2,torch.arange(nclass))
    return (labels & bits.view(-1,1,1))>0

@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                reduce_zero_label=False,
                file_client_args=dict(backend='disk'),
                imdecode_backend='pillow',
                with_calib=False,
                with_calib_kittiraw=False,
                with_calib_kittiodometry=False,
                with_calib_kittiobject=False,
                with_depth=False,
                depth_dir=None,
                depth_scale=256.0,
                depth_suffix='.png'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.with_calib = with_calib
        self.with_calib_kittiraw = with_calib_kittiraw
        self.with_calib_kittiodometry = with_calib_kittiodometry
        self.with_calib_kittiobject = with_calib_kittiobject
        self.with_depth = with_depth
        self.depth_dir = depth_dir
        self.depth_scale = depth_scale
        self.depth_suffix = depth_suffix
        # enter your path of calibration matrix of each dataset
        # nuScenes数据集路径统一配置为: /media/ldk950413/data0/nuScenes（注意大小写）
        if self.with_calib:
            # 尝试两种大小写变体
            nuscenes_calib_lower = '/media/ldk950413/data0/nuscenes/calib.json'
            nuscenes_calib_upper = '/media/ldk950413/data0/nuScenes/calib.json'
            
            # 尝试加载预生成的calib.json（可选）
            self.nuscenes_calib = None
            if osp.exists(nuscenes_calib_upper):
                try:
                    self.nuscenes_calib = json.load(open(nuscenes_calib_upper, 'r'))
                except:
                    pass
            elif osp.exists(nuscenes_calib_lower):
                try:
                    self.nuscenes_calib = json.load(open(nuscenes_calib_lower, 'r'))
                except:
                    pass
            
            # 如果没有预生成的calib.json，或者需要动态加载，则初始化nuScenes数据集
            # 这将用于动态加载标定信息
            try:
                from nuscenes import NuScenes
                nuscenes_dataroot = '/media/ldk950413/data0/nuScenes'
                if not osp.exists(nuscenes_dataroot):
                    nuscenes_dataroot = '/media/ldk950413/data0/nuscenes'
                self.nuscenes_obj = NuScenes(version='v1.0-trainval', dataroot=nuscenes_dataroot, verbose=False)
            except:
                self.nuscenes_obj = None
        if self.with_calib_kittiraw:
            self.kittiraw = json.load(open('YOUR_PATH/kitti_raw/calib.json','r'))
        if self.with_calib_kittiodometry:
            self.kittiodometry = json.load(open('YOUR_PATH/kitti_odometry/calib.json','r'))
        if self.with_calib_kittiobject:
            self.kittiobject = json.load(open('YOUR_PATH/kitti_object/calib.json','r'))
    
    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        if self.imdecode_backend=='pyramid':
            encoded_labels = torchvision.transforms.functional.to_tensor(Image.open(filename)).long()
            # decode to binary labels,the data type of gt_semantic_seg is bool,i.e. 0 or 1, gt_semantic_seg is numpy array
            if self.with_calib:
                gt_semantic_seg = decode_binary_labels(encoded_labels,15).numpy()
            if self.with_calib_kittiraw or self.with_calib_kittiodometry or self.with_calib_kittiobject:
                # only one class for kitti dataset
                gt_semantic_seg = np.zeros((2,196,200), dtype=bool)
                gt_semantic_seg[0,...] = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
                gt_semantic_seg[0,...] = cv2.flip(gt_semantic_seg[0,...].astype(np.uint8),0).astype(bool)
                gt_semantic_seg[-1,...] = cv2.imread("./mask_vis.png",cv2.IMREAD_GRAYSCALE).astype(bool)
                gt_semantic_seg[-1,...] = np.invert(gt_semantic_seg[-1,...])
            gt_semantic_seg[-1,...] = np.invert(gt_semantic_seg[-1,...])
        else:
            img_bytes = self.file_client.get(filename)
            gt_semantic_seg = mmcv.imfrombytes(
                img_bytes, flag='unchanged',
                backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')

        # Optional depth supervision
        if self.with_depth:
            # Resolve depth directory
            seg_prefix = results.get('seg_prefix', None)
            depth_dir = self.depth_dir
            if depth_dir is None and seg_prefix is not None:
                # ann_bev_dir/train -> ann_bev_dir/train_depth
                depth_dir = osp.join(osp.dirname(seg_prefix),
                                     osp.basename(seg_prefix) + '_depth')
            elif depth_dir is not None and seg_prefix is not None and not osp.isabs(depth_dir):
                # If depth_dir starts with ann_bev_dir, use data_root
                if depth_dir.startswith('ann_bev_dir'):
                    data_root = osp.dirname(osp.dirname(seg_prefix))
                    depth_dir = osp.join(data_root, depth_dir)
                else:
                    depth_dir = osp.join(osp.dirname(seg_prefix), depth_dir)

            if depth_dir is not None:
                base_name = osp.splitext(osp.basename(filename))[0]
                depth_file = osp.join(depth_dir, base_name + self.depth_suffix)
                if osp.exists(depth_file):
                    depth = None
                    depth_mask = None
                    if depth_file.endswith('.npz'):
                        npz = np.load(depth_file)
                        depth = npz.get('depth', None)
                        depth_mask = npz.get('mask', None)
                    elif depth_file.endswith('.npy'):
                        depth = np.load(depth_file)
                    else:
                        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)

                    if depth is not None:
                        depth = depth.astype(np.float32)
                        # Normalize if depth values are scaled
                        if self.depth_scale is not None and depth.max() > 1.0:
                            depth = depth / float(self.depth_scale)
                        results['gt_depth'] = depth
                        results['seg_fields'].append('gt_depth')
                    if depth_mask is not None:
                        depth_mask = depth_mask.astype(np.uint8)
                        results['gt_depth_mask'] = depth_mask
                        results['seg_fields'].append('gt_depth_mask')
        if self.with_calib:
            token = osp.basename(filename).split('.')[0]
            
            # 首先尝试从预生成的calib.json中加载
            intrinsics = None
            if self.nuscenes_calib and token in self.nuscenes_calib:
                intrinsics = torch.tensor(self.nuscenes_calib[token])
            # 如果找不到，则从nuScenes数据集中动态加载
            elif hasattr(self, 'nuscenes_obj') and self.nuscenes_obj is not None:
                try:
                    sample_data = self.nuscenes_obj.get('sample_data', token)
                    sensor = self.nuscenes_obj.get(
                        'calibrated_sensor', sample_data['calibrated_sensor_token'])
                    intrinsics = torch.tensor(sensor['camera_intrinsic'], dtype=torch.float32)
                    # 根据图像尺寸缩放标定矩阵
                    intrinsics[0] *= results['img_shape'][1] / sample_data['width']
                    intrinsics[1] *= results['img_shape'][0] / sample_data['height']
                except Exception as e:
                    raise KeyError(f"无法从nuScenes数据集中加载token '{token}'的标定信息: {e}")
            else:
                raise KeyError(
                    f"无法找到token '{token}'的标定信息。"
                    f"请确保calib.json存在，或者nuScenes数据集已正确安装。"
                )
            
            # 根据目标分辨率缩放标定矩阵（800x600）
            intrinsics[0] *= 800 / results['img_shape'][1]
            intrinsics[1] *= 600 / results['img_shape'][0]
            results['calib'] = intrinsics
        if self.with_calib_kittiraw:
            token = osp.basename(filename).split('.')[0]
            intrinsics = torch.tensor(self.kittiraw[token])
            intrinsics[0] *= 1024 / results['img_shape'][1]
            intrinsics[1] *= 1024 /results['img_shape'][0]
            results['calib'] = intrinsics
        if self.with_calib_kittiodometry:
            token = osp.basename(filename).split('.')[0]
            intrinsics = torch.tensor(self.kittiodometry[token])
            intrinsics[0] *= 1024 / results['img_shape'][1]
            intrinsics[1] *= 1024 /results['img_shape'][0]
            results['calib'] = intrinsics
        if self.with_calib_kittiobject:
            token = osp.basename(filename).split('.')[0]
            intrinsics = torch.tensor(self.kittiobject[token])
            intrinsics[0] *= 1024 / results['img_shape'][1]
            intrinsics[1] *= 1024 /results['img_shape'][0]
            results['calib'] = intrinsics
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
