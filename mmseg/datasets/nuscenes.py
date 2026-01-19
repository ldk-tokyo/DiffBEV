"""
nuScenes 数据集实现模块

本模块实现了用于 BEV 语义分割的 nuScenes 数据集类。
nuScenes 是一个大规模自动驾驶数据集，包含多传感器数据（相机、LiDAR等）。
"""
import os.path as osp
import os

from mmengine import Config as mmcv_Config
import mmcv
import torch
import json
import numpy as np
from PIL import Image

from mmengine.logging import print_log
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .custom import CustomDataset
from tqdm import trange

def covert_color(input):
    """
    将十六进制颜色字符串转换为RGB元组
    
    Args:
        input (str): 十六进制颜色字符串，格式如 '#a6cee3'
        
    Returns:
        tuple: RGB颜色值 (r, g, b)
    """
    str1 = input[1:3]
    str2 = input[3:5]
    str3 = input[5:7]
    r = int('0x' + str1, 16)
    g = int('0x' + str2, 16)
    b = int('0x' + str3, 16)
    return (r, g, b)

# visualize_map_mask中的color_map对应的RGB为
# RGB_value = [(166, 206, 227),(31, 120, 180),(178, 223, 138),(51, 160, 44),(251, 154, 153),
#              (227, 26, 28),(253, 191, 111),(255, 127, 0),(202, 178, 214),(106, 61, 154),
#              (126, 119, 46),(0, 255, 0),(0, 0, 255),(0, 255, 255),(48, 48, 48)]
def visualize_map_mask(map_mask):
    """
    将BEV语义分割mask转换为可视化图像
    
    Args:
        map_mask (np.ndarray): BEV语义分割mask，shape为 (num_classes, H, W)
        
    Returns:
        np.ndarray: 可视化的RGB图像，shape为 (H, W, 3)
    """
    color_map = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99',
                 '#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a',
                 '#7e772e','#00ff00','#0000ff','#00ffff','#303030']
    ori_shape = map_mask.shape
    vis = np.zeros((ori_shape[1], ori_shape[2], 3),dtype=np.uint8)
    vis = vis.reshape(-1,3)
    map_mask = map_mask.reshape(ori_shape[0],-1)
    for layer_id in range(map_mask.shape[0]):
        keep = np.where(map_mask[layer_id,:])[0]
        for i in range(3):
            vis[keep, 2-i] = covert_color(color_map[layer_id])[i]
    return vis.reshape(ori_shape[1], ori_shape[2], 3)


@DATASETS.register_module()
class NuscenesDataset(CustomDataset):
    """
    nuScenes BEV语义分割数据集类
    
    该类继承自CustomDataset，实现了nuScenes数据集的加载、预处理和评估功能。
    nuScenes数据集包含14个语义类别，用于BEV（鸟瞰图）语义分割任务。
    """
    # 14个语义类别：可行驶区域、人行横道、人行道、停车场、车辆类、行人等
    CLASSES = ('drivable_area', 'ped_crossing', 'walkway', 'carpark',
               'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
               'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier')

    # 每个类别对应的RGB颜色（用于可视化）
    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51]]

    def __init__(self, **kwargs):
        super(NuscenesDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)

    def results2img(self, results, imgfile_prefix, to_label_id):
        """
        将预测结果转换为图像文件并保存
        
        Args:
            results (list): 预测结果列表
            imgfile_prefix (str): 保存图像的目录前缀
            to_label_id (bool): 是否转换为标签ID
            
        Returns:
            list: 保存的图像文件路径列表
        """
        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            result = result + 1

            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

            prog_bar.update()

        return result_files

    def prepare_test_img(self, idx):
        """
        准备测试图像数据
        
        Args:
            idx (int): 数据索引
            
        Returns:
            dict: 处理后的测试数据字典
        """
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration for nuScenes dataset.
        
        Args:
            preds (list[np.ndarray] | np.ndarray): 预测的分割结果，每个元素是 (H, W) 的numpy数组
                或文件路径（当efficient_test=True时）
            indices (list[int] | int): 对应的ground truth索引
            
        Returns:
            list[tuple[torch.Tensor]]: 每个元素是 (tp, fp, fn, valid) 四个tensor
                - tp: True Positive，形状为 (num_classes,)
                - fp: False Positive，形状为 (num_classes,)
                - fn: False Negative，形状为 (num_classes,)
                - valid: 有效类别标志，形状为 (num_classes,)
        """
        # 导入iou函数
        from mmseg.models.losses.iou import iou
        
        # 兼容batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]
        
        pre_eval_results = []
        
        for pred, index in zip(preds, indices):
            # 如果efficient_test=True，pred可能是文件路径
            if isinstance(pred, str):
                pred = np.load(pred)
            
            # 获取ground truth
            gt_seg_map = self.get_gt_seg_map_by_idx(index)
            
            # 将预测转换为tensor（如果还不是）
            if isinstance(pred, np.ndarray):
                # 如果是分割结果 (H, W)，需要转换为二进制格式 (num_classes, H, W)
                # 首先需要argmax得到类别ID
                if pred.ndim == 2:
                    # (H, W) -> (1, num_classes, H, W) 二进制格式
                    num_classes = len(self.CLASSES)
                    h, w = pred.shape
                    pred_binary = np.zeros((1, num_classes, h, w), dtype=bool)
                    # 将类别ID转换为二进制mask
                    for c in range(num_classes):
                        pred_binary[0, c, ...] = (pred == c)
                    pred = torch.from_numpy(pred_binary).bool()
                else:
                    pred = torch.from_numpy(pred).bool()
            else:
                pred = pred
            
            # 将GT转换为tensor
            if isinstance(gt_seg_map, np.ndarray):
                # nuScenes的GT已经是二进制格式 (num_classes, H, W) 或 (1, num_classes, H, W)
                if gt_seg_map.ndim == 3:
                    # (num_classes, H, W) -> (1, num_classes, H, W)
                    gt_seg_map = gt_seg_map[None, ...]
                elif gt_seg_map.ndim == 2:
                    # 如果是2D，说明不是二进制格式，需要转换
                    num_classes = len(self.CLASSES)
                    h, w = gt_seg_map.shape
                    gt_binary = np.zeros((1, num_classes, h, w), dtype=bool)
                    for c in range(num_classes):
                        gt_binary[0, c, ...] = (gt_seg_map == c)
                    gt_seg_map = gt_binary
                gt_seg_map = torch.from_numpy(gt_seg_map).bool()
            else:
                gt_seg_map = gt_seg_map
                # 确保GT是4D格式
                if gt_seg_map.ndim == 3:
                    gt_seg_map = gt_seg_map.unsqueeze(0)
            
            # 确保形状匹配 (batch_size, num_classes, H, W)
            if pred.shape != gt_seg_map.shape:
                # 如果形状不匹配，尝试调整
                if pred.shape[1:] != gt_seg_map.shape[1:]:
                    # 需要resize（空间尺寸不匹配）
                    import torch.nn.functional as F
                    pred = F.interpolate(
                        pred.float(), 
                        size=gt_seg_map.shape[2:], 
                        mode='nearest'
                    ).bool()
                elif pred.shape[0] != gt_seg_map.shape[0]:
                    # batch维度不匹配
                    if pred.shape[0] == 1:
                        # 如果pred只有1个样本，确保GT也是
                        if gt_seg_map.shape[0] > 1:
                            gt_seg_map = gt_seg_map[:1]
                    elif gt_seg_map.shape[0] == 1:
                        # 如果GT只有1个样本，确保pred也是
                        if pred.shape[0] > 1:
                            pred = pred[:1]
            
            # 计算tp, fp, fn, valid
            tp, fp, fn, valid = iou(pred, gt_seg_map, per_class=True)
            
            pre_eval_results.append((tp, fp, fn, valid))
        
        return pre_eval_results

    def format_results(self, results, imgfile_prefix=None, to_label_id=True):
        """
        格式化预测结果并生成可视化图像
        
        该函数将预测结果、真实标签和原始图像拼接在一起，生成对比可视化图像。
        
        Args:
            results (list): 测试结果列表，每个元素包含 (pred, gt, img_path)
                - pred: 预测的BEV语义分割结果，形状为 (1, num_classes, H, W)
                - gt: 真实标签，形状为 (1, num_classes+1, H, W)
                - img_path: 原始图像路径
            imgfile_prefix (str): 保存可视化图像的目录前缀
            to_label_id (bool): 是否转换为标签ID（未使用，保持兼容性）
        """
        assert isinstance(results, list), 'results must be a list'

        imgfile_prefix = osp.join(imgfile_prefix, 'vis')
        if not osp.exists(imgfile_prefix):
            os.makedirs(imgfile_prefix)
        print_log('\n Start formatting the result')

        for id in trange(len(results)):
            pred, gt, img_path = results[id]
            b,c,h,w = pred.shape
            assert pred.shape[0]==1 and gt.shape[0]==1
            pred = pred[0]
            gt = gt[0]
            gt[-1, ...] = np.invert(gt[-1, ...])
            pred = np.concatenate([pred, gt[-1,...][None,...]], axis=0)
            pred_vis = visualize_map_mask(pred)
            gt_vis = visualize_map_mask(gt)
            img = mmcv.imread(img_path, backend='cv2')
            img = mmcv.imresize(img,(int(float(img.shape[1])*h/float(img.shape[0])), h))
            vis = np.concatenate([img, pred_vis[::-1,...], gt_vis[::-1,]], axis=1)
            save_path = osp.join(imgfile_prefix, os.path.basename(img_path))
            mmcv.imwrite(vis, save_path)

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """
        评估模型在nuScenes数据集上的性能
        
        Args:
            results (list): 测试结果列表，每个元素包含 (tp, fp, fn, valid) 四个tensor
                - tp: True Positive，形状为 (num_classes,)
                - fp: False Positive，形状为 (num_classes,)
                - fn: False Negative，形状为 (num_classes,)
                - valid: 有效类别标志，形状为 (num_classes,)
            metric (str | list[str]): 评估指标，可选 'mIoU', 'mIoUv1', 'mIoUv2'
            logger (logging.Logger): 日志记录器
            efficient_test (bool): 是否使用高效测试模式
            **kwargs: 其他参数
            
        Returns:
            dict: 评估结果字典，包含各类别的IoU和平均mIoU
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mIoUv1', 'mIoUv2']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        # 将所有样本的 tp, fp, fn, valid 拼接成tensor
        # 形状: (N, C)，N为样本数，C为类别数
        tp = torch.cat([res[0][None, ...] for res in results], dim=0) #N*C
        fp = torch.cat([res[1][None, ...] for res in results], dim=0) #N*C
        fn = torch.cat([res[2][None, ...] for res in results], dim=0) #N*C
        valids = torch.cat([res[3][None,...] for res in results],dim=0) #N*C
        
        # 初始化评估结果字典
        eval_results = {}
        
        for met in metric:
            if met=='mIoU':
                ious = tp.sum(0).float()/(tp.sum(0)+fp.sum(0)+fn.sum(0)).float()
                print_log('\nper class results (iou):', logger)
                for cid in range(len(self.CLASSES)):
                    print_log('%.04f:%s tp:%d fp:%d fn:%d' % (ious[cid], self.CLASSES[cid], tp.sum(0)[cid],fp.sum(0)[cid],fn.sum(0)[cid]), logger)
                miou_value = ious.mean().item()
                print_log('%s: %.04f' % (met, miou_value), logger)
                eval_results['mIoU'] = miou_value
            elif met == 'mIoUv1':
                ious = tp.float() / (tp + fp + fn).float()
                print_log('\nper class results (iou):', logger)
                miou, valid_class = 0, 0
                for cid in range(len(self.CLASSES)):
                    iou_c = ious[:, cid][valids[:, cid]]
                    if iou_c.shape[0] > 0:
                        iou_c = iou_c.mean()
                        miou += iou_c
                        valid_class += 1
                    else:
                        iou_c = -1
                    print_log('%.04f:%s' % (iou_c, self.CLASSES[cid]), logger)
                miou_value = miou / valid_class if valid_class > 0 else 0.0
                print_log('%s: %.04f' % (met, miou_value), logger)
                eval_results['mIoUv1'] = miou_value
            elif met == 'mIoUv2':
                ious = tp.sum(-1).float() / (tp.sum(-1) + fp.sum(-1) + fn.sum(-1)).float()
                miou_value = ious.mean().item()
                print_log('\n%s: %.04f' % (met, miou_value), logger)
                eval_results['mIoUv2'] = miou_value
            else:
                assert False, 'nuknown metric type %s'%metric
        
        return eval_results