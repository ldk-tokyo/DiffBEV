# Copyright (c) OpenMMLab. All rights reserved.
"""
BEV Segmentor for BEV semantic segmentation tasks
支持使用transformer（如LSS）进行视图变换，需要calib信息
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class BEVSegmentor(EncoderDecoder):
    """
    BEV Segmentor for Bird's Eye View semantic segmentation
    
    该segmentor继承自EncoderDecoder，但支持使用transformer参数（作为neck）
    并且能够从img_metas中提取calib信息传递给transformer
    """
    
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 transformer=None,  # 支持transformer参数（作为neck的别名）
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        # 如果提供了transformer，将其作为neck使用
        if transformer is not None and neck is None:
            neck = transformer
        
        super(BEVSegmentor, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
    
    def extract_feat(self, img, img_metas=None):
        """
        提取特征，支持从img_metas中提取calib信息传递给transformer
        
        Args:
            img: 输入图像
            img_metas: 图像元信息列表，可能包含calib信息
            
        Returns:
            提取的特征
        """
        x = self.backbone(img)
        
        if self.with_neck:
            # 检查neck是否需要calib信息（如LSS transformer需要intrinstics参数）
            # 通过检查forward方法的签名来判断
            import inspect
            needs_calib = False
            if hasattr(self.neck, 'forward'):
                try:
                    sig = inspect.signature(self.neck.forward)
                    params = list(sig.parameters.keys())
                    # 检查是否有intrinstics或calib参数（LSS transformer使用intrinstics）
                    if len(params) > 1 and ('intrinstics' in params or 'calib' in params):
                        needs_calib = True
                except:
                    # 如果无法检查签名，尝试通过类型判断
                    neck_type = type(self.neck).__name__
                    if 'LiftSplatShoot' in neck_type or 'Transformer' in neck_type:
                        needs_calib = True
            
            if needs_calib:
                # 从img_metas中提取calib信息
                if img_metas is not None and len(img_metas) > 0:
                    # 提取calib信息并转换为tensor
                    calibs = []
                    # 获取设备（确保x不为None且有效）
                    if x is not None:
                        if isinstance(x, (list, tuple)):
                            # 如果x是list/tuple，找到第一个非None元素
                            for elem in x:
                                if elem is not None:
                                    device = elem.device
                                    break
                            else:
                                # 如果所有元素都是None，使用cuda:0
                                device = torch.device('cuda:0')
                        else:
                            device = x.device
                    else:
                        # 如果x为None，默认使用cuda:0
                        device = torch.device('cuda:0')
                    for img_meta in img_metas:
                        if 'calib' in img_meta:
                            calib = img_meta['calib']
                            if isinstance(calib, torch.Tensor):
                                # 确保tensor在正确的设备上
                                calibs.append(calib.to(device))
                            else:
                                calibs.append(torch.tensor(calib, device=device, dtype=torch.float32))
                    
                    if len(calibs) > 0:
                        # 堆叠calibs
                        if len(calibs) == 1:
                            calib_tensor = calibs[0]
                            if calib_tensor.dim() == 2:
                                calib_tensor = calib_tensor.unsqueeze(0)
                        else:
                            calib_tensor = torch.stack(calibs)
                        
                        # 调用transformer，传递calib（LSS使用intrinstics参数名）
                        x = self.neck(x, calib_tensor)
                    else:
                        # 如果没有calib，创建默认calib
                        device = x[0].device if isinstance(x, (list, tuple)) else x.device
                        batch_size = x[0].shape[0] if isinstance(x, (list, tuple)) else x.shape[0]
                        # 创建默认的3x3单位矩阵作为calib
                        default_calib = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
                        x = self.neck(x, default_calib)
                else:
                    # 如果没有img_metas，创建默认calib
                    device = x[0].device if isinstance(x, (list, tuple)) else x.device
                    batch_size = x[0].shape[0] if isinstance(x, (list, tuple)) else x.shape[0]
                    default_calib = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
                    x = self.neck(x, default_calib)
            else:
                # 普通的neck，不需要calib
                x = self.neck(x)
        
        return x
    
    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training."""
        x = self.extract_feat(img, img_metas)
        
        losses = dict()
        loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        losses.update(loss_decode)
        
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        
        return losses
    
    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into semantic segmentation map."""
        x = self.extract_feat(img, img_metas)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out
