"""
PYVA损失函数模块

本模块实现了PYVA方法中使用的损失函数，包括：
1. balanced_binary_cross_entropy: 平衡二元交叉熵损失
2. compute_losses: PYVA的总损失计算（包括topview损失、变换损失等）
3. iou: IoU计算函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# loss['bce'] = balanced_binary_cross_entropy(seg_logit,seg_label[:,:-1,...],seg_label[:,-1,...])
def balanced_binary_cross_entropy(logits, labels, mask, weights):
    """
    平衡二元交叉熵损失
    
    该函数根据类别权重计算平衡的交叉熵损失，用于处理类别不平衡问题。
    
    Args:
        logits: 模型输出的logits，shape为 (bs, num_classes, H, W)
        labels: 真实标签，shape为 (bs, num_classes, H, W)
        mask: 有效区域mask，shape为 (bs, H, W)
        weights: 每个类别的权重，shape为 (num_classes,)
        
    Returns:
        Tensor: 平衡的二元交叉熵损失值
    """
    weights = (logits.new(weights).view(-1, 1, 1) - 1) * labels.float() + 1.
    weights = weights * mask.unsqueeze(1).float()
    return F.binary_cross_entropy_with_logits(logits, labels.float(), weights)

class compute_losses(nn.Module):
    """
    PYVA损失计算模块
    
    该模块计算PYVA方法的总损失，包括：
    1. topview_loss: BEV分割的交叉熵损失
    2. transform_topview_loss: 变换后BEV分割的交叉熵损失
    3. transform_loss: 视图变换的循环一致性损失（L1损失）
    """
    def __init__(self):
        super(compute_losses, self).__init__()
        # L1损失用于计算视图变换的循环一致性
        self.compute_transform_losses = nn.L1Loss()
        # 两类输出的类别权重（初始化为1）
        self.class_weights_1 = torch.ones(14,)
        self.class_weights_2 = torch.ones(14,)
    # opt.type can only be chosen from ["both","static","dynamic"]
    def forward(self, outputs,  gt_semantic_seg, features, retransform_features):
        """
        前向传播计算损失
        
        Args:
            outputs: 模型输出字典，包含：
                - "topview": BEV分割logits，shape为 (bs, num_classes, H, W)
                - "transform_topview": 变换后的BEV分割logits
            gt_semantic_seg: 真实标签，shape为 (bs, 1, num_classes+1, H, W)
            features: 原始特征，用于计算循环一致性损失
            retransform_features: 重变换后的特征，用于计算循环一致性损失
            
        Returns:
            dict: 包含总损失和IoU准确率的字典
        """
        losses = {}
        losses_topview_loss = 0.
        losses_transform_topview_loss = 0.
        losses_transform_loss = 0.
        # logits:outputs,labels:gt_semantic_seg
        gt_semantic_seg = gt_semantic_seg.squeeze(1).bool()  # (bs, num_classes+1, H, W)
        
        # 将权重移动到正确的设备
        self.class_weights_1 = self.class_weights_1.to(outputs["topview"])
        self.class_weights_2 = self.class_weights_2.to(outputs["transform_topview"])
        # 计算BEV分割损失（主要损失）
        losses_topview_loss = balanced_binary_cross_entropy(outputs["topview"],gt_semantic_seg[:,:-1,...],gt_semantic_seg[:,-1,...],self.class_weights_1)
        # 计算变换后BEV分割损失（辅助损失）
        losses_transform_topview_loss = balanced_binary_cross_entropy(
            outputs["transform_topview"],gt_semantic_seg[:,:-1,...],gt_semantic_seg[:,-1,...],self.class_weights_2)
        # 计算视图变换的循环一致性损失（L1损失）
        losses_transform_loss = self.compute_transform_losses(
            features,retransform_features)

        # losses_topview_loss = loss_BCE,losses_transform_loss = loss_cycle,losses_transform_topview_loss = loss_discriminator
        # 总损失 = topview损失 + 0.001 * 循环一致性损失 + 1 * 变换topview损失
        losses["loss"] = losses_topview_loss+ 0.001 * losses_transform_loss + 1* losses_transform_topview_loss

        # acc_seg, only to evaluate, so we use detach here
        # 计算IoU准确率（用于评估，不需要梯度）
        losses['acc_seg'] = iou(outputs["topview"].detach().sigmoid()>0.5, gt_semantic_seg[:,:-1,...], gt_semantic_seg[:,-1,...])
        # return losses['loss']
        return losses

def iou(preds, labels, mask=None, per_class=False):
    """
    计算IoU（Intersection over Union）
    
    Args:
        preds: 预测的二值mask，shape为 (bs, num_classes, H, W)
        labels: 真实标签，shape为 (bs, num_classes, H, W)
        mask: 有效区域mask，shape为 (bs, H, W)，可选
        per_class: 如果为True，返回每个类别的tp, fp, fn, valid；否则返回平均IoU
        
    Returns:
        如果per_class=False: 返回平均IoU值（标量）
        如果per_class=True: 返回 (tp, fp, fn, valid) 四个tensor
    """
    num_class = preds.shape[1]
    # 将特征图展平并重新组织为 (num_class, batch*H*W)
    preds = preds.flatten(2, -1).permute(1, 0, 2).reshape(num_class, -1)
    labels = labels.flatten(2, -1).permute(1, 0, 2).reshape(num_class, -1)
    # 如果提供了mask，只计算有效区域的IoU
    if mask is not None:
        preds = preds[:, mask.flatten()]
        labels = labels[:, mask.flatten()]
    # 计算True Positive, False Positive, False Negative
    true_pos = preds & labels   # 预测为正且真实为正
    false_pos = preds & ~labels # 预测为正但真实为负
    false_neg = ~preds & labels # 预测为负但真实为正
    tp = true_pos.long()
    fp = false_pos.long()
    fn = false_neg.long()
    # 根据per_class参数返回不同的结果
    if not per_class:
        # 返回所有类别的平均IoU
        return tp.sum().float() / (tp.sum() + fn.sum() + fp.sum()).float()
    else:
        # 返回每个类别的tp, fp, fn和valid标志
        valid = labels.int().sum(-1)>0  # 标记哪些类别有标签
        return tp.sum(-1), fp.sum(-1), fn.sum(-1), valid

class simple_loss():
    def __init__(self,priors=[0.002,0.008]):
        super(simple_loss, self).__init__()
        self.compute_transform_losses = nn.L1Loss()
        self.priors = torch.tensor(priors)
        self.class_weights = torch.sqrt(1 / self.priors)
    def forward(self, outputs,  gt_semantic_seg, features, retransform_features):
        losses = {}
        losses_topview_loss = 0.
        losses_transform_loss = 0.
        # logits:outputs,labels:gt_semantic_seg
        gt_semantic_seg = gt_semantic_seg.squeeze(1).bool()
        self.class_weights = self.class_weights.to(outputs)
        losses_topview_loss = balanced_binary_cross_entropy(outputs,gt_semantic_seg[:,:-1,...],gt_semantic_seg[:,-1,...],self.class_weights)
        losses_transform_loss = self.compute_transform_losses(features,retransform_features)
        losses["loss"] = losses_topview_loss + 0.001 * losses_transform_loss
        # acc_seg, only to evaluate, so we use detach here
        losses['acc_seg'] = iou(outputs.detach().sigmoid()>0.5, gt_semantic_seg[:,:-1,...], gt_semantic_seg[:,-1,...])
        return losses