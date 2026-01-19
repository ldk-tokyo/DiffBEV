"""
Pyramid Head 解码头实现

本模块实现了用于BEV语义分割的Pyramid解码头。
该解码头使用TopdownNetwork进行特征金字塔处理，最终输出每个像素的类别预测。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABCMeta
from mmengine.model import BaseModule
# 尝试导入force_fp32（兼容MMCV 1.x和2.x）
try:
    from mmcv.runner import force_fp32
except ImportError:
    try:
        from mmengine.model import force_fp32
    except ImportError:
        # 如果都失败，创建一个兼容装饰器
        def force_fp32(apply_to=()):
            """强制使用FP32的装饰器（简化实现）"""
            def decorator(func):
                return func  # 简化实现，不进行实际转换
            return decorator

from ..builder import HEADS
from ..losses import iou

def prior_uncertainty_loss(x, mask, priors):
    """
    先验不确定性损失
    
    该损失函数鼓励模型在不存在的区域（mask为False）保持先验概率分布。
    
    Args:
        x: 模型输出的logits，shape为 (bs, num_classes, H, W)
        mask: 有效区域mask，True表示该位置有标签，shape为 (bs, H, W)
        priors: 每个类别的先验概率，shape为 (num_classes,)
        
    Returns:
        Tensor: 先验不确定性损失值
    """
    # priors shape: [14]-->[1,14,1,1]-->[bs,14,196,200]
    priors = x.new(priors).view(1, -1, 1, 1).expand_as(x)
    # F.binary_cross_entropy_with_logits(x, priors, reduce=False) return a tensor with the shape of x, i.e. [bs,14,196,200]
    xent = F.binary_cross_entropy_with_logits(x, priors, reduce=False)
    return (xent * (~mask).float().unsqueeze(1)).mean()

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

class OccupancyCriterion(nn.Module):
    """
    占用率损失准则
    
    该损失函数结合了平衡二元交叉熵损失和先验不确定性损失，
    用于训练BEV语义分割模型。主要处理类别不平衡问题。
    """

    def __init__(self, priors=[0.44679, 0.02407, 0.14491, 0.02994, 0.02086, 0.00477, 0.00156, 0.00189, 0.00084, 0.00119, 0.00019, 0.00012, 0.00031, 0.00176],
                xent_weight=1., uncert_weight=0.001,
                weight_mode='sqrt_inverse'):
        super().__init__()

        self.xent_weight = xent_weight
        self.uncert_weight = uncert_weight

        self.priors = torch.tensor(priors)

        if weight_mode == 'inverse':
            self.class_weights = 1 / self.priors
        elif weight_mode == 'sqrt_inverse':
            self.class_weights = torch.sqrt(1 / self.priors)
        elif weight_mode == 'equal':
            self.class_weights = torch.ones_like(self.priors)
        else:
            raise ValueError('Unknown weight mode option: ' + weight_mode)

    def forward(self, logits, labels, mask, *args):
        self.class_weights = self.class_weights.to(logits)
        bce_loss = balanced_binary_cross_entropy(
            logits, labels, mask, self.class_weights)
        self.priors = self.priors.to(logits)
        uncert_loss = prior_uncertainty_loss(logits, mask, self.priors)

        return bce_loss * self.xent_weight + uncert_loss * self.uncert_weight


class LinearClassifier(nn.Conv2d):

    def __init__(self, in_channels, num_classes):
        super().__init__(in_channels, num_classes, 1)

    def initialise(self, prior):
        prior = torch.tensor(prior)
        self.weight.data.zero_()
        self.bias.data.copy_(torch.log(prior / (1 - prior)))


class TopdownNetwork(nn.Sequential):

    def __init__(self, in_channels, channels, layers=[6, 1, 1],
                strides=[1, 2, 2], blocktype='basic'):
        modules = list()
        self.downsample = 1
        for nblocks, stride in zip(layers, strides):
            # Add a new residual layer
            module = ResNetLayer(
                in_channels, channels, nblocks, 1 / stride, blocktype=blocktype)
            modules.append(module)

            # Halve the number of channels at each layer
            in_channels = module.out_channels
            channels = channels // 2
            self.downsample *= stride

        self.out_channels = in_channels

        super().__init__(*modules)


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    # Fractional strides correspond to transpose convolution
    if stride < 1:
        stride = int(round(1 / stride))
        kernel_size = stride + 2
        padding = int((dilation * (kernel_size - 1) - stride + 1) / 2)
        return nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size, stride, padding,
            output_padding=0, dilation=dilation, bias=False)
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=int(stride),
                    dilation=dilation, padding=dilation, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    # Fractional strides correspond to transpose convolution
    if int(1 / stride) > 1:
        stride = int(1 / stride)
        return nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=stride, stride=stride, bias=False)

    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=int(stride), bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.GroupNorm(16, planes)

        self.conv2 = conv3x3(planes, planes, 1, dilation)
        self.bn2 = nn.GroupNorm(16, planes)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride), nn.GroupNorm(16, planes))
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out, inplace=True)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.GroupNorm(16, planes)
        self.conv2 = conv3x3(planes, planes, stride, dilation)
        self.bn2 = nn.GroupNorm(16, planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.GroupNorm(16, planes * self.expansion)

        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion, stride),
                nn.GroupNorm(16, planes * self.expansion))
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out

class ResNetLayer(nn.Sequential):

    def __init__(self, in_channels, channels, num_blocks, stride=1,
                dilation=1, blocktype='bottleneck'):

        # Get block type
        if blocktype == 'basic':
            block = BasicBlock
        elif blocktype == 'bottleneck':
            block = Bottleneck
        else:
            raise Exception("Unknown residual block type: " + str(blocktype))

        # Construct layers
        layers = [block(in_channels, channels, stride, dilation)]
        for _ in range(1, num_blocks):
            layers.append(block(channels * block.expansion, channels, 1, dilation))

        self.in_channels = in_channels
        self.out_channels = channels * block.expansion

        super(ResNetLayer, self).__init__(*layers)


@HEADS.register_module()
class PyramidHead(BaseModule, metaclass=ABCMeta):
    """
    Pyramid解码头
    
    该解码头用于BEV语义分割任务，主要包括：
    1. TopdownNetwork: 特征金字塔网络，用于多尺度特征处理
    2. LinearClassifier: 线性分类器，输出每个像素的类别预测
    
    Args:
        num_classes (int): 分割类别数
        align_corners (bool): 是否对齐角点（用于插值）
        in_channels (int): 输入特征通道数
        channels (int): TopdownNetwork中间通道数
        layers (list): TopdownNetwork每层的block数量
        strides (list): TopdownNetwork每层的stride
    """
    def __init__(self,num_classes,align_corners=True,in_channels=64, channels=128, layers=[4,4], strides=[1,2], **kwargs):
        super(PyramidHead, self).__init__(**kwargs)
        # 特征金字塔网络，使用bottleneck结构
        self.topdown = TopdownNetwork(blocktype='bottleneck',in_channels=in_channels, channels=channels, layers=layers, strides=strides)
        self.num_classes = num_classes
        self.align_corners = align_corners
        # 线性分类器，将特征映射到类别数
        self.classifier = LinearClassifier(self.topdown.out_channels, self.num_classes)

        # 使用nuScenes数据集的类别先验概率初始化分类器bias
        # 这些先验值反映了各类别在数据集中的出现频率
        self.classifier.initialise([0.44679, 0.02407, 0.14491, 0.02994, 0.02086, 0.00477, 0.00156, 0.00189,
                                    0.00084, 0.00119, 0.00019, 0.00012, 0.00031, 0.00176])
        # 占用率损失准则
        self.criterion = OccupancyCriterion()
        
    def forward(self, inputs):
        td_feats = self.topdown(inputs)
        logits = self.classifier(td_feats)
        return logits

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        return self.forward(inputs)

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """
        计算分割损失
        
        Args:
            seg_logit: 模型输出的logits，shape为 (bs, num_classes, H, W)
            seg_label: 真实标签，shape为 (bs, 1, num_classes+1, H, W)
                      - 前num_classes个通道是各类别的二值标签
                      - 最后一个通道是有效区域mask
            
        Returns:
            dict: 包含loss和accuracy的字典
                - loss_seg: 分割损失
                - acc_seg: IoU准确率
        """
        seg_label = seg_label.squeeze(1).bool()  # (bs, num_classes+1, H, W)
        loss = dict()
        # 计算IoU准确率（用于评估，不需要梯度）
        # seg_label[:,:-1,...]是各类别标签，seg_label[:,-1,...]是有效区域mask
        loss['acc_seg'] = iou(seg_logit.detach().sigmoid()>0.5, seg_label[:,:-1,...], seg_label[:,-1,...])
        # 计算分割损失
        loss['loss_seg'] = self.criterion(seg_logit,seg_label[:,:-1,...],seg_label[:,-1,...])
        return loss
