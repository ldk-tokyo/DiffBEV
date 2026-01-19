# Copyright (c) OpenMMLab. All rights reserved.
import warnings

# 尝试从 mmcv.cnn 导入 MODELS（mmcv 1.x），如果失败则创建新的 Registry
try:
    from mmcv.cnn import MODELS as MMCV_MODELS
    from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
    from mmengine.registry import Registry
    
    # 检查 MMCV_MODELS 是否是 Registry 实例
    from mmengine.registry import Registry as RegistryType
    # 检查 parent 参数是否可用（某些版本可能不支持）
    try:
        if isinstance(MMCV_MODELS, RegistryType):
            # 指定 scope 为 'mmseg'，避免自动导入问题
            MODELS = Registry('models', parent=MMCV_MODELS, scope='mmseg')
            ATTENTION = Registry('attention', parent=MMCV_ATTENTION, scope='mmseg')
        else:
            # 如果不是 Registry 实例，创建新的 Registry
            MODELS = Registry('models', scope='mmseg')
            ATTENTION = Registry('attention', scope='mmseg')
    except (TypeError, AssertionError):
        # 如果 parent 参数有问题，创建独立的 Registry
        MODELS = Registry('models', scope='mmseg')
        ATTENTION = Registry('attention', scope='mmseg')
except (ImportError, AttributeError):
    # 如果导入失败，创建独立的 Registry
    from mmengine.registry import Registry
    MODELS = Registry('models', scope='mmseg')
    ATTENTION = Registry('attention', scope='mmseg')

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
SEGMENTORS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """Build segmentor."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return SEGMENTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
