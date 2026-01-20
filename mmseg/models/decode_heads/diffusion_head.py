import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .pyramid_head import PyramidHead, OccupancyCriterion, iou


@HEADS.register_module()
class DiffusionHead(PyramidHead):
    """Diffusion-enabled head for BEV segmentation.

    This implements a lightweight diffusion loss on the segmentation logits.
    It supports FS-BEV conditioning and Cross-Attention fusion by default.
    """

    def __init__(self,
                 num_classes,
                 align_corners=True,
                 in_channels=64,
                 channels=128,
                 layers=(4, 4),
                 strides=(1, 2),
                 use_diffusion=True,
                 diffusion_steps=1000,
                 noise_schedule='linear',
                 condition_type=None,
                 fusion_type=None,
                 xt_encoding=None,
                 depth_range=(1.0, 50.0),
                 loss_depth_weight=10.0,
                 loss_diff_weight=1.0,
                 **kwargs):
        super().__init__(
            num_classes=num_classes,
            align_corners=align_corners,
            in_channels=in_channels,
            channels=channels,
            layers=list(layers),
            strides=list(strides),
            **kwargs,
        )

        # Diffusion config
        self.use_diffusion = use_diffusion
        self.diffusion_steps = diffusion_steps
        self.noise_schedule = noise_schedule
        self.condition_type = condition_type
        self.fusion_type = fusion_type
        self.xt_encoding = xt_encoding
        self.depth_range = depth_range
        self.loss_depth_weight = loss_depth_weight
        self.loss_diff_weight = loss_diff_weight

        # Diffusion modules (lightweight)
        self.cond_channels = 128
        self.xt_encoder = nn.Sequential(
            nn.Conv2d(self.num_classes, self.cond_channels, 3, padding=1),
            nn.GroupNorm(8, self.cond_channels),
            nn.SiLU(),
        )
        self.cond_proj = nn.Sequential(
            nn.Conv2d(self.topdown.out_channels, self.cond_channels, 1),
            nn.GroupNorm(8, self.cond_channels),
            nn.SiLU(),
        )
        self.fuse_attn = nn.MultiheadAttention(
            embed_dim=self.cond_channels,
            num_heads=4,
            batch_first=True,
        )
        self.noise_pred = nn.Sequential(
            nn.Conv2d(self.cond_channels, self.cond_channels, 3, padding=1),
            nn.GroupNorm(8, self.cond_channels),
            nn.SiLU(),
            nn.Conv2d(self.cond_channels, self.num_classes, 1),
        )

        # Precompute diffusion schedule
        self.register_buffer('betas', self._make_beta_schedule())
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

        if not self.use_diffusion:
            warnings.warn("DiffusionHead initialized with use_diffusion=False.")

    def _make_beta_schedule(self):
        if self.noise_schedule == 'linear':
            return torch.linspace(1e-4, 2e-2, self.diffusion_steps)
        raise ValueError(f"Unknown noise_schedule: {self.noise_schedule}")

    def _timestep_embedding(self, t, dim, device):
        half = dim // 2
        freqs = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32) *
            (-math.log(10000.0) / (half - 1))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def _diffusion_loss(self, seg_logits, cond_feats):
        """Compute diffusion loss on segmentation logits."""
        if not self.use_diffusion or self.loss_diff_weight <= 0:
            return None

        # Use detached x0 to avoid destabilizing segmentation head
        x0 = seg_logits.detach()
        b = x0.shape[0]
        device = x0.device

        t = torch.randint(0, self.diffusion_steps, (b,), device=device)
        alpha_bar = self.alphas_cumprod[t].view(b, 1, 1, 1)
        noise = torch.randn_like(x0)
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise

        # Encode xt
        xt_feats = self.xt_encoder(xt)

        # Encode condition (FS-BEV uses topdown features)
        cond = self.cond_proj(cond_feats)

        # Fuse (Cross-Attention)
        if self.fusion_type is None or self.fusion_type.lower().replace('-', '_') == 'cross_attention':
            b, c, h, w = xt_feats.shape
            q = xt_feats.flatten(2).transpose(1, 2)  # B, HW, C
            k = cond.flatten(2).transpose(1, 2)
            v = k
            fused, _ = self.fuse_attn(q, k, v, need_weights=False)
            fused = fused.transpose(1, 2).view(b, c, h, w)
        else:
            # Fallback: simple fusion
            fused = xt_feats + cond

        # Add timestep embedding (xt encoding)
        if self.xt_encoding is None or self.xt_encoding == 'default':
            t_embed = self._timestep_embedding(t, self.cond_channels, device).view(b, self.cond_channels, 1, 1)
            fused = fused + t_embed

        pred_noise = self.noise_pred(fused)
        loss_diff = F.mse_loss(pred_noise, noise)
        return loss_diff

    def forward(self, inputs):
        # Inputs may include depth logits from transformer
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[0]
        td_feats = self.topdown(inputs)
        logits = self.classifier(td_feats)
        return logits, td_feats

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, gt_depth=None, gt_depth_mask=None, **kwargs):
        # Support depth outputs from transformer: (features, depth_logit)
        depth_logit = None
        if isinstance(inputs, (list, tuple)) and len(inputs) >= 2:
            inputs, depth_logit = inputs[0], inputs[1]

        seg_logits, td_feats = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)

        # Diffusion loss
        loss_diff = self._diffusion_loss(seg_logits, td_feats)
        if loss_diff is not None:
            losses['loss_diff'] = loss_diff * self.loss_diff_weight

        # Optional depth loss (if depth supervision is available)
        if (depth_logit is not None and gt_depth is not None and
                hasattr(self, 'loss_depth_weight') and self.loss_depth_weight > 0):
            loss_depth = self._depth_loss(depth_logit, gt_depth, gt_depth_mask)
            if loss_depth is not None:
                losses['loss_depth'] = loss_depth * self.loss_depth_weight

        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[0]
        return self.forward(inputs)[0]

    def _depth_loss(self, depth_logit, gt_depth, gt_depth_mask=None):
        """Depth supervision loss based on expected depth."""
        # depth_logit: [B, D, H, W], gt_depth: [B, 1, H', W']
        if depth_logit is None or gt_depth is None:
            return None

        b, d, h, w = depth_logit.shape
        gt = gt_depth.float()
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)
        if gt.shape[-2:] != (h, w):
            gt = F.interpolate(gt, size=(h, w), mode='bilinear', align_corners=False)
        mask = None
        if gt_depth_mask is not None:
            mask = gt_depth_mask.float()
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            if mask.shape[-2:] != (h, w):
                mask = F.interpolate(mask, size=(h, w), mode='nearest')

        # Build depth bins
        z_min, z_max = self.depth_range
        depth_bins = torch.linspace(z_min, z_max, d, device=depth_logit.device)
        depth_bins = depth_bins.view(1, d, 1, 1)
        prob = F.softmax(depth_logit, dim=1)
        depth_pred = (prob * depth_bins).sum(dim=1, keepdim=True)

        # L1 loss on depth (apply mask if available)
        if mask is not None:
            valid = mask > 0.5
            if valid.any():
                return F.l1_loss(depth_pred[valid], gt[valid])
            return None
        return F.l1_loss(depth_pred, gt)

    @staticmethod
    def losses(seg_logit, seg_label):
        seg_label = seg_label.squeeze(1).bool()  # (bs, num_classes+1, H, W)
        loss = dict()
        loss['acc_seg'] = iou(seg_logit.detach().sigmoid() > 0.5,
                              seg_label[:, :-1, ...],
                              seg_label[:, -1, ...])
        loss['loss_seg'] = OccupancyCriterion()(seg_logit,
                                                seg_label[:, :-1, ...],
                                                seg_label[:, -1, ...])
        return loss
