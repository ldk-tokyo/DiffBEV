#!/usr/bin/env python
"""
分割训练诊断工具：统计GT/Pred分布、per-class IoU、并保存可视化。
"""
import argparse
import os
import os.path as osp
import random

import numpy as np
import torch

from mmengine import Config as mmcv_Config
from mmengine.runner import load_checkpoint

try:
    from mmcv.parallel import collate, scatter
except ImportError:
    try:
        from mmengine.parallel import collate, scatter
    except ImportError:
        from torch.utils.data.dataloader import default_collate as _torch_collate

        def collate(batch, samples_per_gpu=None):
            return _torch_collate(batch)

        def scatter(inputs, target_gpus, dim=0):
            if isinstance(inputs, (list, tuple)):
                return [scatter(item, target_gpus, dim) for item in inputs]
            if isinstance(inputs, dict):
                return {k: scatter(v, target_gpus, dim) for k, v in inputs.items()}
            return inputs

from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor


def parse_args():
    parser = argparse.ArgumentParser(description='Diagnose segmentation training issues.')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--num-samples', type=int, default=50, help='Number of samples to diagnose')
    parser.add_argument('--vis-num', type=int, default=8, help='Number of samples to visualize')
    parser.add_argument('--out-dir', default='runs/diagnose', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--device', default='cuda:0', help='Device (cuda:0/cpu)')
    parser.add_argument('--camera', default='CAM_FRONT', help='Camera name to filter (if available)')
    parser.add_argument('--dataset-split', default='val', choices=['val', 'test', 'train'],
                        help='Dataset split to use')
    parser.add_argument('--probe-only', action='store_true', help='Probe batch structure only')
    return parser.parse_args()


def _unwrap_dc(obj):
    if hasattr(obj, 'data') and hasattr(obj, 'stack') and hasattr(obj, 'padding_value'):
        return obj.data
    return obj


def _unwrap_meta(meta):
    # unwrap nested DataContainer/list wrappers
    while hasattr(meta, 'data') and hasattr(meta, 'stack') and hasattr(meta, 'padding_value'):
        meta = meta.data
    # unwrap nested list-of-list to dict
    while isinstance(meta, list) and len(meta) == 1:
        meta = meta[0]
        if hasattr(meta, 'data') and hasattr(meta, 'stack') and hasattr(meta, 'padding_value'):
            meta = meta.data
    return meta


def _describe_value(name, value, indent='  '):
    def _shape_dtype(x):
        if isinstance(x, torch.Tensor):
            return tuple(x.shape), str(x.dtype)
        if isinstance(x, np.ndarray):
            return x.shape, str(x.dtype)
        return None, None

    lines = []
    if hasattr(value, 'data') and hasattr(value, 'stack') and hasattr(value, 'padding_value'):
        data = value.data
        shape, dtype = _shape_dtype(data)
        lines.append(f"{indent}{name}: DataContainer -> {type(data)} shape={shape} dtype={dtype}")
        return lines
    if isinstance(value, list):
        lines.append(f"{indent}{name}: list(len={len(value)})")
        if len(value) > 0:
            lines += _describe_value(f"{name}[0]", value[0], indent=indent + '  ')
        return lines
    shape, dtype = _shape_dtype(value)
    lines.append(f"{indent}{name}: {type(value)} shape={shape} dtype={dtype}")
    return lines


def _describe_meta(meta, indent='  '):
    lines = []
    if not isinstance(meta, dict):
        lines.append(f"{indent}meta: {type(meta)}")
        return lines
    lines.append(f"{indent}meta keys: {list(meta.keys())}")
    for k in ['filename', 'ori_filename', 'img_shape', 'pad_shape', 'ori_shape', 'scale_factor']:
        if k in meta:
            lines.append(f"{indent}meta.{k}: {meta[k]}")
    if 'gt_semantic_seg' in meta:
        gt = meta['gt_semantic_seg']
        gt = _unwrap_dc(gt)
        gt = _to_numpy(gt)
        lines.append(f"{indent}meta.gt_semantic_seg shape={getattr(gt, 'shape', None)} dtype={getattr(gt, 'dtype', None)}")
        if isinstance(gt, np.ndarray):
            uniq = np.unique(gt)
            lines.append(f"{indent}meta.gt_semantic_seg unique={uniq.tolist()}")
    return lines


def _prepare_model_inputs(batch, device):
    """Unwrap DataContainer and move tensors to device (similar to runner_compat)."""
    unwrapped = {}
    for key, value in batch.items():
        if hasattr(value, 'data') and hasattr(value, 'stack') and hasattr(value, 'padding_value'):
            data = value.data
            if not getattr(value, 'cpu_only', False) and isinstance(data, torch.Tensor):
                data = data.to(device)
            unwrapped[key] = data
        elif isinstance(value, torch.Tensor):
            unwrapped[key] = value.to(device)
        else:
            unwrapped[key] = value

    if 'img_metas' in unwrapped:
        img_metas = unwrapped['img_metas']
        if isinstance(img_metas, list):
            metas = []
            for meta in img_metas:
                if hasattr(meta, 'data'):
                    metas.append(meta.data)
                else:
                    metas.append(meta)
            unwrapped['img_metas'] = metas
    # ensure img tensor is on device if nested list (augmented)
    if 'img' in unwrapped:
        imgs = unwrapped['img']
        if isinstance(imgs, list):
            moved = []
            for img in imgs:
                if isinstance(img, torch.Tensor):
                    moved.append(img.to(device))
                else:
                    moved.append(img)
            unwrapped['img'] = moved
    return unwrapped


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _denormalize_img(img, img_norm_cfg):
    img = img.astype(np.float32)
    mean = np.array(img_norm_cfg.get('mean', [0, 0, 0]), dtype=np.float32)
    std = np.array(img_norm_cfg.get('std', [1, 1, 1]), dtype=np.float32)
    to_rgb = img_norm_cfg.get('to_rgb', False)
    img = img * std + mean
    img = np.clip(img, 0, 255)
    if to_rgb:
        img = img[..., ::-1]
    return img.astype(np.uint8)


def _palette_from_dataset(dataset, num_classes):
    palette = getattr(dataset, 'PALETTE', None)
    if palette is not None and len(palette) >= num_classes:
        return np.array(palette[:num_classes], dtype=np.uint8)
    # fallback fixed palette (tab20-like)
    colors = []
    for i in range(num_classes):
        colors.append(((i * 37) % 255, (i * 17) % 255, (i * 29) % 255))
    return np.array(colors, dtype=np.uint8)


def _colorize_mask(mask, palette, ignore_mask=None):
    h, w = mask.shape
    color = palette[mask.clip(0, len(palette) - 1)]
    if ignore_mask is not None:
        color[~ignore_mask] = np.array([0, 0, 0], dtype=np.uint8)
    return color.reshape(h, w, 3)


def _overlay(img, mask_rgb, alpha=0.5):
    if img.shape[:2] != mask_rgb.shape[:2]:
        # resize mask to image size for visualization
        import cv2
        mask_rgb = cv2.resize(mask_rgb, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    blended = img.astype(np.float32) * (1 - alpha) + mask_rgb.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _extract_gt_from_batch(batch, num_classes, ignore_index):
    # priority: DataSample-like in batch
    if 'data_samples' in batch:
        data_samples = batch['data_samples']
        if isinstance(data_samples, list) and len(data_samples) > 0:
            sample = data_samples[0]
            if hasattr(sample, 'gt_sem_seg'):
                gt = sample.gt_sem_seg.data
                gt = _to_numpy(gt)
                if gt.ndim == 3 and gt.shape[0] == 1:
                    gt = gt[0]
                valid_mask = gt != ignore_index
                return gt.astype(np.int64), valid_mask, 'label'
    # common mmcv pipeline: gt in img_metas
    if 'img_metas' in batch:
        meta = _unwrap_meta(batch['img_metas'])
        if isinstance(meta, list):
            meta = meta[0]
        if isinstance(meta, dict) and 'gt_semantic_seg' in meta:
            gt = meta['gt_semantic_seg']
            gt = _unwrap_dc(gt)
            gt = _to_numpy(gt)
            if gt.ndim == 4 and gt.shape[0] == 1:
                gt = gt[0]
            if gt.ndim == 3 and gt.shape[0] in (num_classes, num_classes + 1):
                gt_binary = gt[:num_classes].astype(bool)
                valid_mask = None
                if gt.shape[0] == num_classes + 1:
                    valid_mask = gt[-1].astype(bool)
                return gt_binary, valid_mask, 'binary'
            if gt.ndim == 2:
                valid_mask = gt != ignore_index
                return gt.astype(np.int64), valid_mask, 'label'
    # fallback: gt_semantic_seg key in batch
    if 'gt_semantic_seg' in batch:
        gt = _unwrap_dc(batch['gt_semantic_seg'])
        gt = _to_numpy(gt)
        if gt.ndim == 4 and gt.shape[0] == 1:
            gt = gt[0]
        if gt.ndim == 3 and gt.shape[0] in (num_classes, num_classes + 1):
            gt_binary = gt[:num_classes].astype(bool)
            valid_mask = None
            if gt.shape[0] == num_classes + 1:
                valid_mask = gt[-1].astype(bool)
            return gt_binary, valid_mask, 'binary'
        if gt.ndim == 2:
            valid_mask = gt != ignore_index
            return gt.astype(np.int64), valid_mask, 'label'
    return None, None, 'unknown'


def _parse_gt_array(gt, num_classes, ignore_index):
    gt = _to_numpy(gt)
    if gt.ndim == 4 and gt.shape[0] == 1:
        gt = gt[0]
    # binary mask format: (C, H, W) or (C+1, H, W)
    if gt.ndim == 3 and gt.shape[0] in (num_classes, num_classes + 1):
        gt_binary = gt[:num_classes].astype(bool)
        valid_mask = None
        if gt.shape[0] == num_classes + 1:
            valid_mask = gt[-1].astype(bool)
        return gt_binary, valid_mask, 'binary'
    # label map format: (H, W)
    if gt.ndim == 2:
        valid_mask = gt != ignore_index
        return gt.astype(np.int64), valid_mask, 'label'
    return None, None, 'unknown'


def _binary_to_label(gt_binary, background_id=0):
    """Convert multi-hot binary masks to a single-label map."""
    h, w = gt_binary.shape[1:]
    label = np.full((h, w), background_id, dtype=np.int64)
    # deterministic: first positive class wins
    for c in range(gt_binary.shape[0]):
        mask = gt_binary[c] & (label == background_id)
        label[mask] = c
    return label


def get_gt_mask(batch, num_classes, ignore_index, dataset=None, idx=None):
    gt, valid_mask, gt_format = _extract_gt_from_batch(batch, num_classes, ignore_index)
    gt_source = 'batch'
    if gt is None and dataset is not None and idx is not None:
        gt_raw = dataset.get_gt_seg_map_by_idx(idx)
        gt, valid_mask, gt_format = _parse_gt_array(gt_raw, num_classes, ignore_index)
        gt_source = 'dataset.get_gt_seg_map_by_idx'
    if gt is None:
        raise RuntimeError('GT mask not found in batch or dataset. Please run with --probe-only.')
    gt_binary = None
    if gt_format == 'binary':
        gt_binary = gt
        if gt_binary.ndim == 4:
            gt_binary = gt_binary[0]
        gt_label = _binary_to_label(gt_binary, background_id=0)
    else:
        gt_label = gt.astype(np.int64)

    unique, counts = np.unique(gt_label, return_counts=True)
    print(f"GT source: {gt_source}")
    print(f"GT unique: {unique.tolist()} | counts: {counts.tolist()}")
    if gt_label.max() >= num_classes:
        print("⚠️  GT label max exceeds num_classes. "
              "If values look like bit-encoded labels, prefer pipeline meta GT.")
    if gt_label.size == 0 or counts.sum() == 0:
        raise RuntimeError('GT mask is empty. Please check pipeline/annotations.')
    return gt_label, valid_mask, gt_format, gt_binary


def get_pred_mask(pred_output):
    pred = pred_output
    if isinstance(pred, list) and len(pred) > 0:
        pred0 = pred[0]
    else:
        pred0 = pred
    if isinstance(pred0, dict) and 'seg_logits' in pred0:
        logits = pred0['seg_logits']
        if isinstance(logits, torch.Tensor):
            print(f"seg_logits shape: {tuple(logits.shape)}")
    if hasattr(pred0, 'pred_sem_seg'):
        pred_mask = pred0.pred_sem_seg.data
        pred_mask = _to_numpy(pred_mask)
        if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
            pred_mask = pred_mask[0]
    else:
        pred_mask = pred0
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.detach().cpu().numpy()
    if not isinstance(pred_mask, np.ndarray) or pred_mask.ndim != 2:
        raise RuntimeError(f'Pred mask shape invalid: {type(pred_mask)} {getattr(pred_mask, "shape", None)}')
    unique, counts = np.unique(pred_mask, return_counts=True)
    print(f"Pred unique: {unique.tolist()} | counts: {counts.tolist()}")
    if pred_mask.size == 0 or counts.sum() == 0:
        raise RuntimeError('Pred mask is empty. Please check model output.')
    return pred_mask.astype(np.int64)


def main():
    args = parse_args()
    _ensure_dir(args.out_dir)
    summary_path = osp.join(args.out_dir, 'summary.txt')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = mmcv_Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    # select dataset split
    dataset_cfg = None
    if args.dataset_split in cfg.data:
        dataset_cfg = cfg.data.get(args.dataset_split)
    else:
        dataset_cfg = cfg.data.get('val', cfg.data.get('test', cfg.data.get('train')))
    if dataset_cfg is None:
        raise RuntimeError('Failed to locate dataset config in cfg.data')

    # ensure test_mode is False so ann_info exists
    dataset_cfg = dataset_cfg.copy()
    dataset_cfg['test_mode'] = False
    dataset = build_dataset(dataset_cfg)

    # reduce_zero_label consistency check
    dataset_reduce = getattr(dataset, 'reduce_zero_label', None)
    pipeline_reduce = None
    for step in dataset_cfg.get('pipeline', []):
        if isinstance(step, dict) and step.get('type') == 'LoadAnnotations':
            pipeline_reduce = step.get('reduce_zero_label', None)
            break
    cfg_reduce = dataset_cfg.get('reduce_zero_label', None)
    if cfg_reduce is not None:
        print(f"Config reduce_zero_label: {cfg_reduce}")
    if dataset_reduce is not None and pipeline_reduce is not None and dataset_reduce != pipeline_reduce:
        print("\033[0;31m❌ reduce_zero_label mismatch detected.\033[0m")
        print(f"  dataset.reduce_zero_label={dataset_reduce}")
        print(f"  pipeline.LoadAnnotations.reduce_zero_label={pipeline_reduce}")
        print("请先修复config后再运行诊断工具。")
        return

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        model.PALETTE = dataset.PALETTE

    device = torch.device(args.device)
    model.to(device)
    model.eval()

    # config sanity check info
    num_classes_sources = []
    if hasattr(model, 'num_classes'):
        num_classes_sources.append(('model.num_classes', model.num_classes))
    if hasattr(model, 'decode_head') and hasattr(model.decode_head, 'num_classes'):
        num_classes_sources.append(('model.decode_head.num_classes', model.decode_head.num_classes))
    if hasattr(cfg, 'model') and 'decode_head' in cfg.model and 'num_classes' in cfg.model.decode_head:
        num_classes_sources.append(('cfg.model.decode_head.num_classes', cfg.model.decode_head.num_classes))
    if hasattr(dataset, 'CLASSES') and dataset.CLASSES is not None:
        num_classes_sources.append(('dataset.CLASSES', len(dataset.CLASSES)))
    num_classes = num_classes_sources[0][1]

    ignore_sources = [('dataset.ignore_index', getattr(dataset, 'ignore_index', None))]
    if 'ignore_index' in dataset_cfg:
        ignore_sources.append(('cfg.data.' + args.dataset_split + '.ignore_index', dataset_cfg['ignore_index']))
    ignore_index = ignore_sources[0][1] if ignore_sources[0][1] is not None else 255

    reduce_sources = [('dataset.reduce_zero_label', getattr(dataset, 'reduce_zero_label', None))]
    # LoadAnnotations reduce_zero_label from pipeline (if exists)
    for step in dataset_cfg.get('pipeline', []):
        if isinstance(step, dict) and step.get('type') == 'LoadAnnotations':
            if 'reduce_zero_label' in step:
                reduce_sources.append(('pipeline.LoadAnnotations.reduce_zero_label', step['reduce_zero_label']))
            break

    lines = []
    lines.append("Config sanity check")
    for src, val in num_classes_sources:
        lines.append(f"  num_classes ({src}): {val}")
    for src, val in ignore_sources:
        lines.append(f"  ignore_index ({src}): {val}")
    for src, val in reduce_sources:
        lines.append(f"  reduce_zero_label ({src}): {val}")
    lines.append("")

    # dataloader for probe
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=min(cfg.data.workers_per_gpu, 1) if hasattr(cfg, 'data') else 1,
        dist=False,
        shuffle=False)

    if args.probe_only:
        batch = next(iter(data_loader))
        print("=== Probe: batch structure ===")
        for key, value in batch.items():
            for line in _describe_value(key, value):
                print(line)
        # inspect data_sample if exists
        if 'data_samples' in batch:
            data_samples = batch['data_samples']
            if isinstance(data_samples, list) and len(data_samples) > 0:
                sample = data_samples[0]
                print("data_sample.__dict__:", getattr(sample, '__dict__', {}))
                if hasattr(sample, 'gt_sem_seg'):
                    gt = sample.gt_sem_seg.data
                    gt = _to_numpy(gt)
                    print(f"data_sample.gt_sem_seg: shape={gt.shape} dtype={gt.dtype}")
                    print(f"data_sample.gt_sem_seg unique={np.unique(gt).tolist()}")
                if hasattr(sample, 'pred_sem_seg'):
                    pred = sample.pred_sem_seg.data
                    pred = _to_numpy(pred)
                    print(f"data_sample.pred_sem_seg: shape={pred.shape} dtype={pred.dtype}")
                if hasattr(sample, 'seg_logits'):
                    logits = sample.seg_logits.data
                    logits = _to_numpy(logits)
                    print(f"data_sample.seg_logits: shape={logits.shape} dtype={logits.dtype}")
        # print meta details
        if 'img_metas' in batch:
            meta = _unwrap_meta(batch['img_metas'])
            for line in _describe_meta(meta):
                print(line)

        # extract GT mask and save (fallback to dataset if not in batch)
        gt_label, valid_mask, _, _ = get_gt_mask(batch, num_classes, ignore_index, dataset=dataset, idx=0)
        np.save(osp.join(args.out_dir, 'gt_probe.npy'), gt_label)
        if valid_mask is not None:
            np.save(osp.join(args.out_dir, 'gt_probe_valid.npy'), valid_mask.astype(np.uint8))
        # visualize
        meta = _unwrap_meta(batch['img_metas'])
        if isinstance(meta, dict):
            for k in ['img_shape', 'pad_shape', 'ori_shape', 'scale_factor']:
                if k in meta:
                    print(f"meta.{k}: {meta[k]}")
        img = None
        if 'img' in batch:
            img_val = _unwrap_dc(batch['img'])
            if isinstance(img_val, list):
                img_val = _unwrap_dc(img_val[0])
            if isinstance(img_val, torch.Tensor):
                img_np = img_val.detach().cpu().numpy()
                if img_np.ndim == 4:
                    img_np = img_np[0]
                img_np = img_np.transpose(1, 2, 0)
                img_norm_cfg = meta.get('img_norm_cfg', {}) if isinstance(meta, dict) else {}
                img = _denormalize_img(img_np, img_norm_cfg)
            elif isinstance(img_val, np.ndarray):
                img = img_val
        if img is not None:
            gt_color = _colorize_mask(gt_label, _palette_from_dataset(dataset, num_classes),
                                      valid_mask if valid_mask is not None else None)
            import cv2
            cv2.imwrite(osp.join(args.out_dir, 'probe_img.png'), img[..., ::-1])
            cv2.imwrite(osp.join(args.out_dir, 'probe_gt.png'), gt_color[..., ::-1])
        print(f"Probe outputs saved to: {osp.abspath(args.out_dir)}")
        return

    # sampling indices
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    sample_indices = all_indices[:args.num_samples]

    gt_counts = np.zeros(num_classes, dtype=np.int64)
    pred_counts = np.zeros(num_classes, dtype=np.int64)
    total_pixels = 0
    ignore_pixels = 0

    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)  # gt x pred
    multi_label_pixels = 0
    processed = 0
    vis_files = []

    palette = _palette_from_dataset(dataset, num_classes)

    camera_filter_enabled = True if args.camera else False
    checked_camera_filter = False

    sample_indices_set = set(sample_indices)

    with torch.no_grad():
        for idx, batch_cpu in enumerate(data_loader):
            if idx not in sample_indices_set:
                continue
            meta = _unwrap_meta(batch_cpu['img_metas'])
            filename = meta.get('filename', meta.get('ori_filename', 'unknown')) if isinstance(meta, dict) else 'unknown'
            if camera_filter_enabled and isinstance(filename, str):
                if not checked_camera_filter:
                    checked_camera_filter = True
                    if args.camera not in filename:
                        print(f"⚠️  camera filter '{args.camera}' not found in filename. "
                              "Disable camera filter to avoid skipping all samples.")
                        camera_filter_enabled = False
                if camera_filter_enabled and args.camera not in filename:
                    continue

            # prepare data for model
            batch_model = _prepare_model_inputs(batch_cpu, device)

            gt_label, valid_mask, gt_format, gt_binary = get_gt_mask(
                batch_cpu, num_classes, ignore_index, dataset=dataset, idx=idx
            )
            pred_output = model(return_loss=False, rescale=True, **batch_model)
            pred = get_pred_mask(pred_output)

            # build gt label map for stats
            if gt_binary is not None and isinstance(gt_binary, np.ndarray) and gt_binary.ndim == 3:
                multi_label_pixels += int((gt_binary.sum(axis=0) > 1).sum())

            if valid_mask is None:
                valid_mask = np.ones_like(gt_label, dtype=bool)

            if gt_label.shape != pred.shape:
                # resize pred to gt for comparison
                import cv2
                pred = cv2.resize(pred, (gt_label.shape[1], gt_label.shape[0]), interpolation=cv2.INTER_NEAREST)

            total_pixels += valid_mask.size
            ignore_pixels += int((~valid_mask).sum())

            # gt distribution
            if gt_binary is not None and gt_binary.ndim == 3:
                for c in range(num_classes):
                    gt_counts[c] += int((gt_binary[c] & valid_mask).sum())
            else:
                for c in range(num_classes):
                    gt_counts[c] += int(((gt_label == c) & valid_mask).sum())
                # check out-of-range labels
                unique = np.unique(gt_label)
                invalid = [v for v in unique if v not in list(range(num_classes)) and v != ignore_index]
                if invalid:
                    lines.append(f"⚠️  GT label out of range at idx={idx}: {invalid}")

            # pred distribution
            for c in range(num_classes):
                pred_counts[c] += int(((pred == c) & valid_mask).sum())

            # confusion matrix
            for c in range(num_classes):
                gt_c = (gt_label == c)
                if gt_format == 'binary':
                    gt_c = gt_binary[c]
                gt_c = gt_c & valid_mask
                for p in range(num_classes):
                    conf_mat[c, p] += int((gt_c & (pred == p)).sum())

            # visualization
            if processed < args.vis_num:
                img = _unwrap_dc(batch_cpu['img'])
                if isinstance(img, list):
                    img = img[0]
                if isinstance(img, torch.Tensor):
                    img_np = img.detach().cpu().numpy()
                    if img_np.ndim == 3:
                        img_np = img_np.transpose(1, 2, 0)
                    elif img_np.ndim == 4:
                        img_np = img_np[0].transpose(1, 2, 0)
                    img_norm_cfg = meta.get('img_norm_cfg', {})
                    img_vis = _denormalize_img(img_np, img_norm_cfg)
                else:
                    img_vis = _to_numpy(img)
                    if isinstance(img_vis, list):
                        img_vis = _to_numpy(img_vis[0])
                    if hasattr(img_vis, 'ndim') and img_vis.ndim == 3 and img_vis.shape[2] == 3:
                        img_vis = img_vis.astype(np.uint8)

                gt_vis = gt_label
                pred_vis = pred

                gt_color = _colorize_mask(gt_vis, palette, valid_mask)
                pred_color = _colorize_mask(pred_vis, palette, valid_mask)

                gt_overlay = _overlay(img_vis, gt_color, alpha=0.5)
                pred_overlay = _overlay(img_vis, pred_color, alpha=0.5)

                base = osp.splitext(osp.basename(str(filename)))[0]
                prefix = f"{idx:06d}_{base}"

                files = {
                    'img': osp.join(args.out_dir, f"{prefix}_img.png"),
                    'gt': osp.join(args.out_dir, f"{prefix}_gt.png"),
                    'pred': osp.join(args.out_dir, f"{prefix}_pred.png"),
                    'gt_overlay': osp.join(args.out_dir, f"{prefix}_gt_overlay.png"),
                    'pred_overlay': osp.join(args.out_dir, f"{prefix}_pred_overlay.png"),
                }
                import cv2
                cv2.imwrite(files['img'], img_vis[..., ::-1] if img_vis.shape[2] == 3 else img_vis)
                cv2.imwrite(files['gt'], gt_color[..., ::-1])
                cv2.imwrite(files['pred'], pred_color[..., ::-1])
                cv2.imwrite(files['gt_overlay'], gt_overlay[..., ::-1])
                cv2.imwrite(files['pred_overlay'], pred_overlay[..., ::-1])
                vis_files.extend(list(files.values()))

            processed += 1

    valid_pixels = total_pixels - ignore_pixels
    if processed == 0:
        raise RuntimeError('No samples processed. Check --camera filter or dataset.')
    lines.append("GT distribution")
    lines.append("label_id | pixel_count | ratio")
    for c in range(num_classes):
        ratio = gt_counts[c] / max(valid_pixels, 1)
        lines.append(f"{c:7d} | {gt_counts[c]:11d} | {ratio:.6f}")
    ignore_ratio = ignore_pixels / max(total_pixels, 1)
    lines.append(f"ignore_ratio: {ignore_ratio:.6f}")
    lines.append(f"valid_ratio: {1 - ignore_ratio:.6f}")
    if multi_label_pixels > 0:
        lines.append(f"⚠️  multi-label pixels detected: {multi_label_pixels}")
    lines.append("")

    lines.append("Pred distribution")
    lines.append("label_id | pixel_count | ratio")
    for c in range(num_classes):
        ratio = pred_counts[c] / max(valid_pixels, 1)
        lines.append(f"{c:7d} | {pred_counts[c]:11d} | {ratio:.6f}")
    collapse_ratio = pred_counts.max() / max(valid_pixels, 1)
    if collapse_ratio > 0.95:
        lines.append(f"⚠️  Pred collapse detected: max_ratio={collapse_ratio:.6f}")
    lines.append("")

    # metrics
    lines.append("Metrics")
    # per-class IoU
    ious = []
    lines.append("class_id | IoU | gt_pixels | pred_pixels")
    for c in range(num_classes):
        tp = conf_mat[c, c]
        gt_p = conf_mat[c, :].sum()
        pred_p = conf_mat[:, c].sum()
        union = gt_p + pred_p - tp
        if gt_p > 0:
            iou = tp / max(union, 1)
            ious.append(iou)
        else:
            iou = float('nan')
        lines.append(f"{c:8d} | {iou:.6f} | {gt_p:9d} | {pred_p:10d}")
    miou = float(np.nanmean(ious)) if len(ious) > 0 else 0.0
    overall_acc = np.trace(conf_mat) / max(conf_mat.sum(), 1)
    lines.append(f"overall_acc: {overall_acc:.6f}")
    lines.append(f"mIoU: {miou:.6f} (exclude classes with gt_pixels=0)")
    lines.append("")

    lines.append("Artifacts")
    lines.append(f"output_dir: {osp.abspath(args.out_dir)}")
    if vis_files:
        lines.append("files:")
        for f in vis_files[:20]:
            lines.append(f"  - {osp.basename(f)}")
        if len(vis_files) > 20:
            lines.append(f"  ... ({len(vis_files)} files total)")
    else:
        lines.append("files: (none)")

    output_text = "\n".join(lines)
    print(output_text)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(output_text)


if __name__ == '__main__':
    main()
