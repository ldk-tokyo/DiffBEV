#!/usr/bin/env python3
import argparse

from mmengine import Config
from mmseg.datasets import build_dataset, build_dataloader


def unwrap(dc):
    if hasattr(dc, 'data'):
        return dc.data
    return dc


def main():
    parser = argparse.ArgumentParser(description="Check depth GT in dataloader batch.")
    parser.add_argument('--config', type=str, required=True,
                        help='Config file path')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.train)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        pin_memory=False,
        persistent_workers=False,
    )

    batch = next(iter(data_loader))
    img = unwrap(batch.get('img'))
    gt_seg = unwrap(batch.get('gt_semantic_seg'))
    gt_depth = unwrap(batch.get('gt_depth', None))
    gt_depth_mask = unwrap(batch.get('gt_depth_mask', None))

    print("Batch keys:", list(batch.keys()))
    if img is not None:
        print("img:", getattr(img, 'shape', None))
    if gt_seg is not None:
        print("gt_semantic_seg:", getattr(gt_seg, 'shape', None))
    if gt_depth is not None:
        print("gt_depth:", getattr(gt_depth, 'shape', None))
        if hasattr(gt_depth, 'min'):
            print("gt_depth stats: min=%.4f max=%.4f" % (gt_depth.min(), gt_depth.max()))
    else:
        print("gt_depth: None")
    if gt_depth_mask is not None:
        print("gt_depth_mask:", getattr(gt_depth_mask, 'shape', None))
        if hasattr(gt_depth_mask, 'sum') and hasattr(gt_depth_mask, 'numel'):
            valid_ratio = float(gt_depth_mask.sum()) / float(gt_depth_mask.numel())
            print("gt_depth_mask valid_ratio: %.6f" % valid_ratio)
    else:
        print("gt_depth_mask: None")


if __name__ == '__main__':
    main()
