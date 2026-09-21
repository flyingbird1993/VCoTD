"""Evaluate a VCoTD checkpoint against cached teacher features."""

import argparse
import json
import logging
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.vcotd_student import build_student_from_checkpoint
from train_vcotd import VCoTDDataset, collate_fn, evaluate


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--teacher_feat_dir', default='./teacher_features')
    parser.add_argument('--metadata', default='./create_data/cached_nuscenes_info.pkl')
    parser.add_argument('--split_file', default='./create_data/full_split.json')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    with open(args.split_file) as stream:
        split_info = json.load(stream)
    with open(args.metadata, 'rb') as stream:
        trajectory_data = pickle.load(stream)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_student_from_checkpoint(checkpoint).to(device)
    dataset = VCoTDDataset(
        split='val',
        data=trajectory_data,
        tokens=split_info['val'],
        teacher_feat_dir=args.teacher_feat_dir,
        mode='distill',
        T_obs=model.T_obs,
        T_pred=model.T_pred,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    print(
        f"Checkpoint: {Path(args.checkpoint)} | epoch={checkpoint.get('epoch')} | "
        f"step={checkpoint.get('step')}"
    )
    print(
        f"Model: d_vlm={model.d_vlm}, aggregation={model.visual_aggregation}, "
        f"vis_tokens={model.vis_tokens}"
    )
    metrics = evaluate(model, loader, device, T_pred=model.T_pred, mode='distill')
    print("\n=== VCoTD validation ===")
    print(f"official-mask L2 @ 1s: {metrics['l2_1s']:.4f} m")
    print(f"official-mask L2 @ 2s: {metrics['l2_2s']:.4f} m")
    print(f"official-mask L2 @ 3s: {metrics['l2_3s']:.4f} m")
    print(f"official-mask avg L2 : {metrics['l2_avg']:.4f} m")
    print(f"valid-only avg L2    : {metrics['l2_avg_valid_only']:.4f} m")


if __name__ == '__main__':
    main(parse_args())
