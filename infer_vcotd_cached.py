"""
VCoTD Cached-Feature Inference — no Qwen2-VL encoder required at inference time.

Reads pre-extracted teacher features from teacher_features/val/*.pt directly,
so inference only requires the 4.3M student model.

Output format is compatible with tools/evaluation/evaluation.py.

Usage:
    cd FSDrive-main
    python infer_vcotd_cached.py \
        --ckpt saves/vcotd_distill_single/checkpoint_best.pt \
        --feat_dir teacher_features/val \
        --output_path results_ablation_full.json

    python tools/evaluation/evaluation.py \
        --metric stp3 \
        --result_file results_ablation_full.json \
        --method VCoTD-full
"""

import os
import json
import pickle
import argparse
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from model.vcotd_student import build_student_from_checkpoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def load_hist(d, T_obs=6):
    his = torch.tensor(d['gt_ego_his_trajs'], dtype=torch.float32)
    if his.shape[0] >= T_obs:
        return his[-T_obs:]
    pad = torch.zeros(T_obs - his.shape[0], 2)
    return torch.cat([pad, his], dim=0)


def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log.info("Loading nuScenes metadata...")
    data       = pickle.load(open('./create_data/cached_nuscenes_info.pkl', 'rb'))
    split_info = json.load(open('./create_data/full_split.json'))
    tokens     = split_info[args.split]

    log.info(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_student_from_checkpoint(ckpt).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(
        f"Student params: {total_params:.2f}M | d_vlm={model.d_vlm} | "
        f"aggregation={model.visual_aggregation} | vis_tokens={model.vis_tokens}"
    )

    feat_dir = Path(args.feat_dir)
    results  = {}
    missing  = 0

    for token in tqdm(tokens, desc='Inferring'):
        if token not in data:
            missing += 1
            continue

        feat_path = feat_dir / f"{token}.pt"
        if args.zero_feat or not feat_path.exists():
            feat = torch.zeros(1, model.vis_tokens, model.d_vlm, device=device)
        else:
            feat_dict = torch.load(feat_path, map_location=device, weights_only=False)
            feat = feat_dict['deep'].float().unsqueeze(0)

        hist = load_hist(data[token], model.T_obs).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model.predict(feat, hist)        # (1, T_pred, 2)

        traj = pred[0].cpu()
        results[token] = '[' + ', '.join(f'({x:.2f},{y:.2f})' for x, y in traj.tolist()) + ']'

    log.info(f"Done: {len(results)} results, {missing} missing tokens")

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, 'w'), indent=2)
    log.info(f"Saved → {out}")
    log.info(f"Evaluate with: python tools/evaluation/evaluation.py --metric {args.metric} "
             f"--result_file {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',        type=str, required=True)
    p.add_argument('--feat_dir',    type=str,
                   default='/media/flyingbird/07419DF8D71B0526/Dataset/nuScenes/teacher_features/val')
    p.add_argument('--split',       type=str, default='val', choices=['train', 'val'])
    p.add_argument('--output_path', type=str, default='results_ablation.json')
    p.add_argument('--metric',      type=str, default='stp3', choices=['uniad', 'stp3'])
    p.add_argument('--zero_feat',   action='store_true',
                   help='Use zero visual features (for gt_only / pretrain baseline)')
    return p.parse_args()


if __name__ == '__main__':
    infer(parse_args())
