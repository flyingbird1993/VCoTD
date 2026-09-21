"""
VCoTD E2E Inference Script — no Qwen2-VL required.

Runs the full pipeline: raw images → LightEncoder → VCoTDStudentE2E → trajectory.
Output format is compatible with tools/evaluation/evaluation.py.

Usage:
    cd FSDrive-main
    python infer_vcotd_e2e.py \
        --ckpt saves/vcotd_e2e/checkpoint_best.pt \
        --split val \
        --output_path results_vcotd_e2e.json

    python tools/evaluation/evaluation.py \
        --metric uniad \
        --result_file results_vcotd_e2e.json \
        --method VCoTD-E2E
"""

import os
import glob
import json
import pickle
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

from model.light_encoder import LightEncoder, CAMERA_TYPES
from model.vcotd_student_e2e import VCoTDStudentE2E

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def build_image_index(nuscenes_root: str) -> dict:
    index = {}
    for path in glob.glob(os.path.join(nuscenes_root, '*', 'samples', '*', '*.jpg')):
        index[os.path.basename(path)] = path
    log.info(f"Image index: {len(index)} files")
    return index


# ── Dataset ───────────────────────────────────────────────────────────────────

class InferDataset(Dataset):
    def __init__(self, data: dict, tokens: list, image_index: dict, T_obs: int = 6):
        self.data        = data
        self.image_index = image_index
        self.T_obs       = T_obs
        self.transform   = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        self.valid_tokens = [t for t in tokens if t in data]
        log.info(f"Inference dataset: {len(self.valid_tokens)} samples")

    def __len__(self):
        return len(self.valid_tokens)

    def _load_image(self, orig_path: str) -> torch.Tensor:
        fn     = os.path.basename(orig_path)
        actual = self.image_index.get(fn)
        if actual is None:
            return torch.zeros(3, 224, 224)
        try:
            return self.transform(Image.open(actual).convert('RGB'))
        except Exception:
            return torch.zeros(3, 224, 224)

    def __getitem__(self, idx):
        token = self.valid_tokens[idx]
        d     = self.data[token]

        his = torch.tensor(d['gt_ego_his_trajs'], dtype=torch.float32)
        if his.shape[0] >= self.T_obs:
            hist_traj = his[-self.T_obs:]
        else:
            pad = torch.zeros(self.T_obs - his.shape[0], 2)
            hist_traj = torch.cat([pad, his], dim=0)

        images = torch.stack([
            self._load_image(d['cams'][cam]['data_path']) for cam in CAMERA_TYPES
        ])  # (6, 3, 224, 224)

        return token, hist_traj, images


def infer_collate(batch):
    tokens    = [b[0] for b in batch]
    hist_traj = torch.stack([b[1] for b in batch])
    images    = torch.stack([b[2] for b in batch])
    return tokens, hist_traj, images


# ── Efficiency benchmark ──────────────────────────────────────────────────────

def benchmark_latency(encoder, student, device, n_warmup=10, n_runs=50):
    """Measure per-sample inference latency (ms) on GPU."""
    encoder.eval()
    student.eval()
    dummy_imgs = torch.zeros(1, 6, 3, 224, 224, device=device)
    dummy_traj = torch.zeros(1, student.T_obs, 2, device=device)

    import time
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = student.predict(encoder(dummy_imgs), dummy_traj)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = student.predict(encoder(dummy_imgs), dummy_traj)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / n_runs * 1000.0   # ms per sample


# ── Main inference ─────────────────────────────────────────────────────────────

def infer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_index = build_image_index(args.nuscenes_root)

    log.info("Loading nuScenes metadata...")
    data       = pickle.load(open('./create_data/cached_nuscenes_info.pkl', 'rb'))
    split_info = json.load(open('./create_data/full_split.json'))
    tokens     = split_info[args.split]

    # Load checkpoint
    log.info(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg  = ckpt.get('args', {})

    encoder = LightEncoder(
        model_name=cfg.get('encoder_model', 'efficientvit_m4'),
        pretrained=False,
    ).to(device)
    encoder.load_state_dict(ckpt['encoder_state'])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    student = VCoTDStudentE2E(
        d_enc=encoder.d_enc,
        d_teacher=cfg.get('d_teacher', 1280),
        d=cfg.get('d', 256), K=cfg.get('K', 4),
        h=cfg.get('h', 4), d_ff=cfg.get('d_ff', 1024),
        T_obs=cfg.get('T_obs', 6), T_pred=cfg.get('T_pred', 6),
        vis_tokens=cfg.get('vis_tokens', 16),
        K_det=cfg.get('K_det', 20),
    ).to(device)
    student.load_state_dict(ckpt['student_state'])
    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)

    enc_p = sum(p.numel() for p in encoder.parameters()) / 1e6
    stu_p = sum(p.numel() for p in student.parameters()) / 1e6
    log.info(f"Encoder: {enc_p:.2f}M | Student: {stu_p:.2f}M | Total: {enc_p+stu_p:.2f}M params")

    # Latency benchmark
    if device.type == 'cuda':
        lat_ms = benchmark_latency(encoder, student, device)
        log.info(f"Inference latency: {lat_ms:.1f} ms/sample (GPU)")

    # Inference
    dataset = InferDataset(data, tokens, image_index, T_obs=cfg.get('T_obs', 6))
    loader  = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=infer_collate,
    )

    results = {}
    with torch.no_grad():
        for batch_tokens, hist_traj, images in tqdm(loader, desc='Inferring'):
            enc_feats = encoder(images.to(device))
            pred_traj = student.predict(enc_feats, hist_traj.to(device))
            for i, token in enumerate(batch_tokens):
                traj = pred_traj[i].cpu()
                results[token] = '[' + ', '.join(
                    f'({x:.2f},{y:.2f})' for x, y in traj.tolist()
                ) + ']'

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(output_path, 'w'), indent=2)
    log.info(f"Saved {len(results)} results → {output_path}")
    log.info("Evaluate with:")
    log.info(f"  python tools/evaluation/evaluation.py "
             f"--metric uniad --result_file {output_path} --method VCoTD-E2E")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',          type=str, required=True)
    p.add_argument('--nuscenes_root', type=str,
                   default='/media/flyingbird/07419DF8D71B0526/Dataset/nuScenes/nuscenes')
    p.add_argument('--split',         type=str, default='val', choices=['train', 'val'])
    p.add_argument('--output_path',   type=str, default='results_vcotd_e2e.json')
    p.add_argument('--batch_size',    type=int, default=32)
    return p.parse_args()


if __name__ == '__main__':
    infer(parse_args())
