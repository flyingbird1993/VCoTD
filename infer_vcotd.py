"""
VCoTD Inference Script.

Inference pipeline:
    1. Load frozen Qwen2-VL visual encoder (teacher, parameter frozen)
    2. For each sample: run ONE forward pass through encoder to get deep features
    3. Feed features + historical trajectory into lightweight student model
    4. Output predicted trajectory in the same format as FSDrive for evaluation

Output format matches FSDrive's eval_traj.json:
    {
        "<sample_token>": "[(x1,y1), (x2,y2), ..., (x6,y6)]",
        ...
    }

Usage:
    cd FSDrive-main
    python infer_vcotd.py \
        --student_ckpt saves/vcotd/checkpoint_best.pt \
        --teacher_model_path saves/qwen2_vl-2b/pretrain \
        --split val \
        --output_path results_vcotd.json
"""

import os
import re
import json
import pickle
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from model.vcotd_student import build_student_from_checkpoint
from create_data.extract_teacher_features import (
    FeatureHook, load_and_resize_image,
    CAMERA_TYPES, LAYER_SHALLOW, LAYER_MIDDLE, LAYER_DEEP,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


# ── Inference dataset ──────────────────────────────────────────────────────────

class VCoTDInferDataset(Dataset):
    """Loads trajectory data for inference (no pre-cached features needed)."""

    def __init__(self, data, tokens, T_obs=6):
        self.data = data
        self.T_obs = T_obs
        self.valid_tokens = [t for t in tokens if t in data]
        log.info(f"Inference dataset: {len(self.valid_tokens)} samples")

    def __len__(self):
        return len(self.valid_tokens)

    def __getitem__(self, idx):
        token = self.valid_tokens[idx]
        d = self.data[token]

        his_trajs = torch.tensor(d['gt_ego_his_trajs'], dtype=torch.float32)
        if his_trajs.shape[0] >= self.T_obs:
            hist_traj = his_trajs[-self.T_obs:]
        else:
            pad = torch.zeros(self.T_obs - his_trajs.shape[0], 2)
            hist_traj = torch.cat([pad, his_trajs], dim=0)

        # Image paths for all 6 cameras
        image_paths = []
        for cam in CAMERA_TYPES:
            img_path = d['cams'][cam]['data_path']
            image_paths.append(img_path)

        return token, hist_traj, image_paths


# ── Trajectory formatting ─────────────────────────────────────────────────────

def format_trajectory(traj: torch.Tensor) -> str:
    """Convert (T_pred, 2) tensor to FSDrive-compatible string format."""
    coords = [f"({x:.2f},{y:.2f})" for x, y in traj.tolist()]
    return '[' + ', '.join(coords) + ']'


# ── Main inference ─────────────────────────────────────────────────────────────

def infer(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ── Load metadata ─────────────────────────────────────────────────────
    log.info("Loading nuScenes metadata...")
    data = pickle.load(open('./create_data/cached_nuscenes_info.pkl', 'rb'))
    split_info = json.load(open('./create_data/full_split.json', 'r'))
    tokens = split_info[args.split]

    # ── Load student model ────────────────────────────────────────────────
    log.info(f"Loading student model from {args.student_ckpt}")
    ckpt = torch.load(args.student_ckpt, map_location=device)

    student = build_student_from_checkpoint(ckpt).to(device)
    student.eval()
    log.info("Student model loaded.")

    # ── Inference loop ────────────────────────────────────────────────────
    results = {}

    if args.mode == 'gt_only':
        log.info("Mode: gt_only — skipping teacher, using zero visual features")

        for token in tqdm(tokens, desc='Inferring'):
            if token not in data:
                continue

            d = data[token]
            his_trajs = torch.tensor(d['gt_ego_his_trajs'], dtype=torch.float32)
            T_obs = student.T_obs
            if his_trajs.shape[0] >= T_obs:
                hist_traj = his_trajs[-T_obs:]
            else:
                pad = torch.zeros(T_obs - his_trajs.shape[0], 2)
                hist_traj = torch.cat([pad, his_trajs], dim=0)
            hist_traj = hist_traj.unsqueeze(0).to(device)

            feat_deep = torch.zeros(
                1, student.vis_tokens, student.d_vlm, device=device
            )
            with torch.no_grad():
                pred_traj = student.predict(feat_deep, hist_traj)
            results[token] = format_trajectory(pred_traj[0].cpu())

    else:
        # ── Load teacher encoder ──────────────────────────────────────────
        log.info(f"Loading Qwen2-VL encoder from {args.teacher_model_path}")
        dtype_map = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}
        torch_dtype = dtype_map.get(args.dtype, torch.bfloat16)

        teacher = Qwen2VLForConditionalGeneration.from_pretrained(
            args.teacher_model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        processor = AutoProcessor.from_pretrained(args.teacher_model_path, trust_remote_code=True)

        visual_encoder = teacher.visual
        num_blocks = len(visual_encoder.blocks)
        L_d = min(LAYER_DEEP, num_blocks - 1)
        deep_hook = FeatureHook()
        deep_hook.register(visual_encoder.blocks[L_d])
        log.info(f"Teacher encoder: {num_blocks} blocks, hooking at layer {L_d} (deep)")

        nuscenes_root = args.nuscenes_root

        for token in tqdm(tokens, desc='Inferring'):
            if token not in data:
                continue

            d = data[token]

            images = []
            ok = True
            for cam in CAMERA_TYPES:
                img_path = d['cams'][cam]['data_path']
                img_path = img_path.replace('/localdata_ssd/nuScenes', nuscenes_root, 1)
                if not os.path.exists(img_path):
                    ok = False
                    break
                images.append(load_and_resize_image(img_path, args.image_size))
            if not ok:
                continue

            his_trajs = torch.tensor(d['gt_ego_his_trajs'], dtype=torch.float32)
            T_obs = student.T_obs
            if his_trajs.shape[0] >= T_obs:
                hist_traj = his_trajs[-T_obs:]
            else:
                pad = torch.zeros(T_obs - his_trajs.shape[0], 2)
                hist_traj = torch.cat([pad, his_trajs], dim=0)
            hist_traj = hist_traj.unsqueeze(0).to(device)

            messages = [{
                'role': 'user',
                'content': [
                    *[{'type': 'image', 'image': img} for img in images],
                    {'type': 'text', 'text': 'Describe the scene.'},
                ],
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            inputs = processor(text=[text], images=images, return_tensors='pt', padding=True)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

            deep_hook.value = None
            with torch.no_grad():
                try:
                    teacher(**inputs)
                except Exception:
                    del inputs
                    torch.cuda.empty_cache()
                    continue

            del inputs
            torch.cuda.empty_cache()

            if deep_hook.value is None:
                continue

            feat_deep = deep_hook.value.to(device).unsqueeze(0)
            deep_hook.value = None
            if torch_dtype != torch.float32:
                feat_deep = feat_deep.float()

            with torch.no_grad():
                pred_traj = student.predict(feat_deep, hist_traj)

            results[token] = format_trajectory(pred_traj[0].cpu())
            del feat_deep, hist_traj, pred_traj
            torch.cuda.empty_cache()

        deep_hook.remove()

    # ── Save results ──────────────────────────────────────────────────────
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    log.info(f"Results saved to {output_path} ({len(results)} samples)")
    log.info("Run evaluation with:")
    log.info(f"  python tools/evaluation/evaluation.py --metric uniad "
             f"--result_file {output_path} --method VCoTD")


def parse_args():
    p = argparse.ArgumentParser(description='VCoTD inference')
    p.add_argument('--student_ckpt',       type=str, required=True)
    p.add_argument('--mode',               type=str, default='gt_only', choices=['gt_only', 'distill'])
    p.add_argument('--teacher_model_path', type=str, default='model/FSDrive_pretrain')
    p.add_argument('--split',              type=str, default='val', choices=['train', 'val'])
    p.add_argument('--output_path',        type=str, default='./results_vcotd.json')
    p.add_argument('--nuscenes_root',      type=str, default='./LLaMA-Factory/data/nuscenes')
    p.add_argument('--image_size',         type=int, default=224)
    p.add_argument('--device',             type=str, default='cuda')
    p.add_argument('--dtype',              type=str, default='bf16')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    infer(args)
