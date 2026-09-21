"""
VCoTD End-to-End Training Script.

Trains LightEncoder (EfficientViT-M4) + VCoTDStudentE2E jointly.
At inference, only these two lightweight models are needed — no Qwen2-VL.

Inputs during training:
  - Raw 6-camera images (loaded from nuScenes, no pre-extraction at runtime)
  - Cached teacher features (Qwen2-VL, pre-extracted once) — distillation targets
  - nuScenes GT boxes — auxiliary BEV detection supervision
  - GT ego trajectory — main prediction target

Loss:
  L = L_pred + alpha * (lambda1*L_global + lambda2*L_spatial + lambda3*L_detail)
    + lambda_det * L_det

Usage:
    cd FSDrive-main
    python train_vcotd_e2e.py \
        --teacher_feat_dir /media/flyingbird/.../teacher_features \
        --nuscenes_root /media/flyingbird/.../nuscenes \
        --output_dir saves/vcotd_e2e
"""

import os
import glob
import json
import pickle
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms as T
from PIL import Image

from model.light_encoder import LightEncoder, CAMERA_TYPES
from model.vcotd_student_e2e import VCoTDStudentE2E

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


# ── Image path resolution ─────────────────────────────────────────────────────

def build_image_index(nuscenes_root: str) -> dict:
    """One-time scan: filename → full absolute path for all camera JPEGs."""
    log.info(f"Scanning images under {nuscenes_root} ...")
    index = {}
    for path in glob.glob(os.path.join(nuscenes_root, '*', 'samples', '*', '*.jpg')):
        index[os.path.basename(path)] = path
    log.info(f"Image index built: {len(index)} files")
    return index


# ── Dataset ───────────────────────────────────────────────────────────────────

class E2EDataset(Dataset):
    """
    Returns raw images + cached teacher features + trajectory + BEV GT boxes.
    """

    def __init__(
        self,
        split:            str,
        data:             dict,
        tokens:           list,
        teacher_feat_dir: str,
        image_index:      dict,
        T_obs:            int   = 6,
        T_pred:           int   = 6,
        K_det:            int   = 20,
        perception_range: float = 50.0,
    ):
        self.data             = data
        self.T_obs            = T_obs
        self.T_pred           = T_pred
        self.K_det            = K_det
        self.perception_range = perception_range
        self.teacher_feat_dir = Path(teacher_feat_dir) / split
        self.image_index      = image_index

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),           # → [0, 1]; normalisation done in LightEncoder
        ])

        self.valid_tokens = []
        missing_traj = missing_feat = missing_img = 0
        for token in tokens:
            if token not in data:
                missing_traj += 1
                continue
            d = data[token]
            if d.get('gt_ego_his_trajs') is None or d.get('gt_ego_fut_trajs') is None:
                missing_traj += 1
                continue
            if not (self.teacher_feat_dir / f"{token}.pt").exists():
                missing_feat += 1
                continue
            # Require at least the front camera to be accessible
            front_fn = os.path.basename(d['cams']['CAM_FRONT']['data_path'])
            if front_fn not in image_index:
                missing_img += 1
                continue
            self.valid_tokens.append(token)

        log.info(
            f"[E2E] Split '{split}': {len(self.valid_tokens)} valid samples "
            f"(missing traj={missing_traj}, feat={missing_feat}, img={missing_img})"
        )

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

    def _load_traj(self, d):
        his = torch.tensor(d['gt_ego_his_trajs'], dtype=torch.float32)
        if his.shape[0] >= self.T_obs:
            hist_traj = his[-self.T_obs:]
        else:
            pad = torch.zeros(self.T_obs - his.shape[0], 2)
            hist_traj = torch.cat([pad, his], dim=0)
        fut    = torch.tensor(d['gt_ego_fut_trajs'], dtype=torch.float32)
        gt_traj = fut[1:self.T_pred + 1]
        return hist_traj, gt_traj

    def _load_det_gt(self, d):
        """Top-K nearest objects in ego BEV, normalised to [-1, 1]."""
        K = self.K_det
        if 'gt_boxes' not in d or len(d['gt_boxes']) == 0:
            return torch.zeros(K, 2), torch.zeros(K, dtype=torch.bool)

        boxes_xy = torch.tensor(d['gt_boxes'][:, :2], dtype=torch.float32)
        dists    = torch.norm(boxes_xy, dim=1)
        valid    = dists < self.perception_range
        boxes_xy = boxes_xy[valid]
        dists    = dists[valid]

        if len(boxes_xy) == 0:
            return torch.zeros(K, 2), torch.zeros(K, dtype=torch.bool)

        idx      = torch.argsort(dists)
        boxes_xy = (boxes_xy[idx] / self.perception_range).clamp(-1, 1)

        n      = len(boxes_xy)
        det_gt = torch.zeros(K, 2)
        det_gt[:min(n, K)] = boxes_xy[:K]
        det_mask = torch.zeros(K, dtype=torch.bool)
        det_mask[:min(n, K)] = True
        return det_gt, det_mask

    def __getitem__(self, idx):
        token = self.valid_tokens[idx]
        d     = self.data[token]

        hist_traj, gt_traj = self._load_traj(d)

        images = torch.stack([
            self._load_image(d['cams'][cam]['data_path']) for cam in CAMERA_TYPES
        ])  # (6, 3, 224, 224)

        feats = torch.load(
            self.teacher_feat_dir / f"{token}.pt",
            map_location='cpu', weights_only=True,
        )

        det_gt, det_mask = self._load_det_gt(d)

        return {
            'token':        token,
            'images':       images,
            'hist_traj':    hist_traj,
            'gt_traj':      gt_traj,
            'feat_shallow': feats['shallow'].float(),
            'feat_middle':  feats['middle'].float(),
            'feat_deep':    feats['deep'].float(),
            'det_gt':       det_gt,
            'det_mask':     det_mask,
        }


def collate_fn(batch):
    keys_stack = [
        'images', 'hist_traj', 'gt_traj',
        'feat_shallow', 'feat_middle', 'feat_deep',
        'det_gt', 'det_mask',
    ]
    out = {k: torch.stack([b[k] for b in batch]) for k in keys_stack}
    out['tokens'] = [b['token'] for b in batch]
    return out


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(encoder, student, val_loader, device):
    encoder.eval()
    student.eval()
    l2_sum = torch.zeros(student.T_pred)
    n = 0

    for batch in val_loader:
        images    = batch['images'].to(device)
        hist_traj = batch['hist_traj'].to(device)
        gt_traj   = batch['gt_traj'].to(device)

        enc_feats = encoder(images)
        pred_traj = student.predict(enc_feats, hist_traj)

        l2 = torch.sqrt(((pred_traj - gt_traj) ** 2).sum(dim=-1)).cpu()
        l2_sum += l2.sum(dim=0)
        n += images.size(0)

    encoder.train()
    student.train()

    l2_mean = l2_sum / max(n, 1)
    l2_1s   = l2_mean[1].item()
    l2_2s   = l2_mean[3].item()
    l2_3s   = l2_mean[5].item()
    return {'l2_1s': l2_1s, 'l2_2s': l2_2s, 'l2_3s': l2_3s,
            'l2_avg': (l2_1s + l2_2s + l2_3s) / 3.0}


# ── Training ──────────────────────────────────────────────────────────────────

def save_checkpoint(path, encoder, student, optimizer, scheduler,
                    epoch, global_step, best_avg_l2, args):
    torch.save({
        'epoch':          epoch,
        'global_step':    global_step,
        'best_avg_l2':    best_avg_l2,
        'encoder_state':  encoder.state_dict(),
        'student_state':  student.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'args':           vars(args),
    }, path)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    image_index = build_image_index(args.nuscenes_root)

    log.info("Loading nuScenes metadata...")
    data       = pickle.load(open('./create_data/cached_nuscenes_info.pkl', 'rb'))
    split_info = json.load(open('./create_data/full_split.json'))

    train_ds = E2EDataset(
        split='train', data=data, tokens=split_info['train'],
        teacher_feat_dir=args.teacher_feat_dir, image_index=image_index,
        T_obs=args.T_obs, T_pred=args.T_pred,
        K_det=args.K_det, perception_range=args.perception_range,
    )
    val_ds = E2EDataset(
        split='val', data=data, tokens=split_info['val'],
        teacher_feat_dir=args.teacher_feat_dir, image_index=image_index,
        T_obs=args.T_obs, T_pred=args.T_pred,
        K_det=args.K_det, perception_range=args.perception_range,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=(args.num_workers > 0),
    )

    # Build models
    encoder = LightEncoder(
        model_name=args.encoder_model,
        pretrained=True,
        freeze=args.freeze_encoder,
    ).to(device)

    student = VCoTDStudentE2E(
        d_enc=encoder.d_enc, d_teacher=args.d_teacher,
        d=args.d, K=args.K, h=args.h, d_ff=args.d_ff,
        T_obs=args.T_obs, T_pred=args.T_pred,
        alpha_min=args.alpha_min, alpha_max=args.alpha_max,
        lambda1=args.lambda1, lambda2=args.lambda2, lambda3=args.lambda3,
        vis_tokens=args.vis_tokens,
        K_det=args.K_det, lambda_det=args.lambda_det,
    ).to(device)

    enc_p = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    stu_p = sum(p.numel() for p in student.parameters() if p.requires_grad)
    log.info(f"Encoder: {enc_p/1e6:.2f}M params | Student: {stu_p/1e6:.2f}M params | "
             f"Total: {(enc_p+stu_p)/1e6:.2f}M")
    log.info(f"Encoder output: {encoder.total_tokens} tokens × {encoder.d_enc} dims")

    all_params = list(encoder.parameters()) + list(student.parameters())
    optimizer  = AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=args.lr * 0.01,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Checkpoints → {output_dir}")

    # Resume
    start_epoch = 0
    global_step = 0
    best_avg_l2 = float('inf')
    latest = output_dir / 'checkpoint_latest.pt'
    if args.resume and latest.exists():
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt['encoder_state'])
        student.load_state_dict(ckpt['student_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step']
        best_avg_l2 = ckpt.get('best_avg_l2', float('inf'))
        log.info(f"Resumed from epoch {ckpt['epoch']}, step {global_step}")

    for epoch in range(start_epoch, args.epochs):
        encoder.train()
        student.train()
        epoch_losses = {k: 0.0 for k in
                        ['total', 'pred', 'distill', 'det',
                         'L_global', 'L_spatial', 'L_detail']}
        n_batches = 0

        for batch in train_loader:
            images    = batch['images'].to(device)
            hist_traj = batch['hist_traj'].to(device)
            gt_traj   = batch['gt_traj'].to(device)
            det_gt    = batch['det_gt'].to(device)
            det_mask  = batch['det_mask'].to(device)
            teacher_features = {
                'shallow': batch['feat_shallow'].to(device),
                'middle':  batch['feat_middle'].to(device),
                'deep':    batch['feat_deep'].to(device),
            }

            enc_feats = encoder(images)
            pred_traj, student_features, gamma, det_pred = student(enc_feats, hist_traj)

            loss, losses = student.compute_total_loss(
                pred_traj, gt_traj, student_features, teacher_features, gamma,
                det_pred=det_pred, det_gt=det_gt, det_mask=det_mask,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            if global_step < warmup_steps:
                scale = (global_step + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg['lr'] = args.lr * scale
            else:
                scheduler.step()

            global_step += 1
            n_batches   += 1
            for k in epoch_losses:
                epoch_losses[k] += losses.get(k, 0.0)

            if global_step % args.log_interval == 0:
                avg = {k: v / n_batches for k, v in epoch_losses.items()}
                lr  = optimizer.param_groups[0]['lr']
                log.info(
                    f"[Ep {epoch}/{args.epochs} Step {global_step}] "
                    f"total={avg['total']:.4f} pred={avg['pred']:.4f} "
                    f"distill={avg['distill']:.4f} det={avg['det']:.4f} "
                    f"(g={avg['L_global']:.4f} sp={avg['L_spatial']:.4f} "
                    f"dt={avg['L_detail']:.4f}) lr={lr:.2e}"
                )

        # Save checkpoint
        ckpt_path = output_dir / 'checkpoint_latest.pt'
        save_checkpoint(ckpt_path, encoder, student, optimizer, scheduler,
                        epoch, global_step, best_avg_l2, args)
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(output_dir / f'checkpoint_epoch{epoch}.pt',
                            encoder, student, optimizer, scheduler,
                            epoch, global_step, best_avg_l2, args)

        # Validate
        metrics = evaluate(encoder, student, val_loader, device)
        log.info(
            f"[Ep {epoch}] Val L2 — 1s: {metrics['l2_1s']:.4f} m | "
            f"2s: {metrics['l2_2s']:.4f} m | 3s: {metrics['l2_3s']:.4f} m | "
            f"avg: {metrics['l2_avg']:.4f} m"
        )
        if metrics['l2_avg'] < best_avg_l2:
            best_avg_l2 = metrics['l2_avg']
            save_checkpoint(output_dir / 'checkpoint_best.pt',
                            encoder, student, optimizer, scheduler,
                            epoch, global_step, best_avg_l2, args)
            log.info(f"  -> New best: {best_avg_l2:.4f} m")

    log.info(f"Training complete. Best Val L2 avg: {best_avg_l2:.4f} m")


def parse_args():
    p = argparse.ArgumentParser(description='Train VCoTD E2E (LightEncoder + Student)')

    # Data paths
    p.add_argument('--teacher_feat_dir', type=str,
                   default='/media/flyingbird/07419DF8D71B0526/Dataset/nuScenes/teacher_features',
                   help='Root of cached teacher features (train/ and val/ subdirs)')
    p.add_argument('--nuscenes_root', type=str,
                   default='/media/flyingbird/07419DF8D71B0526/Dataset/nuScenes/nuscenes',
                   help='nuScenes dataset root (contains v1.0-trainvalXX_keyframes/)')

    # Encoder
    p.add_argument('--encoder_model',  type=str, default='efficientvit_m4',
                   help='timm model name for lightweight encoder')
    p.add_argument('--freeze_encoder', action='store_true',
                   help='Freeze encoder backbone (useful for debugging)')

    # Student model
    p.add_argument('--d_teacher',  type=int,   default=1280,
                   help='Teacher feature dim (Qwen2-VL visual encoder: 1280)')
    p.add_argument('--d',          type=int,   default=256)
    p.add_argument('--K',          type=int,   default=4)
    p.add_argument('--h',          type=int,   default=4)
    p.add_argument('--d_ff',       type=int,   default=1024)
    p.add_argument('--T_obs',      type=int,   default=6)
    p.add_argument('--T_pred',     type=int,   default=6)
    p.add_argument('--vis_tokens', type=int,   default=16)

    # Distillation
    p.add_argument('--alpha_min',  type=float, default=0.3)
    p.add_argument('--alpha_max',  type=float, default=1.0)
    p.add_argument('--lambda1',    type=float, default=1.0,
                   help='Structural distillation weight (shallow)')
    p.add_argument('--lambda2',    type=float, default=1.0,
                   help='Relational distillation weight (middle)')
    p.add_argument('--lambda3',    type=float, default=0.5,
                   help='Semantic distillation weight (deep)')

    # Auxiliary detection
    p.add_argument('--K_det',            type=int,   default=20,
                   help='Number of BEV detection slots')
    p.add_argument('--lambda_det',       type=float, default=0.5,
                   help='BEV detection auxiliary loss weight')
    p.add_argument('--perception_range', type=float, default=50.0,
                   help='Ego-frame perception range for detection GT (metres)')

    # Training
    p.add_argument('--epochs',       type=int,   default=12)
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--warmup_ratio', type=float, default=0.1)
    p.add_argument('--num_workers',  type=int,   default=4)
    p.add_argument('--log_interval', type=int,   default=50)
    p.add_argument('--save_every',   type=int,   default=3)
    p.add_argument('--output_dir',   type=str,   default='saves/vcotd_e2e')
    p.add_argument('--resume',       action='store_true')

    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
