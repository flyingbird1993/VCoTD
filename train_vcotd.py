"""
VCoTD Training Script.

Three training modes (controlled by --mode):

  gt_only     - Baseline A: train student with GT waypoints only, no distillation.
                Requires only trajectory data, no teacher features.
                Use this to establish the baseline performance.

  distill     - Full VCoTD: GT supervision + three-level feature distillation.
                Requires pre-cached teacher features.
                Phase 1 (offline, run once):
                    python create_data/extract_teacher_features.py --split train
                    python create_data/extract_teacher_features.py --split val

    distill_kd  - Full VCoTD + Response-based KD: GT + feature distillation +
                teacher waypoint soft labels.
                This is a code extension and is not part of the VCoTD paper.
                Requires teacher features AND teacher_waypoints_{split}.pkl.

Usage examples:
    # Baseline A (no distillation, runs immediately):
    python train_vcotd.py --mode gt_only --output_dir ./saves/vcotd_baseline

    # Full distillation (after extracting teacher features):
    python train_vcotd.py --mode distill --teacher_feat_dir ./teacher_features \
        --output_dir ./saves/vcotd

    # Full distillation + response-based KD:
    python train_vcotd.py --mode distill_kd --teacher_feat_dir ./teacher_features \
        --teacher_wp_dir ./teacher_waypoints --output_dir ./saves/vcotd_kd
"""

import os
import json
import pickle
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.vcotd_student import VCoTDStudentModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger(__name__)


def is_dist():
    return dist.is_available() and dist.is_initialized()

def rank():
    return dist.get_rank() if is_dist() else 0

def world_size():
    return dist.get_world_size() if is_dist() else 1


# ── Dataset ───────────────────────────────────────────────────────────────────

class VCoTDDataset(Dataset):
    """
    Unified dataset for all three training modes.

    gt_only mode:   loads trajectory data only (no teacher features required).
    distill mode:   loads teacher features + trajectory data.
    distill_kd mode: loads teacher features + trajectory data + teacher waypoints.

    Each item returns a dict with keys:
        hist_traj    : (T_obs, 2)   historical waypoints
        gt_traj      : (T_pred, 2)  ground-truth future waypoints
        token        : str          sample token
      [distill/distill_kd only]
        feat_shallow : (N, d_vlm)   teacher shallow features
        feat_middle  : (N, d_vlm)   teacher middle features
        feat_deep    : (N, d_vlm)   teacher deep features
      [distill_kd only]
        teacher_wp   : (T_pred, 2)  teacher predicted waypoints (soft labels)
    """

    def __init__(
        self,
        split: str,
        data: dict,
        tokens: list,
        mode: str = 'distill',
        teacher_feat_dir: str = None,
        teacher_wp_dir: str = None,
        T_obs: int = 6,
        T_pred: int = 6,
    ):
        self.data = data
        self.mode = mode
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.teacher_feat_dir = Path(teacher_feat_dir) / split if teacher_feat_dir else None
        self.teacher_wp_dir   = Path(teacher_wp_dir) / split  if teacher_wp_dir  else None
        if mode in ('distill', 'distill_kd') and self.teacher_feat_dir is None:
            raise ValueError(f"{mode} requires teacher_feat_dir")
        if mode == 'distill_kd' and self.teacher_wp_dir is None:
            raise ValueError("distill_kd requires teacher_wp_dir")

        # Keep only tokens with trajectory data (and features if needed)
        self.valid_tokens = []
        missing_feat = 0
        missing_traj = 0
        teacher_wp_available = 0
        for token in tokens:
            if token not in data:
                missing_traj += 1
                continue
            d = data[token]
            if (d.get('gt_ego_his_trajs') is None or
                    d.get('gt_ego_fut_trajs') is None):
                missing_traj += 1
                continue
            if mode in ('distill', 'distill_kd'):
                feat_path = self.teacher_feat_dir / f"{token}.pt"
                if not feat_path.exists():
                    missing_feat += 1
                    continue
            self.valid_tokens.append(token)
            if mode == 'distill_kd' and (
                self.teacher_wp_dir / f"{token}.pt"
            ).exists():
                teacher_wp_available += 1

        log.info(f"[{mode}] Split '{split}': {len(self.valid_tokens)} valid samples "
                 f"(missing traj: {missing_traj}, missing feat: {missing_feat})")
        if mode == 'distill_kd':
            log.info(
                f"[{mode}] Split '{split}': teacher waypoint coverage "
                f"{teacher_wp_available}/{len(self.valid_tokens)}"
            )

    def __len__(self):
        return len(self.valid_tokens)

    def _load_traj(self, d):
        """Load and pad historical + future trajectory from nuScenes cache."""
        his_trajs = torch.tensor(d['gt_ego_his_trajs'], dtype=torch.float32)
        if his_trajs.shape[0] >= self.T_obs:
            hist_traj = his_trajs[-self.T_obs:]
        else:
            pad = torch.zeros(self.T_obs - his_trajs.shape[0], 2)
            hist_traj = torch.cat([pad, his_trajs], dim=0)

        fut_trajs = torch.tensor(d['gt_ego_fut_trajs'], dtype=torch.float32)
        gt_traj = fut_trajs[1:self.T_pred + 1]   # cache index 0 is current pose
        raw_mask = d.get('gt_ego_fut_masks')
        if raw_mask is None:
            gt_mask = torch.ones(gt_traj.shape[0], dtype=torch.bool)
        else:
            raw_mask = torch.as_tensor(raw_mask, dtype=torch.bool)
            # The local cache stores six future-mask entries for seven poses.
            # Also accept caches whose mask includes the current pose.
            if raw_mask.shape[0] == fut_trajs.shape[0]:
                raw_mask = raw_mask[1:self.T_pred + 1]
            else:
                raw_mask = raw_mask[:self.T_pred]
            gt_mask = raw_mask.all(dim=-1) if raw_mask.ndim == 2 else raw_mask
        if gt_traj.shape[0] < self.T_pred:
            missing = self.T_pred - gt_traj.shape[0]
            gt_traj = torch.cat([gt_traj, torch.zeros(missing, 2)], dim=0)
            gt_mask = torch.cat([gt_mask, torch.zeros(missing, dtype=torch.bool)])
        if gt_mask.shape[0] < self.T_pred:
            gt_mask = torch.cat([
                gt_mask,
                torch.zeros(self.T_pred - gt_mask.shape[0], dtype=torch.bool),
            ])
        gt_traj = gt_traj[:self.T_pred]
        gt_mask = gt_mask[:self.T_pred]
        return hist_traj, gt_traj, gt_mask

    def __getitem__(self, idx):
        token = self.valid_tokens[idx]
        d = self.data[token]

        hist_traj, gt_traj, gt_mask = self._load_traj(d)
        item = {
            'hist_traj': hist_traj,
            'gt_traj': gt_traj,
            'gt_mask': gt_mask,
            'token': token,
        }

        if self.mode in ('distill', 'distill_kd'):
            feats = torch.load(
                self.teacher_feat_dir / f"{token}.pt",
                map_location='cpu', weights_only=True,
            )
            item['feat_shallow'] = feats['shallow'].float()
            item['feat_middle']  = feats['middle'].float()
            item['feat_deep']    = feats['deep'].float()

        if self.mode == 'distill_kd' and self.teacher_wp_dir is not None:
            wp_path = self.teacher_wp_dir / f"{token}.pt"
            if wp_path.exists():
                item['teacher_wp'] = torch.load(
                    wp_path, map_location='cpu', weights_only=True
                ).float()   # (T_pred, 2)
                item['teacher_wp_valid'] = True
            else:
                item['teacher_wp'] = gt_traj.clone()   # fallback to GT
                item['teacher_wp_valid'] = False

        return item


def collate_fn(batch):
    """
    Custom collate: teacher features may have different N across samples.
    Pads feature sequences to the max N in the batch.
    Works for all three modes (gt_only / distill / distill_kd).
    """
    out = {
        'hist_traj': torch.stack([item['hist_traj'] for item in batch]),
        'gt_traj':   torch.stack([item['gt_traj']   for item in batch]),
        'gt_mask':   torch.stack([item['gt_mask']   for item in batch]),
        'tokens':    [item['token'] for item in batch],
    }

    if 'feat_deep' in batch[0]:
        max_N = max(item['feat_deep'].shape[0] for item in batch)

        def pad_feat(f):
            pad_len = max_N - f.shape[0]
            if pad_len == 0:
                return f
            return torch.cat([f, torch.zeros(pad_len, f.shape[1])], dim=0)

        out['feat_shallow'] = torch.stack([pad_feat(item['feat_shallow']) for item in batch])
        out['feat_middle']  = torch.stack([pad_feat(item['feat_middle'])  for item in batch])
        out['feat_deep']    = torch.stack([pad_feat(item['feat_deep'])    for item in batch])
        out['feat_mask'] = torch.stack([
            torch.arange(max_N) < item['feat_deep'].shape[0] for item in batch
        ])

    if 'teacher_wp' in batch[0]:
        out['teacher_wp'] = torch.stack([item['teacher_wp'] for item in batch])
        out['teacher_wp_valid'] = torch.tensor([
            item['teacher_wp_valid'] for item in batch
        ])

    return out


# ── Training utilities ────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    step,
    output_dir,
    args,
    tag='latest',
    best_avg_l2=float('inf'),
):
    raw_model = model.module if isinstance(model, DDP) else model
    state = raw_model.state_dict()
    ckpt = {
        'epoch': epoch,
        'step': step,
        'best_avg_l2': best_avg_l2,
        'model_state': state,
        'model_config': raw_model.model_config(),
        'args': vars(args),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
    }
    path = Path(output_dir) / f"checkpoint_{tag}.pt"
    torch.save(ckpt, path)
    return path


# ── Main training loop ────────────────────────────────────────────────────────

def train(args):
    if args.T_pred < 6:
        raise ValueError("Evaluation at 1/2/3 seconds requires T_pred >= 6")
    if not 0.0 <= args.kd_beta <= 1.0:
        raise ValueError("kd_beta must be in [0, 1]")
    if args.fixed_alpha < 0.0:
        raise ValueError("fixed_alpha must be non-negative")
    if args.mode == 'distill_kd' and args.teacher_wp_dir is None:
        raise ValueError("distill_kd requires --teacher_wp_dir")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # ── Distributed setup ─────────────────────────────────────────────────
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if 'LOCAL_RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if rank() == 0:
        log.info(f"Device: {device}, world_size: {world_size()}")

    # ── Load dataset metadata ─────────────────────────────────────────────
    if rank() == 0:
        log.info("Loading nuScenes metadata...")
    data = pickle.load(open('./create_data/cached_nuscenes_info.pkl', 'rb'))
    split_info = json.load(open('./create_data/full_split.json', 'r'))

    train_tokens = split_info['train']
    if getattr(args, 'max_train_samples', None) is not None:
        train_tokens = train_tokens[:args.max_train_samples]

    train_dataset = VCoTDDataset(
        split='train',
        data=data,
        tokens=train_tokens,
        mode=args.mode,
        teacher_feat_dir=args.teacher_feat_dir,
        teacher_wp_dir=args.teacher_wp_dir,
        T_obs=args.T_obs,
        T_pred=args.T_pred,
    )
    val_tokens = split_info['val']
    if getattr(args, 'max_val_samples', None) is not None:
        val_tokens = val_tokens[:args.max_val_samples]

    val_dataset = VCoTDDataset(
        split='val',
        data=data,
        tokens=val_tokens,
        mode=args.mode,
        teacher_feat_dir=args.teacher_feat_dir,
        teacher_wp_dir=args.teacher_wp_dir,
        T_obs=args.T_obs,
        T_pred=args.T_pred,
    )

    if len(train_dataset) == 0:
        raise RuntimeError("No valid training samples were found")
    if len(val_dataset) == 0:
        raise RuntimeError("No valid validation samples were found")

    if args.d_vlm <= 0:
        args.d_vlm = (
            int(train_dataset[0]['feat_deep'].shape[-1])
            if args.mode != 'gt_only'
            else 1280
        )
    elif args.mode != 'gt_only':
        actual_dim = int(train_dataset[0]['feat_deep'].shape[-1])
        if actual_dim != args.d_vlm:
            raise ValueError(
                f"Configured d_vlm={args.d_vlm}, but cached features use {actual_dim}"
            )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_dist() else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ── Build student model ───────────────────────────────────────────────
    model = VCoTDStudentModel(
        d_vlm=args.d_vlm,
        d=args.d,
        K=args.K,
        h=args.h,
        d_ff=args.d_ff,
        T_obs=args.T_obs,
        T_pred=args.T_pred,
        d_h=args.d_h,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        visual_aggregation=args.visual_aggregation,
        vis_tokens=args.vis_tokens,
    ).to(device)

    if is_dist():
        has_unused_parameters = args.mode == 'gt_only' or args.disable_adaptive
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=has_unused_parameters,
        )

    raw_model = model.module if isinstance(model, DDP) else model
    n_params = count_parameters(raw_model)
    if rank() == 0:
        log.info(f"Student model parameters: {n_params / 1e6:.2f}M")

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps - warmup_steps, 1),
        eta_min=args.lr * 0.01,
    )

    # ── Output directory ──────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if rank() == 0:
        log.info(f"Checkpoints will be saved to: {output_dir}")

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    best_avg_l2 = float('inf')
    if args.resume and (output_dir / 'checkpoint_latest.pt').exists():
        ckpt = torch.load(output_dir / 'checkpoint_latest.pt', map_location=device)
        raw_model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['step']
        best_avg_l2 = ckpt.get('best_avg_l2', float('inf'))
        if rank() == 0:
            log.info(f"Resumed from epoch {ckpt['epoch']}, step {global_step}")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_losses = {
            'total': 0., 'pred': 0., 'distill': 0.,
            'L_global': 0., 'L_spatial': 0., 'L_detail': 0.,
            'alpha': 0., 'alpha_min_batch': 0., 'alpha_max_batch': 0.,
        }
        n_batches = 0

        for batch in train_loader:
            hist_traj = batch['hist_traj'].to(device)
            gt_traj   = batch['gt_traj'].to(device)
            gt_mask   = batch['gt_mask'].to(device)

            if global_step < warmup_steps:
                lr_scale = (global_step + 1) / max(warmup_steps, 1)
                for pg in optimizer.param_groups:
                    pg['lr'] = args.lr * lr_scale

            if args.mode == 'gt_only':
                # ── Baseline A: GT supervision only ──────────────────────
                # Use zeros as dummy visual features; student learns from GT alone
                B = hist_traj.size(0)
                dummy_feat = torch.zeros(B, args.vis_tokens, args.d_vlm, device=device)
                pred_traj, student_features, gamma = model(dummy_feat, hist_traj)
                loss = raw_model.trajectory_loss_per_sample(
                    pred_traj, gt_traj, gt_mask
                ).mean()
                losses = {'total': loss.item(), 'pred': loss.item(),
                          'distill': 0., 'L_global': 0., 'L_spatial': 0., 'L_detail': 0.,
                          'alpha': 0., 'alpha_min_batch': 0., 'alpha_max_batch': 0.}

            else:
                # ── Distillation modes ────────────────────────────────────
                feat_shallow = batch['feat_shallow'].to(device)
                feat_middle  = batch['feat_middle'].to(device)
                feat_deep    = batch['feat_deep'].to(device)
                feat_mask    = batch['feat_mask'].to(device)

                teacher_features = {
                    'shallow': feat_shallow,
                    'middle':  feat_middle,
                    'deep':    feat_deep,
                }

                pred_traj, student_features, gamma = model(
                    feat_deep, hist_traj, feat_mask
                )

                if args.mode == 'distill_kd' and 'teacher_wp' in batch:
                    # Response KD extension: blend Euclidean GT and teacher losses.
                    teacher_wp      = batch['teacher_wp'].to(device)
                    L_gt = raw_model.trajectory_loss_per_sample(
                        pred_traj, gt_traj, gt_mask
                    )
                    L_kd = raw_model.trajectory_loss_per_sample(
                        pred_traj, teacher_wp, gt_mask
                    )
                    pred_per_sample = (1 - args.kd_beta) * L_gt + args.kd_beta * L_kd
                    terms = raw_model.compute_distillation_terms(
                        student_features, teacher_features, feat_mask
                    )
                    distill_per_sample = (
                        raw_model.lambda1 * terms['L_global']
                        + raw_model.lambda2 * terms['L_spatial']
                        + raw_model.lambda3 * terms['L_detail']
                    )
                    if args.disable_adaptive:
                        alpha = torch.full_like(pred_per_sample, args.fixed_alpha)
                    else:
                        alpha = raw_model.alpha_min + (
                            raw_model.alpha_max - raw_model.alpha_min
                        ) * gamma.squeeze(-1)
                    loss = (pred_per_sample + alpha * distill_per_sample).mean()
                    losses = {
                        'total':    loss.item(),
                        'pred':     pred_per_sample.mean().item(),
                        'L_gt':     L_gt.mean().item(),
                        'L_kd':     L_kd.mean().item(),
                        'distill':  distill_per_sample.mean().item(),
                        'alpha': alpha.mean().item(),
                        'alpha_min_batch': alpha.min().item(),
                        'alpha_max_batch': alpha.max().item(),
                        **{name: value.mean().item() for name, value in terms.items()},
                    }
                else:
                    loss, losses = raw_model.compute_total_loss(
                        pred_traj,
                        gt_traj,
                        student_features,
                        teacher_features,
                        gamma,
                        teacher_mask=feat_mask,
                        trajectory_mask=gt_mask,
                        adaptive=not args.disable_adaptive,
                        fixed_alpha=args.fixed_alpha,
                    )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if global_step >= warmup_steps:
                scheduler.step()

            global_step += 1
            n_batches += 1
            for k in epoch_losses:
                epoch_losses[k] += losses.get(k, 0.)

            if global_step % args.log_interval == 0 and rank() == 0:
                avg = {k: v / n_batches for k, v in epoch_losses.items()}
                lr = optimizer.param_groups[0]['lr']
                log.info(
                    f"[Epoch {epoch}/{args.epochs} Step {global_step}] "
                    f"total={avg['total']:.4f} pred={avg['pred']:.4f} "
                    f"distill={avg['distill']:.4f} "
                    f"(global={avg['L_global']:.4f} spatial={avg['L_spatial']:.4f} "
                    f"detail={avg['L_detail']:.4f}) alpha={avg['alpha']:.4f} lr={lr:.2e}"
                )

        # ── Epoch-end checkpoint (rank 0 only) ───────────────────────────
        if rank() == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                output_dir, args, 'latest', best_avg_l2,
            )
            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                                output_dir, args, f'epoch_{epoch}', best_avg_l2)

        # ── Validation (rank 0 only) ──────────────────────────────────────
        if rank() == 0:
            metrics = evaluate(model, val_loader, device, args.T_pred, mode=args.mode)
            log.info(
                f"[Epoch {epoch}] Val L2 — "
                f"1s: {metrics['l2_1s']:.4f} m | "
                f"2s: {metrics['l2_2s']:.4f} m | "
                f"3s: {metrics['l2_3s']:.4f} m | "
                f"avg: {metrics['l2_avg']:.4f} m"
            )
            avg_l2 = metrics['l2_avg']
            if avg_l2 < best_avg_l2:
                best_avg_l2 = avg_l2
                save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                                output_dir, args, 'best', best_avg_l2)
                save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                                output_dir, args, 'latest', best_avg_l2)
                log.info(f"  -> New best model saved (avg L2 = {best_avg_l2:.4f} m)")

        if is_dist():
            dist.barrier()

    if rank() == 0:
        log.info(f"Training complete. Best Val L2 avg(1s/2s/3s): {best_avg_l2:.4f} m")
    if is_dist():
        dist.destroy_process_group()


@torch.no_grad()
def evaluate(model, val_loader, device, T_pred=6, mode='distill'):
    """
    Compute L2 displacement error over validation set.

    Follows the standard nuScenes evaluation protocol:
      - 6 future waypoints at 0.5 s intervals → indices 0..5
      - Report L2 at 1 s (idx 1), 2 s (idx 3), 3 s (idx 5)
      - Also report the average of those three horizons for checkpoint selection

    Returns:
        dict with keys: 'l2_1s', 'l2_2s', 'l2_3s', 'l2_avg'
    """
    model.eval()
    raw_model = model.module if isinstance(model, DDP) else model
    l2_sum = torch.zeros(T_pred)
    l2_valid_sum = torch.zeros(T_pred)
    valid_count = torch.zeros(T_pred)
    n_samples = 0

    for batch in val_loader:
        hist_traj = batch['hist_traj'].to(device)
        gt_traj   = batch['gt_traj'].to(device)
        gt_mask   = batch['gt_mask'].bool()
        B         = hist_traj.size(0)

        if mode == 'gt_only':
            feat_deep = torch.zeros(B, raw_model.vis_tokens, raw_model.d_vlm, device=device)
        else:
            feat_deep = batch['feat_deep'].to(device)
            feat_mask = batch['feat_mask'].to(device)

        pred_traj = raw_model.predict(
            feat_deep,
            hist_traj,
            feature_mask=feat_mask if mode != 'gt_only' else None,
        )   # (B, T_pred, 2)

        # L2 per sample per timestep: (B, T_pred)
        l2 = torch.sqrt(((pred_traj - gt_traj) ** 2).sum(dim=-1)).cpu()
        masked_l2 = l2 * gt_mask
        l2_sum += masked_l2.sum(dim=0)
        l2_valid_sum += masked_l2.sum(dim=0)
        valid_count += gt_mask.sum(dim=0)
        n_samples += B

    model.train()

    l2_mean = l2_sum / max(n_samples, 1)   # mean per timestep
    l2_valid_mean = l2_valid_sum / valid_count.clamp_min(1)
    # Standard horizons: 1s=idx1, 2s=idx3, 3s=idx5  (0.5s step)
    l2_1s  = l2_mean[1].item()
    l2_2s  = l2_mean[3].item()
    l2_3s  = l2_mean[5].item()
    l2_avg = (l2_1s + l2_2s + l2_3s) / 3.0
    return {
        'l2_1s': l2_1s,
        'l2_2s': l2_2s,
        'l2_3s': l2_3s,
        'l2_avg': l2_avg,
        'l2_avg_valid_only': float(l2_valid_mean[[1, 3, 5]].mean()),
    }


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train VCoTD student model')

    # Training mode
    p.add_argument('--mode', type=str, default='distill',
                   choices=['gt_only', 'distill', 'distill_kd'],
                   help='gt_only: baseline, distill: feature KD, distill_kd: feature+response KD')

    # Data
    p.add_argument('--teacher_feat_dir', type=str, default='./teacher_features',
                   help='Root dir of cached teacher features (subdirs: train/, val/)')
    p.add_argument('--teacher_wp_dir',   type=str, default=None,
                   help='Root dir of teacher waypoint soft labels (distill_kd mode only)')
    p.add_argument('--T_obs',  type=int, default=6, help='Historical trajectory length')
    p.add_argument('--T_pred', type=int, default=6, help='Future trajectory length')

    # Model architecture
    p.add_argument('--d_vlm',      type=int, default=0,
                   help='Teacher feature dim; 0 infers it from the cache')
    p.add_argument('--d',          type=int, default=256,  help='Student hidden dim')
    p.add_argument('--K',          type=int, default=4,    help='Transformer encoder layers')
    p.add_argument('--h',          type=int, default=4,    help='Attention heads')
    p.add_argument('--d_ff',       type=int, default=1024, help='FFN hidden dim')
    p.add_argument('--d_h',        type=int, default=64,   help='Complexity estimator hidden dim')
    p.add_argument('--visual_aggregation', choices=['gap', 'uniform'], default='gap',
                   help='gap is the paper model; uniform is the later multi-token extension')
    p.add_argument('--vis_tokens', type=int, default=1,
                   help='Visual token count; the paper GAP model requires 1')

    # Distillation weights
    p.add_argument('--alpha_min', type=float, default=0.3)
    p.add_argument('--alpha_max', type=float, default=1.0)
    p.add_argument('--lambda1',   type=float, default=1.0, help='Global distill weight')
    p.add_argument('--lambda2',   type=float, default=1.0, help='Spatial distill weight')
    p.add_argument('--lambda3',   type=float, default=0.5, help='Detail distill weight')
    p.add_argument('--kd_beta',   type=float, default=0.5,
                   help='Response KD blend: (1-beta)*L_gt + beta*L_teacher_wp (distill_kd only)')
    p.add_argument('--disable_adaptive', action='store_true',
                   help='Use a fixed distillation weight for the non-adaptive ablation')
    p.add_argument('--fixed_alpha', type=float, default=1.0,
                   help='Distillation weight used with --disable_adaptive')

    # Training
    p.add_argument('--epochs',       type=int,   default=12)
    p.add_argument('--batch_size',   type=int,   default=64,
                   help='Batch size per process (global batch is this value times world size)')
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--warmup_ratio', type=float, default=0.1)
    p.add_argument('--num_workers',  type=int,   default=8)
    p.add_argument('--log_interval', type=int,   default=50)
    p.add_argument('--save_every',   type=int,   default=3,
                   help='Save checkpoint every N epochs')

    # I/O
    p.add_argument('--output_dir', type=str, default='./saves/vcotd')
    p.add_argument('--device',     type=str, default='cuda')
    p.add_argument('--resume',     action='store_true')
    p.add_argument('--max_train_samples', type=int, default=None,
                   help='Limit training set size for quick ablation experiments')
    p.add_argument('--max_val_samples', type=int, default=None,
                   help='Limit validation size for smoke tests; do not use for reported results')
    p.add_argument('--seed', type=int, default=20260918)

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
