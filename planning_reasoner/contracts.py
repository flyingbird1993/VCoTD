"""Canonical tensor contract shared by data, models, losses, and metrics."""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import Tensor


class CanonicalBatch(TypedDict, total=False):
    sample_tokens: list[str]
    visual_tokens: Tensor       # (B, N, Dv)
    visual_mask: Tensor         # (B, N), bool
    visual_positions: Tensor    # (B, N, Dp), optional calibrated positions
    hist_traj: Tensor           # (B, T_obs, 2), ego-local metres
    hist_mask: Tensor           # (B, T_obs), bool
    command: Tensor             # (B, 3), [right, left, forward]
    gt_traj: Tensor             # (B, T_pred, 2), ego-local metres
    gt_traj_mask: Tensor        # (B, T_pred) or (B, T_pred, 2), bool
    road_target: Tensor         # (B, C_road, H, W)
    road_target_mask: Tensor    # broadcastable to road_target
    agent_classes: Tensor       # (B, K), -1 for padding
    agent_centers: Tensor       # (B, K, 2)
    agent_velocities: Tensor    # (B, K, 2)
    agent_futures: Tensor       # (B, K, T_pred, 2)
    agent_future_mask: Tensor   # (B, K, T_pred), bool
    agent_mask: Tensor          # (B, K), bool
    future_occupancy: Tensor    # (B, T_pred, H, W)
    future_occupancy_mask: Tensor


def _expect_shape(name: str, tensor: Tensor, ndim: int, tail: tuple[int, ...] = ()) -> None:
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {tuple(tensor.shape)}")
    if tail and tuple(tensor.shape[-len(tail):]) != tail:
        raise ValueError(f"{name} must end with {tail}, got {tuple(tensor.shape)}")


def validate_canonical_batch(batch: CanonicalBatch, require_visual: bool = True) -> int:
    """Validate the shared interface and return batch size."""
    required = {"hist_traj", "hist_mask", "command", "gt_traj", "gt_traj_mask"}
    if require_visual:
        required.update({"visual_tokens", "visual_mask"})
    missing = sorted(required.difference(batch))
    if missing:
        raise ValueError(f"Canonical batch is missing: {', '.join(missing)}")

    _expect_shape("hist_traj", batch["hist_traj"], 3, (2,))
    _expect_shape("hist_mask", batch["hist_mask"], 2)
    _expect_shape("command", batch["command"], 2, (3,))
    _expect_shape("gt_traj", batch["gt_traj"], 3, (2,))
    if batch["gt_traj_mask"].ndim not in (2, 3):
        raise ValueError("gt_traj_mask must be (B,T) or (B,T,2)")

    batch_size = batch["hist_traj"].shape[0]
    for name in ("hist_mask", "command", "gt_traj", "gt_traj_mask"):
        if batch[name].shape[0] != batch_size:
            raise ValueError(f"{name} has inconsistent batch size")
    if batch["hist_mask"].shape != batch["hist_traj"].shape[:2]:
        raise ValueError("hist_mask must align with hist_traj")
    if batch["gt_traj_mask"].shape[:2] != batch["gt_traj"].shape[:2]:
        raise ValueError("gt_traj_mask must align with gt_traj")

    if require_visual:
        _expect_shape("visual_tokens", batch["visual_tokens"], 3)
        _expect_shape("visual_mask", batch["visual_mask"], 2)
        if batch["visual_mask"].shape != batch["visual_tokens"].shape[:2]:
            raise ValueError("visual_mask must align with visual_tokens")
        if batch["visual_tokens"].shape[0] != batch_size:
            raise ValueError("visual_tokens has inconsistent batch size")
        if "visual_positions" in batch:
            _expect_shape("visual_positions", batch["visual_positions"], 3)
            if batch["visual_positions"].shape[:2] != batch["visual_tokens"].shape[:2]:
                raise ValueError("visual_positions must align with visual_tokens")

    if batch["hist_mask"].dtype != torch.bool:
        raise TypeError("hist_mask must be bool")
    if batch["gt_traj_mask"].dtype != torch.bool:
        raise TypeError("gt_traj_mask must be bool")
    if require_visual and batch["visual_mask"].dtype != torch.bool:
        raise TypeError("visual_mask must be bool")
    return batch_size
