"""One mask-aware trajectory metric implementation for training and JSON evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor


_POINT_PATTERN = re.compile(
    r"\(\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*,\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*\)"
)


def parse_trajectory(value: str, expected_steps: int | None = None) -> Tensor:
    points = [(float(x), float(y)) for x, y in _POINT_PATTERN.findall(value)]
    if expected_steps is not None and len(points) != expected_steps:
        raise ValueError(f"Expected {expected_steps} trajectory points, found {len(points)}")
    if not points:
        raise ValueError("No trajectory points found")
    return torch.tensor(points, dtype=torch.float32)


def format_trajectory(trajectory: Tensor, decimals: int = 4) -> str:
    if trajectory.ndim != 2 or trajectory.shape[-1] != 2:
        raise ValueError("trajectory must be (T,2)")
    return "[" + ", ".join(
        f"({x:.{decimals}f},{y:.{decimals}f})" for x, y in trajectory.detach().cpu().tolist()
    ) + "]"


def coordinate_valid_mask(mask: Tensor, target: Tensor) -> Tensor:
    mask = mask.to(dtype=torch.bool, device=target.device)
    if mask.ndim == 2:
        mask = mask.unsqueeze(-1).expand_as(target)
    if mask.shape != target.shape:
        raise ValueError(f"Mask {tuple(mask.shape)} does not align with target {tuple(target.shape)}")
    return mask


@dataclass
class PlanningMetrics:
    official_l2: list[float]
    valid_only_l2: list[float]
    total_samples: int
    valid_samples: list[int]

    def horizon_summary(self, indices: Iterable[int] = (1, 3, 5)) -> dict[str, float]:
        selected = list(indices)
        if max(selected) >= len(self.official_l2):
            raise ValueError("Requested horizon exceeds available trajectory length")
        return {
            **{f"l2_{(index + 1) / 2:g}s": self.official_l2[index] for index in selected},
            "avg_l2": sum(self.official_l2[index] for index in selected) / len(selected),
            "avg_l2_valid_only": sum(self.valid_only_l2[index] for index in selected) / len(selected),
        }


class PlanningMetricAccumulator:
    """Accumulate both legacy official-mask and valid-only L2 metrics.

    The official FSDrive/UniAD port divides masked sums by all samples, so an
    invalid horizon contributes zero. The valid-only value is reported beside
    it to make the denominator explicit.
    """

    def __init__(self, horizon: int) -> None:
        self.horizon = horizon
        self.l2_sum = torch.zeros(horizon, dtype=torch.float64)
        self.total = 0
        self.valid = torch.zeros(horizon, dtype=torch.long)

    def update(self, prediction: Tensor, target: Tensor, mask: Tensor) -> None:
        if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 2:
            raise ValueError("prediction and target must both be (B,T,2)")
        if prediction.shape[1] != self.horizon:
            raise ValueError(f"Expected horizon {self.horizon}, got {prediction.shape[1]}")
        coord_mask = coordinate_valid_mask(mask, target)
        squared = ((prediction - target) ** 2) * coord_mask
        l2 = torch.sqrt(squared.sum(dim=-1))
        valid_horizon = coord_mask.all(dim=-1)
        self.l2_sum += l2.double().sum(dim=0).cpu()
        self.valid += valid_horizon.sum(dim=0).cpu()
        self.total += prediction.shape[0]

    def compute(self) -> PlanningMetrics:
        if self.total == 0:
            raise RuntimeError("No samples were accumulated")
        official = self.l2_sum / self.total
        valid_only = self.l2_sum / self.valid.clamp_min(1)
        return PlanningMetrics(
            official_l2=official.tolist(),
            valid_only_l2=valid_only.tolist(),
            total_samples=self.total,
            valid_samples=self.valid.tolist(),
        )
