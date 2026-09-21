"""Small training primitives that keep experiment scripts auditable."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from planning_reasoner.contracts import CanonicalBatch, validate_canonical_batch
from planning_reasoner.losses.structured import StructuredLossConfig, compute_structured_loss
from planning_reasoner.models.common import ProgressivePlannerOutput


def move_batch_to_device(batch: CanonicalBatch, device: torch.device | str) -> CanonicalBatch:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def structured_train_step(
    model: nn.Module,
    batch: CanonicalBatch,
    optimizer: torch.optim.Optimizer,
    loss_config: StructuredLossConfig,
    gradient_clip_norm: float | None = 1.0,
    max_stage: str = "future",
) -> dict[str, float]:
    validate_canonical_batch(batch)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch, max_stage=max_stage)
    if not isinstance(output, ProgressivePlannerOutput):
        raise TypeError("structured_train_step expects ProgressivePlannerOutput")
    total, components = compute_structured_loss(output, batch, loss_config)
    total.backward()
    if gradient_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    report: dict[str, Any] = {name: float(value.detach()) for name, value in components.items()}
    report["total"] = float(total.detach())
    return report
