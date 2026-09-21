"""Losses for fixed-depth structured planning."""

from planning_reasoner.losses.structured import (
    StructuredLossConfig,
    compute_structured_loss,
    trajectory_smooth_l1,
)

__all__ = ["StructuredLossConfig", "compute_structured_loss", "trajectory_smooth_l1"]
