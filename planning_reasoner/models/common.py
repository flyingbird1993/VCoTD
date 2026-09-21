"""Typed outputs shared by models, losses, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class TrajectoryPrediction:
    mean: Tensor       # (B, T_pred, 2)
    log_scale: Tensor  # (B, T_pred, 2)


@dataclass
class RoadStageOutput:
    latent: Tensor
    map_logits: Tensor
    trajectory: TrajectoryPrediction


@dataclass
class InteractionPrediction:
    class_logits: Tensor
    centers: Tensor
    velocities: Tensor
    futures: Tensor


@dataclass
class InteractionStageOutput:
    latent: Tensor
    prediction: InteractionPrediction
    trajectory: TrajectoryPrediction


@dataclass
class FutureStageOutput:
    latent: Tensor
    occupancy_logits: Tensor
    trajectory: TrajectoryPrediction


@dataclass
class ProgressivePlannerOutput:
    road: RoadStageOutput
    interaction: InteractionStageOutput | None = None
    future: FutureStageOutput | None = None

    @property
    def final_trajectory(self) -> TrajectoryPrediction:
        if self.future is not None:
            return self.future.trajectory
        if self.interaction is not None:
            return self.interaction.trajectory
        return self.road.trajectory
