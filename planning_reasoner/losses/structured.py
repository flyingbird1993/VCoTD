"""Deep supervision for trajectory, road, interaction, and future outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from planning_reasoner.contracts import CanonicalBatch
from planning_reasoner.metrics.planning import coordinate_valid_mask
from planning_reasoner.models.common import InteractionPrediction, ProgressivePlannerOutput, TrajectoryPrediction


@dataclass(frozen=True)
class StructuredLossConfig:
    trajectory_objective: str = "smooth_l1"
    smooth_l1_beta: float = 1.0
    road_trajectory_weight: float = 0.3
    interaction_trajectory_weight: float = 0.5
    future_trajectory_weight: float = 1.0
    road_map_weight: float = 0.5
    interaction_weight: float = 0.5
    future_occupancy_weight: float = 0.5
    dice_weight: float = 1.0
    agent_class_weight: float = 1.0
    agent_center_weight: float = 2.0
    agent_velocity_weight: float = 0.5
    agent_future_weight: float = 1.0
    matching_class_cost: float = 1.0
    matching_center_cost: float = 2.0


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def trajectory_nll(prediction: TrajectoryPrediction, target: Tensor, mask: Tensor) -> Tensor:
    coordinate_mask = coordinate_valid_mask(mask, target)
    log_scale = prediction.log_scale.clamp(-5.0, 3.0)
    squared_error = (prediction.mean - target) ** 2
    nll = 0.5 * squared_error * torch.exp(-2.0 * log_scale) + log_scale
    return _masked_mean(nll, coordinate_mask)


def trajectory_smooth_l1(
    prediction: TrajectoryPrediction,
    target: Tensor,
    mask: Tensor,
    beta: float = 1.0,
) -> Tensor:
    """Stable trajectory-mean objective for B1-B4 input ablations."""
    coordinate_mask = coordinate_valid_mask(mask, target)
    loss = functional.smooth_l1_loss(
        prediction.mean, target, reduction="none", beta=beta
    )
    return _masked_mean(loss, coordinate_mask)


def _trajectory_loss(
    prediction: TrajectoryPrediction,
    target: Tensor,
    mask: Tensor,
    config: StructuredLossConfig,
) -> Tensor:
    if config.trajectory_objective == "smooth_l1":
        return trajectory_smooth_l1(
            prediction, target, mask, beta=config.smooth_l1_beta
        )
    if config.trajectory_objective == "gaussian_nll":
        return trajectory_nll(prediction, target, mask)
    raise ValueError(f"Unsupported trajectory_objective: {config.trajectory_objective}")


def _resize_like(target: Tensor, prediction: Tensor) -> Tensor:
    if target.shape[-2:] == prediction.shape[-2:]:
        return target
    leading_shape = target.shape[:-2]
    resized = functional.interpolate(
        target.reshape(-1, 1, *target.shape[-2:]).float(),
        size=prediction.shape[-2:],
        mode="nearest",
    )
    return resized.reshape(*leading_shape, *prediction.shape[-2:])


def _binary_grid_loss(logits: Tensor, target: Tensor, mask: Tensor | None, dice_weight: float) -> Tensor:
    target = _resize_like(target.float(), logits)
    valid = torch.ones_like(target, dtype=torch.bool) if mask is None else _resize_like(mask.float(), logits).bool()
    bce = _masked_mean(functional.binary_cross_entropy_with_logits(logits, target, reduction="none"), valid)

    probabilities = logits.sigmoid() * valid
    masked_target = target * valid
    spatial_dims = tuple(range(2, logits.ndim))
    intersection = (probabilities * masked_target).sum(dim=spatial_dims)
    denominator = probabilities.sum(dim=spatial_dims) + masked_target.sum(dim=spatial_dims)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + dice_weight * dice


def _interaction_loss(
    prediction: InteractionPrediction,
    batch: CanonicalBatch,
    config: StructuredLossConfig,
) -> Tensor:
    required = {
        "agent_classes",
        "agent_centers",
        "agent_velocities",
        "agent_futures",
        "agent_future_mask",
        "agent_mask",
    }
    missing = sorted(required.difference(batch))
    if missing:
        raise ValueError(f"Interaction supervision is missing: {', '.join(missing)}")

    background_index = prediction.class_logits.shape[-1] - 1
    classification_targets = torch.full(
        prediction.class_logits.shape[:2],
        background_index,
        dtype=torch.long,
        device=prediction.class_logits.device,
    )
    center_losses: list[Tensor] = []
    velocity_losses: list[Tensor] = []
    future_losses: list[Tensor] = []

    for batch_index in range(prediction.class_logits.shape[0]):
        target_valid = batch["agent_mask"][batch_index] & (batch["agent_classes"][batch_index] >= 0)
        target_indices = target_valid.nonzero(as_tuple=False).flatten()
        if target_indices.numel() == 0:
            continue

        target_classes = batch["agent_classes"][batch_index, target_indices].long()
        if int(target_classes.max()) >= background_index:
            raise ValueError("agent class index exceeds configured agent_classes")
        probabilities = prediction.class_logits[batch_index].softmax(dim=-1)
        class_cost = -probabilities[:, target_classes]
        center_cost = torch.cdist(
            prediction.centers[batch_index],
            batch["agent_centers"][batch_index, target_indices],
            p=1,
        )
        cost = config.matching_class_cost * class_cost + config.matching_center_cost * center_cost
        query_rows, target_columns = linear_sum_assignment(cost.detach().cpu().numpy())
        query_indices = torch.as_tensor(query_rows, device=cost.device, dtype=torch.long)
        matched_targets = target_indices[
            torch.as_tensor(target_columns, device=target_indices.device, dtype=torch.long)
        ]

        classification_targets[batch_index, query_indices] = batch[
            "agent_classes"
        ][batch_index, matched_targets].long()
        center_losses.append(
            functional.smooth_l1_loss(
                prediction.centers[batch_index, query_indices],
                batch["agent_centers"][batch_index, matched_targets],
            )
        )
        velocity_losses.append(
            functional.smooth_l1_loss(
                prediction.velocities[batch_index, query_indices],
                batch["agent_velocities"][batch_index, matched_targets],
            )
        )
        future_error = functional.smooth_l1_loss(
            prediction.futures[batch_index, query_indices],
            batch["agent_futures"][batch_index, matched_targets],
            reduction="none",
        )
        future_mask = batch["agent_future_mask"][batch_index, matched_targets].unsqueeze(-1)
        future_losses.append(_masked_mean(future_error, future_mask.expand_as(future_error)))

    class_loss = functional.cross_entropy(
        prediction.class_logits.flatten(0, 1), classification_targets.flatten()
    )
    zero = prediction.class_logits.sum() * 0.0
    center_loss = torch.stack(center_losses).mean() if center_losses else zero
    velocity_loss = torch.stack(velocity_losses).mean() if velocity_losses else zero
    future_loss = torch.stack(future_losses).mean() if future_losses else zero
    return (
        config.agent_class_weight * class_loss
        + config.agent_center_weight * center_loss
        + config.agent_velocity_weight * velocity_loss
        + config.agent_future_weight * future_loss
    )


def compute_structured_loss(
    output: ProgressivePlannerOutput,
    batch: CanonicalBatch,
    config: StructuredLossConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return weighted total and detached-friendly named components."""
    target = batch["gt_traj"]
    target_mask = batch["gt_traj_mask"]
    zero = output.road.map_logits.sum() * 0.0
    components = {
        "trajectory_road": _trajectory_loss(
            output.road.trajectory, target, target_mask, config
        ),
        "road_map": _binary_grid_loss(
            output.road.map_logits,
            batch["road_target"],
            batch.get("road_target_mask"),
            config.dice_weight,
        ),
        "trajectory_interaction": zero,
        "interaction": zero,
        "trajectory_future": zero,
        "future_occupancy": zero,
    }
    if output.interaction is not None:
        components["trajectory_interaction"] = _trajectory_loss(
            output.interaction.trajectory, target, target_mask, config
        )
        components["interaction"] = _interaction_loss(
            output.interaction.prediction, batch, config
        )
    elif config.interaction_trajectory_weight > 0 or config.interaction_weight > 0:
        raise ValueError("Interaction loss is enabled but the interaction stage was not executed")
    if output.future is not None:
        components["trajectory_future"] = _trajectory_loss(
            output.future.trajectory, target, target_mask, config
        )
        components["future_occupancy"] = _binary_grid_loss(
            output.future.occupancy_logits,
            batch["future_occupancy"],
            batch.get("future_occupancy_mask"),
            config.dice_weight,
        )
    elif config.future_trajectory_weight > 0 or config.future_occupancy_weight > 0:
        raise ValueError("Future loss is enabled but the future stage was not executed")
    total = (
        config.road_trajectory_weight * components["trajectory_road"]
        + config.interaction_trajectory_weight * components["trajectory_interaction"]
        + config.future_trajectory_weight * components["trajectory_future"]
        + config.road_map_weight * components["road_map"]
        + config.interaction_weight * components["interaction"]
        + config.future_occupancy_weight * components["future_occupancy"]
    )
    return total, components
