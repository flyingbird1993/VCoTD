"""Dataset-level semantic probes for fixed-depth stage validation."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from planning_reasoner.models.common import InteractionPrediction


def _resize_grid(value: Tensor, size: tuple[int, int]) -> Tensor:
    if value.shape[-2:] == size:
        return value
    leading = value.shape[:-2]
    resized = functional.interpolate(
        value.reshape(-1, 1, *value.shape[-2:]).float(),
        size=size,
        mode="nearest",
    )
    return resized.reshape(*leading, *size)


class BinaryGridMetricAccumulator:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.valid_elements = 0

    def update(self, logits: Tensor, target: Tensor, mask: Tensor | None = None) -> None:
        target = _resize_grid(target, logits.shape[-2:]).to(logits.device) >= 0.5
        if target.shape != logits.shape:
            raise ValueError("Grid target must match logits after resizing")
        if mask is None:
            valid = torch.ones_like(target)
        else:
            valid = _resize_grid(mask, logits.shape[-2:]).to(logits.device).bool()
            valid = torch.broadcast_to(valid, target.shape)
        prediction = logits.sigmoid() >= self.threshold
        self.true_positive += int((prediction & target & valid).sum())
        self.false_positive += int((prediction & ~target & valid).sum())
        self.false_negative += int((~prediction & target & valid).sum())
        self.valid_elements += int(valid.sum())

    def compute(self) -> dict[str, float | int]:
        union = self.true_positive + self.false_positive + self.false_negative
        f1_denominator = 2 * self.true_positive + self.false_positive + self.false_negative
        return {
            "iou": self.true_positive / max(union, 1),
            "f1": 2 * self.true_positive / max(f1_denominator, 1),
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "valid_elements": self.valid_elements,
        }


class InteractionMetricAccumulator:
    """Oracle-center-matched semantic probe, not a detection AP metric."""

    def __init__(self) -> None:
        self.matched = 0
        self.class_correct = 0
        self.center_distance = 0.0
        self.velocity_distance = 0.0
        self.future_distance = 0.0
        self.future_points = 0
        self.final_distance = 0.0
        self.final_agents = 0

    def update(
        self,
        prediction: InteractionPrediction,
        classes: Tensor,
        centers: Tensor,
        velocities: Tensor,
        futures: Tensor,
        future_mask: Tensor,
        agent_mask: Tensor,
    ) -> None:
        background = prediction.class_logits.shape[-1] - 1
        for batch_index in range(prediction.class_logits.shape[0]):
            valid_indices = (
                agent_mask[batch_index] & (classes[batch_index] >= 0)
            ).nonzero(as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                continue
            cost = torch.cdist(
                prediction.centers[batch_index], centers[batch_index, valid_indices], p=2
            )
            rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
            query_indices = torch.as_tensor(rows, device=cost.device, dtype=torch.long)
            target_indices = valid_indices[
                torch.as_tensor(columns, device=valid_indices.device, dtype=torch.long)
            ]
            count = query_indices.numel()
            self.matched += count
            predicted_classes = prediction.class_logits[batch_index, query_indices].argmax(-1)
            self.class_correct += int(
                (predicted_classes == classes[batch_index, target_indices]).sum()
            )
            self.center_distance += float(
                torch.linalg.vector_norm(
                    prediction.centers[batch_index, query_indices]
                    - centers[batch_index, target_indices],
                    dim=-1,
                ).sum()
            )
            self.velocity_distance += float(
                torch.linalg.vector_norm(
                    prediction.velocities[batch_index, query_indices]
                    - velocities[batch_index, target_indices],
                    dim=-1,
                ).sum()
            )
            distances = torch.linalg.vector_norm(
                prediction.futures[batch_index, query_indices]
                - futures[batch_index, target_indices],
                dim=-1,
            )
            masks = future_mask[batch_index, target_indices]
            self.future_distance += float((distances * masks).sum())
            self.future_points += int(masks.sum())
            for row_index in range(count):
                valid_times = masks[row_index].nonzero(as_tuple=False).flatten()
                if valid_times.numel():
                    self.final_distance += float(distances[row_index, valid_times[-1]])
                    self.final_agents += 1

    def compute(self) -> dict[str, float | int]:
        return {
            "matched_agents": self.matched,
            "matched_class_accuracy": self.class_correct / max(self.matched, 1),
            "matched_center_error": self.center_distance / max(self.matched, 1),
            "matched_velocity_error": self.velocity_distance / max(self.matched, 1),
            "matched_future_ade": self.future_distance / max(self.future_points, 1),
            "matched_future_fde": self.final_distance / max(self.final_agents, 1),
        }
