#!/usr/bin/env python3
"""CPU smoke test for baseline and all three structured stages."""

from __future__ import annotations

import torch

from planning_reasoner.contracts import validate_canonical_batch
from planning_reasoner.losses.structured import StructuredLossConfig, compute_structured_loss
from planning_reasoner.models.baselines import UnstructuredPlanner, UnstructuredPlannerConfig
from planning_reasoner.models.stages import ProgressivePlannerConfig, ProgressiveStructuredPlanner


def synthetic_batch() -> dict[str, torch.Tensor | list[str]]:
    batch_size, visual_count, visual_dim = 2, 12, 32
    t_obs, t_pred, agents = 6, 6, 4
    return {
        "sample_tokens": ["synthetic-0", "synthetic-1"],
        "visual_tokens": torch.randn(batch_size, visual_count, visual_dim),
        "visual_mask": torch.ones(batch_size, visual_count, dtype=torch.bool),
        "visual_positions": torch.randn(batch_size, visual_count, 3),
        "hist_traj": torch.randn(batch_size, t_obs, 2),
        "hist_mask": torch.ones(batch_size, t_obs, dtype=torch.bool),
        "command": torch.eye(3)[torch.tensor([0, 2])],
        "gt_traj": torch.randn(batch_size, t_pred, 2),
        "gt_traj_mask": torch.ones(batch_size, t_pred, dtype=torch.bool),
        "road_target": torch.randint(0, 2, (batch_size, 3, 8, 8)).float(),
        "road_target_mask": torch.ones(batch_size, 3, 8, 8, dtype=torch.bool),
        "agent_classes": torch.tensor([[0, 2, -1, -1], [1, 3, 4, -1]]),
        "agent_centers": torch.randn(batch_size, agents, 2),
        "agent_velocities": torch.randn(batch_size, agents, 2),
        "agent_futures": torch.randn(batch_size, agents, t_pred, 2),
        "agent_future_mask": torch.ones(batch_size, agents, t_pred, dtype=torch.bool),
        "agent_mask": torch.tensor([[True, True, False, False], [True, True, True, False]]),
        "future_occupancy": torch.randint(0, 2, (batch_size, t_pred, 8, 8)).float(),
        "future_occupancy_mask": torch.ones(batch_size, t_pred, 8, 8, dtype=torch.bool),
    }


def main() -> None:
    torch.manual_seed(7)
    batch = synthetic_batch()
    validate_canonical_batch(batch)

    baseline = UnstructuredPlanner(
        UnstructuredPlannerConfig(
            visual_dim=32,
            model_dim=32,
            max_visual_tokens=16,
            num_layers=1,
            num_heads=4,
            ffn_dim=64,
            dropout=0.0,
        )
    )
    baseline_output = baseline(batch)
    assert baseline_output.mean.shape == (2, 6, 2)

    config = ProgressivePlannerConfig(
        visual_dim=32,
        model_dim=32,
        max_visual_tokens=16,
        num_heads=4,
        ffn_dim=64,
        context_layers=1,
        stage_layers=1,
        dropout=0.0,
        road_height=4,
        road_width=4,
        agent_queries=5,
        agent_classes=5,
        future_height=4,
        future_width=4,
    )
    model = ProgressiveStructuredPlanner(config)
    output = model(batch)
    assert output.road.map_logits.shape == (2, 3, 4, 4)
    assert output.interaction.prediction.futures.shape == (2, 5, 6, 2)
    assert output.future.occupancy_logits.shape == (2, 6, 4, 4)
    total, components = compute_structured_loss(output, batch, StructuredLossConfig())
    total.backward()
    assert torch.isfinite(total)
    print(
        {
            "baseline_shape": tuple(baseline_output.mean.shape),
            "final_shape": tuple(output.final_trajectory.mean.shape),
            "loss": float(total.detach()),
            "components": sorted(components),
        }
    )


if __name__ == "__main__":
    main()
