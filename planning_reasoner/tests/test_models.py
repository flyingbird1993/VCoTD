from __future__ import annotations

import unittest

import torch

from planning_reasoner.losses.structured import StructuredLossConfig, compute_structured_loss
from planning_reasoner.models.baselines import (
    ConstantVelocityBaseline,
    UnstructuredPlanner,
    UnstructuredPlannerConfig,
)
from planning_reasoner.models.stages import ProgressivePlannerConfig, ProgressiveStructuredPlanner
from planning_reasoner.scripts.smoke_test import synthetic_batch


class ModelTest(unittest.TestCase):
    def test_constant_velocity_uses_last_two_valid_points(self) -> None:
        batch = {
            "hist_traj": torch.tensor([[[9.0, 9.0], [0.0, 0.0], [1.0, 2.0]]]),
            "hist_mask": torch.tensor([[False, True, True]]),
        }
        prediction = ConstantVelocityBaseline(horizon=2)(batch).mean
        expected = torch.tensor([[[2.0, 4.0], [3.0, 6.0]]])
        self.assertTrue(torch.equal(prediction, expected))

    def test_history_only_unstructured_baseline(self) -> None:
        batch = synthetic_batch()
        model = UnstructuredPlanner(
            UnstructuredPlannerConfig(
                model_dim=16,
                horizon=6,
                num_layers=1,
                num_heads=4,
                ffn_dim=32,
                dropout=0.0,
                use_visual=False,
                use_history=True,
                use_command=False,
            )
        )
        self.assertEqual(tuple(model(batch).mean.shape), (2, 6, 2))

    def test_all_structured_stages_backpropagate(self) -> None:
        batch = synthetic_batch()
        config = ProgressivePlannerConfig(
            visual_dim=32,
            model_dim=16,
            max_visual_tokens=16,
            num_heads=4,
            ffn_dim=32,
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
        total, _ = compute_structured_loss(output, batch, StructuredLossConfig())
        total.backward()
        for name in ("road_stage.queries", "interaction_stage.queries", "future_stage.queries"):
            parameter = dict(model.named_parameters())[name]
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_road_depth_skips_later_stages(self) -> None:
        batch = synthetic_batch()
        config = ProgressivePlannerConfig(
            visual_dim=32,
            model_dim=16,
            max_visual_tokens=16,
            num_heads=4,
            ffn_dim=32,
            context_layers=1,
            stage_layers=1,
            dropout=0.0,
            road_channels=3,
            road_height=4,
            road_width=4,
            agent_queries=5,
            agent_classes=5,
            future_height=4,
            future_width=4,
        )
        output = ProgressiveStructuredPlanner(config)(batch, max_stage="road")
        self.assertIsNone(output.interaction)
        self.assertIsNone(output.future)
        self.assertIs(output.final_trajectory, output.road.trajectory)

    def test_zero_residuals_preserve_motion_prior(self) -> None:
        batch = synthetic_batch()
        config = ProgressivePlannerConfig(
            visual_dim=32,
            model_dim=16,
            max_visual_tokens=16,
            num_heads=4,
            ffn_dim=32,
            context_layers=1,
            stage_layers=1,
            dropout=0.0,
            road_height=4,
            road_width=4,
            agent_queries=5,
            agent_classes=5,
            future_height=4,
            future_width=4,
            use_motion_prior=True,
            zero_initialize_residual=True,
        )
        model = ProgressiveStructuredPlanner(config).eval()
        prior = model.motion_prior(batch).mean
        output = model(batch, max_stage="future")
        self.assertTrue(torch.equal(output.road.trajectory.mean, prior))
        self.assertTrue(torch.equal(output.interaction.trajectory.mean, prior))
        self.assertTrue(torch.equal(output.future.trajectory.mean, prior))


if __name__ == "__main__":
    unittest.main()
