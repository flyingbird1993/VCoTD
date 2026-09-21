from __future__ import annotations

import unittest

import torch

from planning_reasoner.metrics.planning import (
    PlanningMetricAccumulator,
    format_trajectory,
    parse_trajectory,
)
from planning_reasoner.losses.structured import trajectory_smooth_l1
from planning_reasoner.models.common import TrajectoryPrediction
from planning_reasoner.metrics.structured import BinaryGridMetricAccumulator


class PlanningMetricTest(unittest.TestCase):
    def test_parser_round_trip_and_strict_length(self) -> None:
        trajectory = torch.tensor([[1.25, -2.0], [3.0, 4.5]])
        rendered = format_trajectory(trajectory, decimals=3)
        self.assertTrue(torch.allclose(parse_trajectory(rendered, 2), trajectory))
        with self.assertRaises(ValueError):
            parse_trajectory(rendered, 3)

    def test_official_and_valid_only_denominators_are_both_reported(self) -> None:
        prediction = torch.tensor([[[1.0, 0.0]], [[2.0, 0.0]]])
        target = torch.zeros_like(prediction)
        mask = torch.tensor([[[True, True]], [[False, False]]])
        accumulator = PlanningMetricAccumulator(horizon=1)
        accumulator.update(prediction, target, mask)
        metrics = accumulator.compute()
        self.assertEqual(metrics.official_l2, [0.5])
        self.assertEqual(metrics.valid_only_l2, [1.0])
        self.assertEqual(metrics.valid_samples, [1])

    def test_smooth_l1_ignores_invalid_coordinates(self) -> None:
        prediction = TrajectoryPrediction(
            mean=torch.tensor([[[1.0, 100.0]]]),
            log_scale=torch.zeros(1, 1, 2),
        )
        target = torch.zeros(1, 1, 2)
        mask = torch.tensor([[[True, False]]])
        self.assertEqual(float(trajectory_smooth_l1(prediction, target, mask)), 0.5)

    def test_binary_grid_metrics(self) -> None:
        metric = BinaryGridMetricAccumulator()
        logits = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]])
        target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        metric.update(logits, target)
        result = metric.compute()
        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["false_negative"], 1)
        self.assertAlmostEqual(result["iou"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
