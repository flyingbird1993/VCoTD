from __future__ import annotations

import unittest

import torch

from planning_reasoner.contracts import validate_canonical_batch
from planning_reasoner.data.canonical import (
    CachedTrajectoryDataset,
    cached_interaction_targets,
    canonical_agent_class,
    canonical_command,
    canonical_trajectory,
    collate_canonical,
    reduce_camera_grid_tokens,
    rasterize_agent_futures,
)


class DataContractTest(unittest.TestCase):
    def test_camera_grid_pooling_does_not_mix_cameras(self) -> None:
        first = torch.ones(4, 1)
        second = torch.full((4, 1), 10.0)
        pooled, positions = reduce_camera_grid_tokens(
            torch.cat([first, second]), camera_count=2, source_grid=(2, 2), target_grid=(1, 1)
        )
        self.assertTrue(torch.equal(pooled[:, 0], torch.tensor([1.0, 10.0])))
        self.assertEqual(tuple(positions.shape), (2, 3))
        self.assertTrue(torch.equal(positions[:, 0], torch.tensor([-1.0, 1.0])))

    def test_canonical_helpers_and_collation(self) -> None:
        history, history_mask = canonical_trajectory([[1.0, 2.0]], length=3, keep_last=True)
        future, future_mask = canonical_trajectory([[3.0, 4.0]], length=3, keep_last=False)
        item = {
            "sample_tokens": ["token"],
            "hist_traj": history,
            "hist_mask": history_mask,
            "command": canonical_command([0, 1, 0]),
            "gt_traj": future,
            "gt_traj_mask": future_mask,
            "visual_tokens": torch.randn(2, 4),
            "visual_mask": torch.ones(2, dtype=torch.bool),
            "visual_positions": torch.randn(2, 3),
        }
        batch = collate_canonical([item, item])
        self.assertEqual(validate_canonical_batch(batch), 2)
        self.assertEqual(tuple(batch["visual_positions"].shape), (2, 2, 3))
        self.assertTrue(torch.equal(batch["hist_mask"][0], torch.tensor([False, False, True])))

    def test_grid_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            reduce_camera_grid_tokens(torch.randn(7, 4), 2, (2, 2), (1, 1))

    def test_cached_interaction_targets_and_occupancy(self) -> None:
        sample = {
            "gt_boxes": [[1.0, 2.0, 0, 1, 1, 1, 0], [20.0, 20.0, 0, 1, 1, 1, 0]],
            "gt_velocity": [[1.0, 0.0], [0.0, 0.0]],
            "gt_agent_fut_trajs": [[1.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            "gt_agent_fut_masks": [[1, 1], [1, 1]],
            "gt_names": ["vehicle.car", "animal"],
            "valid_flag": [True, True],
        }
        targets = cached_interaction_targets(sample, max_agents=3, t_pred=2)
        self.assertEqual(canonical_agent_class("vehicle.car"), 0)
        self.assertIsNone(canonical_agent_class("animal"))
        self.assertEqual(targets["agent_mask"].tolist(), [True, False, False])
        occupancy = rasterize_agent_futures(
            targets["agent_centers"],
            targets["agent_futures"],
            targets["agent_future_mask"],
            targets["agent_mask"],
            (4, 4),
            (0.0, 4.0, 0.0, 4.0),
        )
        self.assertEqual(float(occupancy.sum()), 2.0)

    def test_validation_target_overrides_cache_trajectory(self) -> None:
        sample = {
            "gt_ego_his_trajs": [[0.0, 0.0]],
            "gt_ego_fut_trajs": [[0.0, 0.0], [99.0, 99.0]],
            "gt_ego_fut_masks": [1],
            "gt_ego_fut_cmd": [0, 0, 1],
        }
        target = torch.tensor([[[1.0, 2.0]]])
        dataset = CachedTrajectoryDataset(
            {"token": sample},
            ["token"],
            t_obs=1,
            t_pred=1,
            trajectory_targets={"token": target},
            trajectory_masks={"token": torch.ones(1, 1, 2)},
        )
        item = dataset[0]
        self.assertTrue(torch.equal(item["gt_traj"], target[0]))


if __name__ == "__main__":
    unittest.main()
