from __future__ import annotations

import unittest

import numpy as np
import torch

from train_vcotd import VCoTDDataset, collate_fn


class VCoTDDataTest(unittest.TestCase):
    def test_trajectory_padding_and_cache_mask_alignment(self) -> None:
        data = {
            "sample": {
                "gt_ego_his_trajs": np.array([[1.0, 2.0]], dtype=np.float32),
                "gt_ego_fut_trajs": np.array(
                    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32
                ),
                "gt_ego_fut_masks": np.array([1.0, 0.0], dtype=np.float32),
            }
        }
        dataset = VCoTDDataset(
            split="train",
            data=data,
            tokens=["sample"],
            mode="gt_only",
            T_obs=3,
            T_pred=4,
        )
        item = dataset[0]
        self.assertEqual(tuple(item["hist_traj"].shape), (3, 2))
        self.assertEqual(tuple(item["gt_traj"].shape), (4, 2))
        self.assertEqual(item["gt_mask"].tolist(), [True, False, False, False])

    def test_collate_builds_teacher_padding_mask(self) -> None:
        base = {
            "hist_traj": torch.zeros(2, 2),
            "gt_traj": torch.zeros(2, 2),
            "gt_mask": torch.ones(2, dtype=torch.bool),
        }
        batch = []
        for index, length in enumerate((2, 3)):
            item = {**base, "token": str(index)}
            for level in ("shallow", "middle", "deep"):
                item[f"feat_{level}"] = torch.ones(length, 4)
            batch.append(item)
        collated = collate_fn(batch)
        self.assertEqual(tuple(collated["feat_deep"].shape), (2, 3, 4))
        self.assertEqual(
            collated["feat_mask"].tolist(),
            [[True, True, False], [True, True, True]],
        )


if __name__ == "__main__":
    unittest.main()
