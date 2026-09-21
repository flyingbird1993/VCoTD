#!/usr/bin/env python3
"""Run and record the non-learned constant-velocity B0 experiment."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import torch

from planning_reasoner.data.canonical import canonical_trajectory
from planning_reasoner.metrics.planning import PlanningMetricAccumulator, format_trajectory
from planning_reasoner.metrics.provenance import (
    build_manifest,
    record_artifacts,
    seed_everything,
    write_manifest,
)
from planning_reasoner.models.baselines import ConstantVelocityBaseline


def _load(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected a token mapping in {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("create_data/cached_nuscenes_info.pkl"))
    parser.add_argument("--ground-truth", type=Path, default=Path("tools/data/metrics/gt_traj.pkl"))
    parser.add_argument("--masks", type=Path, default=Path("tools/data/metrics/gt_traj_mask.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/planning_reasoner/b0_constant_velocity"))
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--history", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    cache = _load(args.cache)
    ground_truth = _load(args.ground_truth)
    masks = _load(args.masks)
    model = ConstantVelocityBaseline(args.horizon)
    accumulator = PlanningMetricAccumulator(args.horizon)
    predictions: dict[str, str] = {}

    for token, target_value in ground_truth.items():
        if token not in cache:
            raise KeyError(f"Ground-truth token is absent from cache: {token}")
        history, history_mask = canonical_trajectory(
            cache[token]["gt_ego_his_trajs"], args.history, keep_last=True
        )
        prediction = model(
            {"hist_traj": history.unsqueeze(0), "hist_mask": history_mask.unsqueeze(0)}
        ).mean
        target = torch.as_tensor(target_value, dtype=torch.float32)[:, :args.horizon, :2]
        mask = torch.as_tensor(masks[token], dtype=torch.bool)[:, :args.horizon, :2]
        accumulator.update(prediction, target, mask)
        predictions[token] = format_trajectory(prediction[0], decimals=6)

    metrics = accumulator.compute()
    report = {
        "experiment": "B0_constant_velocity",
        "official_l2_by_step": metrics.official_l2,
        "valid_only_l2_by_step": metrics.valid_only_l2,
        "valid_samples_by_step": metrics.valid_samples,
        "summary": metrics.horizon_summary(),
        "sample_count": metrics.total_samples,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions.json").write_text(
        json.dumps(predictions, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = build_manifest(
        config={
            "experiment": "B0_constant_velocity",
            "horizon": args.horizon,
            "history": args.history,
        },
        seed=args.seed,
        project_root=Path.cwd(),
        files={"cache": args.cache, "ground_truth": args.ground_truth, "masks": args.masks},
    )
    write_manifest(args.output_dir, manifest)
    record_artifacts(
        args.output_dir,
        {
            "predictions": args.output_dir / "predictions.json",
            "metrics": args.output_dir / "metrics.json",
        },
    )
    print(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
