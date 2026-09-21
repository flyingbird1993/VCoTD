#!/usr/bin/env python3
"""Canonical mask-aware evaluation for trajectory JSON files."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import torch

from planning_reasoner.metrics.planning import PlanningMetricAccumulator, parse_trajectory


def _load_pickle(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected a token mapping in {path}")
    return value


def evaluate_prediction_mapping(
    predictions: Mapping[str, str],
    ground_truth: Mapping[str, Any],
    masks: Mapping[str, Any],
    horizon: int = 6,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    accumulator = PlanningMetricAccumulator(horizon)
    missing: list[str] = []
    malformed: dict[str, str] = {}
    for token, target_value in ground_truth.items():
        if token not in predictions:
            missing.append(token)
            continue
        try:
            prediction = parse_trajectory(predictions[token], expected_steps=horizon).unsqueeze(0)
        except (TypeError, ValueError) as error:
            malformed[token] = str(error)
            continue
        target = torch.as_tensor(target_value, dtype=torch.float32)
        mask = torch.as_tensor(masks[token], dtype=torch.bool)
        if target.ndim == 2:
            target = target.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        target = target[:, :horizon, :2]
        mask = mask[:, :horizon, :2]
        if target.shape != prediction.shape:
            malformed[token] = (
                f"ground truth {tuple(target.shape)} does not match prediction {tuple(prediction.shape)}"
            )
            continue
        accumulator.update(prediction, target, mask)

    if (missing or malformed) and not allow_incomplete:
        raise ValueError(
            f"Incomplete predictions: {len(missing)} missing and {len(malformed)} malformed; "
            "pass --allow-incomplete only for debugging"
        )
    metrics = accumulator.compute()
    return {
        "protocol": {
            "horizon_steps": horizon,
            "step_seconds": 0.5,
            "official_denominator": "all parsed samples; masked horizons contribute zero",
            "valid_only_denominator": "samples valid at each horizon",
        },
        "coverage": {
            "ground_truth": len(ground_truth),
            "parsed": metrics.total_samples,
            "missing": len(missing),
            "malformed": len(malformed),
            "extra_predictions": len(set(predictions).difference(ground_truth)),
            "missing_examples": missing[:10],
            "malformed_examples": dict(list(malformed.items())[:10]),
        },
        "official_l2_by_step": metrics.official_l2,
        "valid_only_l2_by_step": metrics.valid_only_l2,
        "valid_samples_by_step": metrics.valid_samples,
        "summary": metrics.horizon_summary(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.predictions.open("r", encoding="utf-8") as handle:
        predictions = json.load(handle)
    report = evaluate_prediction_mapping(
        predictions,
        _load_pickle(args.ground_truth),
        _load_pickle(args.masks),
        horizon=args.horizon,
        allow_incomplete=args.allow_incomplete,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
