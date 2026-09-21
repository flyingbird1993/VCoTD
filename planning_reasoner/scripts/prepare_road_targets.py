#!/usr/bin/env python3
"""Prepare ego-centric drivable targets from legacy nuScenes semantic maps."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from pyquaternion import Quaternion


def _load_cache(path: Path) -> Mapping[str, Mapping[str, Any]]:
    with path.open("rb") as handle:
        cache = pickle.load(handle)
    if not isinstance(cache, Mapping):
        raise TypeError(f"Expected a token mapping in {path}")
    return cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("create_data/cached_nuscenes_info.pkl"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/planning_reasoner/road_targets_64x64.pkl")
    )
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--x-min", type=float, default=-30.0)
    parser.add_argument("--x-max", type=float, default=30.0)
    parser.add_argument("--y-min", type=float, default=-10.0)
    parser.add_argument("--y-max", type=float, default=50.0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from nuscenes.nuscenes import NuScenes

    cache = _load_cache(args.cache)
    nu_scenes = NuScenes(
        version="v1.0-trainval", dataroot=str(args.dataroot), verbose=False
    )
    logs = {record["token"]: record["location"] for record in nu_scenes.log}
    map_records = {}
    for record in nu_scenes.map:
        for log_token in record["log_tokens"]:
            if log_token in logs:
                map_records[logs[log_token]] = record
    missing_locations = sorted(
        {sample["map_location"] for sample in cache.values()}.difference(map_records)
    )
    if missing_locations:
        raise ValueError(f"Map metadata misses locations: {missing_locations}")

    map_masks = {
        location: record["mask"] for location, record in map_records.items()
    }
    targets: dict[str, np.ndarray] = {}
    target_masks: dict[str, np.ndarray] = {}

    x_step = (args.x_max - args.x_min) / args.width
    y_step = (args.y_max - args.y_min) / args.height
    lateral = np.linspace(
        args.x_min + x_step / 2, args.x_max - x_step / 2, args.width
    )
    longitudinal = np.linspace(
        args.y_max - y_step / 2, args.y_min + y_step / 2, args.height
    )
    local_x, local_y = np.meshgrid(lateral, longitudinal)

    selected_items = list(cache.items())[:args.limit] if args.limit else cache.items()
    selected_count = min(args.limit, len(cache)) if args.limit else len(cache)
    for index, (token, sample) in enumerate(selected_items, start=1):
        translation = np.asarray(sample["ego2global_translation"], dtype=np.float64)
        rotation = Quaternion(sample["ego2global_rotation"])
        yaw = rotation.yaw_pitch_roll[0]
        cosine, sine = np.cos(yaw), np.sin(yaw)
        global_x = translation[0] + cosine * local_y - sine * local_x
        global_y = translation[1] + sine * local_y + cosine * local_x
        drivable = map_masks[sample["map_location"]].is_on_mask(
            global_x.reshape(-1), global_y.reshape(-1)
        ).reshape(args.height, args.width)
        targets[token] = drivable.astype(np.uint8)[None]
        target_masks[token] = np.full(
            (1, args.height, args.width), bool(drivable.any()), dtype=np.uint8
        )
        if index % 2000 == 0:
            print(f"prepared {index}/{selected_count}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(
            {
                "metadata": {
                    "channels": ["legacy_semantic_prior_drivable"],
                    "grid_shape": [args.height, args.width],
                    "bounds": [args.x_min, args.x_max, args.y_min, args.y_max],
                    "source": "nuScenes v1.0 semantic_prior PNG",
                },
                "targets": targets,
                "masks": target_masks,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"wrote {len(targets)} targets to {args.output}")


if __name__ == "__main__":
    main()
