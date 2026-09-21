#!/usr/bin/env python3
"""Create road-target/ego-trajectory overlays for mandatory orientation QA."""

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("create_data/cached_nuscenes_info.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/planning_reasoner/road_target_qa"))
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--scale", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--x-min", type=float, default=-30.0)
    parser.add_argument("--x-max", type=float, default=30.0)
    parser.add_argument("--y-min", type=float, default=-10.0)
    parser.add_argument("--y-max", type=float, default=50.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.targets.open("rb") as handle:
        payload = pickle.load(handle)
    targets = payload["targets"] if "targets" in payload else payload
    with args.cache.open("rb") as handle:
        cache = pickle.load(handle)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    population = list(targets)
    sample_count = min(args.samples, len(population))
    selected_tokens = random.Random(args.seed).sample(population, sample_count)
    for token in selected_tokens:
        target = np.asarray(targets[token])
        if target.ndim != 3 or target.shape[0] not in (1, 3):
            raise ValueError(f"{token} has invalid road target shape {target.shape}")
        height, width = target.shape[-2:]
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., 1] = target[0] * 120
        if target.shape[0] == 3:
            rgb[..., 2] = target[1] * 150
            rgb[..., 0] = target[2] * 255
        image = Image.fromarray(rgb).resize(
            (width * args.scale, height * args.scale), resample=Image.Resampling.NEAREST
        )
        draw = ImageDraw.Draw(image)
        future = np.asarray(cache[token]["gt_ego_fut_trajs"], dtype=np.float32)
        if future.shape[0] > 6:
            future = future[1:7]
        points = []
        for x_coord, y_coord in future:
            column = (x_coord - args.x_min) / (args.x_max - args.x_min) * width * args.scale
            row = (args.y_max - y_coord) / (args.y_max - args.y_min) * height * args.scale
            points.append((float(column), float(row)))
        if len(points) > 1:
            draw.line(points, fill=(255, 255, 255), width=max(args.scale // 2, 2))
        for point in points:
            radius = max(args.scale // 2, 2)
            draw.ellipse(
                (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
                fill=(255, 255, 255),
            )
        image.save(args.output_dir / f"{token}.png")
    print(f"wrote {sample_count} QA overlays to {args.output_dir}")


if __name__ == "__main__":
    main()
