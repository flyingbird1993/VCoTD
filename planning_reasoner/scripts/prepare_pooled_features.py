#!/usr/bin/env python3
"""Pre-pool one Qwen feature layer to reduce repeated B3/B4 training I/O."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from planning_reasoner.data.canonical import reduce_camera_grid_tokens


def _convert(task: tuple[str, str, str, int, int, int, int, int]) -> tuple[str, str]:
    source_value, target_value, feature_key, camera_count, source_h, source_w, target_h, target_w = task
    source = Path(source_value)
    target = Path(target_value)
    if target.is_file():
        return "skipped", source.stem
    payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    features = payload[feature_key].float() if isinstance(payload, dict) else payload.float()
    pooled, _ = reduce_camera_grid_tokens(
        features,
        camera_count=camera_count,
        source_grid=(source_h, source_w),
        target_grid=(target_h, target_w),
    )
    temporary = target.with_suffix(".tmp")
    torch.save(pooled.half(), temporary)
    temporary.replace(target)
    return "written", source.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--feature-key", default="deep")
    parser.add_argument("--camera-count", type=int, default=6)
    parser.add_argument("--source-grid", type=int, nargs=2, default=(16, 16))
    parser.add_argument("--target-grid", type=int, nargs=2, default=(8, 8))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    sources = sorted(args.input_root.glob("*.pt"))
    if args.limit is not None:
        sources = sources[:args.limit]
    tasks = [
        (
            str(source),
            str(args.output_root / source.name),
            args.feature_key,
            args.camera_count,
            args.source_grid[0],
            args.source_grid[1],
            args.target_grid[0],
            args.target_grid[1],
        )
        for source in sources
    ]
    counts = {"written": 0, "skipped": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, (status, _) in enumerate(executor.map(_convert, tasks, chunksize=8), start=1):
            counts[status] += 1
            if index % 1000 == 0:
                print(f"converted {index}/{len(tasks)}")
    manifest = {
        "input_root": str(args.input_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "feature_key": args.feature_key,
        "camera_count": args.camera_count,
        "source_grid": list(args.source_grid),
        "target_grid": list(args.target_grid),
        "dtype": "float16",
        "source_files": len(sources),
        **counts,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
