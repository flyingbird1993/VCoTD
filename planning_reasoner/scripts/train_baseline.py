#!/usr/bin/env python3
"""Shared B1-B4 trainer with controlled input ablations."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from planning_reasoner.config import load_experiment_config, require_keys
from planning_reasoner.data.canonical import CachedTrajectoryDataset, collate_canonical
from planning_reasoner.losses.structured import trajectory_nll, trajectory_smooth_l1
from planning_reasoner.metrics.planning import PlanningMetricAccumulator, format_trajectory
from planning_reasoner.metrics.provenance import (
    build_manifest,
    record_artifacts,
    seed_everything,
    write_manifest,
)
from planning_reasoner.models.baselines import UnstructuredPlanner, UnstructuredPlannerConfig


def _load_pickle(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected a mapping in {path}")
    return value


def _build_dataset(
    cache: Mapping[str, Any],
    tokens: list[str],
    config: Mapping[str, Any],
    split: str,
    targets: Mapping[str, Any] | None,
    masks: Mapping[str, Any] | None,
) -> CachedTrajectoryDataset:
    use_visual = bool(config["model"]["use_visual"])
    data_config = config["data"]
    return CachedTrajectoryDataset(
        data=cache,
        tokens=tokens,
        t_obs=int(data_config["t_obs"]),
        t_pred=int(data_config["t_pred"]),
        trajectory_targets=targets,
        trajectory_masks=masks,
        teacher_feature_root=Path(data_config[f"{split}_feature_root"]) if use_visual else None,
        teacher_feature_key=str(data_config["feature_key"]),
        camera_count=int(data_config["camera_count"]) if use_visual else None,
        source_grid=tuple(data_config["source_grid"]) if use_visual else None,
        target_grid=tuple(data_config["target_grid"]) if use_visual else None,
    )


@torch.no_grad()
def evaluate(
    model: UnstructuredPlanner,
    loader: DataLoader,
    device: torch.device,
    horizon: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    model.eval()
    accumulator = PlanningMetricAccumulator(horizon)
    predictions: dict[str, str] = {}
    for batch in loader:
        tensor_batch = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        output = model(tensor_batch)
        accumulator.update(output.mean, tensor_batch["gt_traj"], tensor_batch["gt_traj_mask"])
        for token, trajectory in zip(batch["sample_tokens"], output.mean.cpu()):
            predictions[token] = format_trajectory(trajectory, decimals=6)
    metrics = accumulator.compute()
    return {
        "official_l2_by_step": metrics.official_l2,
        "valid_only_l2_by_step": metrics.valid_only_l2,
        "valid_samples_by_step": metrics.valid_samples,
        "summary": metrics.horizon_summary(),
        "sample_count": metrics.total_samples,
    }, predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    require_keys(config, {"experiment", "seed", "data", "model", "training"}, "root")
    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device(args.device)
    output_dir = args.output_dir or Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    data_config = config["data"]
    cache_path = Path(data_config["cache"])
    split_path = Path(data_config["split_file"])
    mask_path = Path(data_config["validation_masks"])
    target_path = Path(data_config["validation_ground_truth"])
    cache = _load_pickle(cache_path)
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    validation_masks = _load_pickle(mask_path)
    validation_targets = _load_pickle(target_path)
    train_dataset = _build_dataset(cache, splits["train"], config, "train", None, None)
    validation_dataset = _build_dataset(
        cache, splits["val"], config, "validation", validation_targets, validation_masks
    )

    training_config = config["training"]
    loader_options = {
        "batch_size": int(training_config["batch_size"]),
        "num_workers": int(training_config["workers"]),
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_canonical,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, drop_last=False, **loader_options)

    model_config = UnstructuredPlannerConfig(**config["model"])
    model = UnstructuredPlanner(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    epochs = int(training_config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    gradient_clip = float(training_config["gradient_clip_norm"])
    loss_name = str(training_config.get("trajectory_loss", "smooth_l1"))
    smooth_l1_beta = float(training_config.get("smooth_l1_beta", 1.0))
    if loss_name not in {"smooth_l1", "gaussian_nll"}:
        raise ValueError(f"Unsupported trajectory_loss: {loss_name}")

    manifest = build_manifest(
        config=config,
        seed=seed,
        project_root=Path.cwd(),
        files={
            "config": args.config,
            "cache": cache_path,
            "split": split_path,
            "ground_truth": target_path,
            "masks": mask_path,
        },
    )
    write_manifest(output_dir, manifest)
    best_score = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_samples = 0
        for batch in train_loader:
            batch = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            if loss_name == "smooth_l1":
                loss = trajectory_smooth_l1(
                    output,
                    batch["gt_traj"],
                    batch["gt_traj_mask"],
                    beta=smooth_l1_beta,
                )
            else:
                loss = trajectory_nll(output, batch["gt_traj"], batch["gt_traj_mask"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            batch_size = batch["gt_traj"].shape[0]
            train_loss += float(loss.detach()) * batch_size
            train_samples += batch_size
        scheduler.step()

        validation, predictions = evaluate(model, validation_loader, device, model_config.horizon)
        record = {
            "epoch": epoch,
            "train_loss": train_loss / train_samples,
            "trajectory_loss": loss_name,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation": validation,
        }
        history.append(record)
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        score = float(validation["summary"]["avg_l2"])
        if score < best_score:
            best_score = score
            torch.save(
                {"model": model.state_dict(), "model_config": config["model"], "epoch": epoch},
                output_dir / "best.pt",
            )
            (output_dir / "best_metrics.json").write_text(
                json.dumps(validation, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (output_dir / "best_predictions.json").write_text(
                json.dumps(predictions, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            record_artifacts(
                output_dir,
                {
                    "checkpoint": output_dir / "best.pt",
                    "predictions": output_dir / "best_predictions.json",
                    "metrics": output_dir / "best_metrics.json",
                },
            )
        print(json.dumps(record, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
