#!/usr/bin/env python3
"""Train the fixed-depth Road -> Interaction -> Future planner."""

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
from planning_reasoner.engine.steps import move_batch_to_device, structured_train_step
from planning_reasoner.losses.structured import StructuredLossConfig
from planning_reasoner.metrics.planning import PlanningMetricAccumulator, format_trajectory
from planning_reasoner.metrics.provenance import (
    build_manifest,
    record_artifacts,
    seed_everything,
    write_manifest,
)
from planning_reasoner.metrics.structured import (
    BinaryGridMetricAccumulator,
    InteractionMetricAccumulator,
)
from planning_reasoner.models.stages import ProgressivePlannerConfig, ProgressiveStructuredPlanner


def _load_pickle(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required artifact does not exist: {path}. Generate real road targets before training."
        )
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected a token mapping in {path}")
    return value


def _road_artifact(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    targets = payload.get("targets", payload)
    if not isinstance(targets, Mapping):
        raise TypeError("Road target artifact must contain a token mapping")
    masks = payload.get("masks")
    if masks is None:
        masks = {
            token: torch.full_like(
                torch.as_tensor(target),
                bool(torch.as_tensor(target).any()),
                dtype=torch.bool,
            )
            for token, target in targets.items()
        }
    if not isinstance(masks, Mapping):
        raise TypeError("Road target masks must be a token mapping")
    return targets, masks


def _dataset(
    cache: Mapping[str, Any],
    tokens: list[str],
    targets: Mapping[str, Any] | None,
    masks: Mapping[str, Any] | None,
    road_targets: Mapping[str, Any],
    road_target_masks: Mapping[str, Any],
    config: Mapping[str, Any],
    split: str,
) -> CachedTrajectoryDataset:
    data = config["data"]
    max_stage = str(config["max_stage"])
    missing_targets = set(tokens).difference(road_targets)
    if missing_targets:
        raise ValueError(f"Road targets miss {len(missing_targets)} {split} tokens")
    return CachedTrajectoryDataset(
        data=cache,
        tokens=tokens,
        t_obs=int(data["t_obs"]),
        t_pred=int(data["t_pred"]),
        trajectory_targets=targets,
        trajectory_masks=masks,
        teacher_feature_root=Path(data[f"{split}_feature_root"]),
        teacher_feature_key=str(data["feature_key"]),
        camera_count=int(data["camera_count"]),
        source_grid=tuple(data["source_grid"]),
        target_grid=tuple(data["target_grid"]),
        include_interaction_targets=max_stage != "road",
        max_agents=int(data["max_agents"]),
        occupancy_grid=tuple(data["occupancy_grid"]),
        occupancy_bounds=tuple(data["occupancy_bounds"]),
        road_targets=road_targets,
        road_target_masks=road_target_masks,
    )


@torch.no_grad()
def evaluate(
    model: ProgressiveStructuredPlanner,
    loader: DataLoader,
    device: torch.device,
    horizon: int,
    max_stage: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    model.eval()
    accumulators = {
        "road": PlanningMetricAccumulator(horizon),
        "interaction": PlanningMetricAccumulator(horizon),
        "future": PlanningMetricAccumulator(horizon),
    }
    stage_order = ["road", "interaction", "future"]
    executed_stages = set(stage_order[: stage_order.index(max_stage) + 1])
    road_metrics = BinaryGridMetricAccumulator()
    interaction_metrics = InteractionMetricAccumulator()
    future_metrics = BinaryGridMetricAccumulator()
    predictions: dict[str, str] = {}
    for cpu_batch in loader:
        batch = move_batch_to_device(cpu_batch, device)
        output = model(batch, max_stage=max_stage)
        exits = {"road": output.road.trajectory.mean}
        road_metrics.update(
            output.road.map_logits,
            batch["road_target"],
            batch.get("road_target_mask"),
        )
        if output.interaction is not None:
            exits["interaction"] = output.interaction.trajectory.mean
            interaction_metrics.update(
                output.interaction.prediction,
                batch["agent_classes"],
                batch["agent_centers"],
                batch["agent_velocities"],
                batch["agent_futures"],
                batch["agent_future_mask"],
                batch["agent_mask"],
            )
        if output.future is not None:
            exits["future"] = output.future.trajectory.mean
            future_metrics.update(
                output.future.occupancy_logits,
                batch["future_occupancy"],
                batch.get("future_occupancy_mask"),
            )
        for name, prediction in exits.items():
            accumulators[name].update(prediction, batch["gt_traj"], batch["gt_traj_mask"])
        for token, trajectory in zip(cpu_batch["sample_tokens"], exits[max_stage].cpu()):
            predictions[token] = format_trajectory(trajectory, decimals=6)

    report: dict[str, Any] = {}
    for name, accumulator in accumulators.items():
        if name not in executed_stages:
            continue
        metrics = accumulator.compute()
        report[name] = {
            "official_l2_by_step": metrics.official_l2,
            "valid_only_l2_by_step": metrics.valid_only_l2,
            "valid_samples_by_step": metrics.valid_samples,
            "summary": metrics.horizon_summary(),
        }
    report["semantic"] = {"road": road_metrics.compute()}
    if max_stage in {"interaction", "future"}:
        report["semantic"]["interaction"] = interaction_metrics.compute()
    if max_stage == "future":
        report["semantic"]["future"] = future_metrics.compute()
    return report, predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("planning_reasoner/configs/structured_fixed_depth.yaml"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    require_keys(
        config,
        {"experiment", "seed", "max_stage", "data", "model", "loss", "training"},
        "root",
    )
    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device(args.device)
    output_dir = args.output_dir or Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    max_stage = str(config["max_stage"])
    if max_stage not in {"road", "interaction", "future"}:
        raise ValueError(f"Unsupported max_stage: {max_stage}")

    data = config["data"]
    cache_path = Path(data["cache"])
    split_path = Path(data["split_file"])
    mask_path = Path(data["validation_masks"])
    target_path = Path(data["validation_ground_truth"])
    road_path = Path(data["road_targets"])
    cache = _load_pickle(cache_path)
    masks = _load_pickle(mask_path)
    targets = _load_pickle(target_path)
    road_targets, road_target_masks = _road_artifact(_load_pickle(road_path))
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    train_dataset = _dataset(
        cache,
        splits["train"],
        None,
        None,
        road_targets,
        road_target_masks,
        config,
        "train",
    )
    validation_dataset = _dataset(
        cache,
        splits["val"],
        targets,
        masks,
        road_targets,
        road_target_masks,
        config,
        "validation",
    )

    training = config["training"]
    loader_options = {
        "batch_size": int(training["batch_size"]),
        "num_workers": int(training["workers"]),
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_canonical,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    model_config = ProgressivePlannerConfig(**config["model"])
    loss_config = StructuredLossConfig(**config["loss"])
    model = ProgressiveStructuredPlanner(model_config).to(device)
    initialization = config.get("initialization", {})
    prior_checkpoint_path = initialization.get("motion_prior_checkpoint")
    if model.motion_prior is not None:
        if prior_checkpoint_path is None:
            raise ValueError("use_motion_prior requires initialization.motion_prior_checkpoint")
        prior_checkpoint = torch.load(
            prior_checkpoint_path, map_location="cpu", weights_only=True
        )
        model.motion_prior.load_state_dict(prior_checkpoint["model"], strict=True)
        if bool(initialization.get("freeze_motion_prior", True)):
            model.freeze_motion_prior()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    manifest_files = {
        "config": args.config,
        "cache": cache_path,
        "split": split_path,
        "ground_truth": target_path,
        "masks": mask_path,
        "road_targets": road_path,
    }
    if prior_checkpoint_path is not None:
        manifest_files["motion_prior_checkpoint"] = Path(prior_checkpoint_path)
    write_manifest(
        output_dir,
        build_manifest(
            config=config,
            seed=seed,
            project_root=Path.cwd(),
            files=manifest_files,
        ),
    )

    initial_validation, initial_predictions = evaluate(
        model, validation_loader, device, model_config.t_pred, max_stage
    )
    best_score = float(initial_validation[max_stage]["summary"]["avg_l2"])
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "training": None,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation": initial_validation,
        }
    ]
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    torch.save(
        {"model": model.state_dict(), "model_config": config["model"], "epoch": 0},
        output_dir / "best.pt",
    )
    (output_dir / "best_metrics.json").write_text(
        json.dumps(initial_validation, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "best_predictions.json").write_text(
        json.dumps(initial_predictions, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
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
    print(json.dumps(history[0], ensure_ascii=True, sort_keys=True))
    for epoch in range(1, epochs + 1):
        component_sums: dict[str, float] = {}
        sample_count = 0
        for cpu_batch in train_loader:
            batch = move_batch_to_device(cpu_batch, device)
            report = structured_train_step(
                model,
                batch,
                optimizer,
                loss_config,
                gradient_clip_norm=float(training["gradient_clip_norm"]),
                max_stage=max_stage,
            )
            batch_size = batch["gt_traj"].shape[0]
            sample_count += batch_size
            for name, value in report.items():
                component_sums[name] = component_sums.get(name, 0.0) + value * batch_size
        scheduler.step()
        validation, predictions = evaluate(
            model, validation_loader, device, model_config.t_pred, max_stage
        )
        record = {
            "epoch": epoch,
            "training": {name: value / sample_count for name, value in component_sums.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation": validation,
        }
        history.append(record)
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        score = float(validation[max_stage]["summary"]["avg_l2"])
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
