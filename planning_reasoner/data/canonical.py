"""Adapters from the existing nuScenes cache to one canonical tensor contract."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from planning_reasoner.contracts import CanonicalBatch


NUSCENES_AGENT_CLASSES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)


def canonical_agent_class(category: str) -> int | None:
    """Map raw nuScenes categories to the standard 10 detection classes."""
    if category == "vehicle.car" or category.startswith("vehicle.emergency"):
        name = "car"
    elif category == "vehicle.truck":
        name = "truck"
    elif category == "vehicle.construction":
        name = "construction_vehicle"
    elif category.startswith("vehicle.bus"):
        name = "bus"
    elif category == "vehicle.trailer":
        name = "trailer"
    elif category == "movable_object.barrier":
        name = "barrier"
    elif category == "vehicle.motorcycle":
        name = "motorcycle"
    elif category == "vehicle.bicycle":
        name = "bicycle"
    elif category in {
        "human.pedestrian.adult",
        "human.pedestrian.child",
        "human.pedestrian.construction_worker",
        "human.pedestrian.police_officer",
    }:
        name = "pedestrian"
    elif category == "movable_object.trafficcone":
        name = "traffic_cone"
    else:
        return None
    return NUSCENES_AGENT_CLASSES.index(name)


def canonical_command(value: Any) -> Tensor:
    """Return command as [right, left, forward]."""
    tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
    if tensor.numel() == 1:
        index = int(tensor.item())
        if index not in (0, 1, 2):
            raise ValueError(f"Command index must be 0, 1, or 2, got {index}")
        return torch.nn.functional.one_hot(torch.tensor(index), 3).float()
    if tensor.numel() != 3:
        raise ValueError(f"Command must contain 3 values, got {tensor.numel()}")
    if tensor.sum() <= 0:
        raise ValueError("Command vector must contain a positive entry")
    return tensor / tensor.sum()


def canonical_trajectory(value: Any, length: int, keep_last: bool) -> tuple[Tensor, Tensor]:
    """Pad/truncate an ego-local trajectory and return a validity mask."""
    traj = torch.as_tensor(value, dtype=torch.float32)
    if traj.ndim != 2 or traj.shape[-1] != 2:
        raise ValueError(f"Trajectory must be (T,2), got {tuple(traj.shape)}")
    if keep_last:
        traj = traj[-length:]
        pad_before = length - traj.shape[0]
        padded = torch.cat([torch.zeros(pad_before, 2), traj], dim=0)
        mask = torch.cat([torch.zeros(pad_before, dtype=torch.bool), torch.ones(traj.shape[0], dtype=torch.bool)])
    else:
        traj = traj[:length]
        pad_after = length - traj.shape[0]
        padded = torch.cat([traj, torch.zeros(pad_after, 2)], dim=0)
        mask = torch.cat([torch.ones(traj.shape[0], dtype=torch.bool), torch.zeros(pad_after, dtype=torch.bool)])
    return padded, mask


def reduce_camera_grid_tokens(
    tokens: Tensor,
    camera_count: int,
    source_grid: tuple[int, int],
    target_grid: tuple[int, int],
) -> tuple[Tensor, Tensor]:
    """Pool each camera grid independently and return camera/x/y positions."""
    source_height, source_width = source_grid
    expected_tokens = camera_count * source_height * source_width
    if tokens.ndim != 2 or tokens.shape[0] != expected_tokens:
        raise ValueError(
            f"Expected ({expected_tokens}, D) tokens for {camera_count}x{source_grid}, "
            f"got {tuple(tokens.shape)}"
        )
    feature_dim = tokens.shape[-1]
    camera_grid = tokens.view(camera_count, source_height, source_width, feature_dim)
    camera_grid = camera_grid.permute(0, 3, 1, 2)
    pooled = torch.nn.functional.adaptive_avg_pool2d(camera_grid, target_grid)
    pooled = pooled.permute(0, 2, 3, 1).reshape(-1, feature_dim)

    target_height, target_width = target_grid
    camera_position = torch.linspace(-1.0, 1.0, camera_count, device=tokens.device)
    y_position = torch.linspace(-1.0, 1.0, target_height, device=tokens.device)
    x_position = torch.linspace(-1.0, 1.0, target_width, device=tokens.device)
    camera, y_coord, x_coord = torch.meshgrid(
        camera_position, y_position, x_position, indexing="ij"
    )
    positions = torch.stack([camera, x_coord, y_coord], dim=-1).reshape(-1, 3)
    return pooled, positions


def cached_interaction_targets(
    sample: Mapping[str, Any],
    max_agents: int,
    t_pred: int,
) -> dict[str, Tensor]:
    """Convert cached nuScenes boxes and futures to fixed-size query targets."""
    boxes = torch.as_tensor(sample["gt_boxes"], dtype=torch.float32)
    velocities = torch.as_tensor(sample["gt_velocity"], dtype=torch.float32)
    futures = torch.as_tensor(sample["gt_agent_fut_trajs"], dtype=torch.float32).reshape(-1, t_pred, 2)
    future_masks = torch.as_tensor(sample["gt_agent_fut_masks"], dtype=torch.bool)[:, :t_pred]
    names = [str(name) for name in sample["gt_names"]]
    if not (len(names) == boxes.shape[0] == velocities.shape[0] == futures.shape[0]):
        raise ValueError("Cached agent fields have inconsistent lengths")

    valid_flag = torch.as_tensor(
        sample.get("valid_flag", torch.ones(len(names))), dtype=torch.bool
    )
    candidates = [
        index
        for index, name in enumerate(names)
        if bool(valid_flag[index]) and canonical_agent_class(name) is not None
    ]
    candidates.sort(key=lambda index: float(torch.linalg.vector_norm(boxes[index, :2])))
    selected = candidates[:max_agents]
    count = len(selected)

    classes = torch.full((max_agents,), -1, dtype=torch.long)
    centers = torch.zeros(max_agents, 2, dtype=torch.float32)
    target_velocities = torch.zeros(max_agents, 2, dtype=torch.float32)
    target_futures = torch.zeros(max_agents, t_pred, 2, dtype=torch.float32)
    target_future_masks = torch.zeros(max_agents, t_pred, dtype=torch.bool)
    agent_mask = torch.zeros(max_agents, dtype=torch.bool)
    if count:
        index_tensor = torch.tensor(selected, dtype=torch.long)
        classes[:count] = torch.tensor(
            [canonical_agent_class(names[index]) for index in selected], dtype=torch.long
        )
        centers[:count] = boxes[index_tensor, :2]
        target_velocities[:count] = torch.nan_to_num(velocities[index_tensor, :2])
        target_futures[:count] = futures[index_tensor]
        target_future_masks[:count] = future_masks[index_tensor]
        agent_mask[:count] = True
    return {
        "agent_classes": classes,
        "agent_centers": centers,
        "agent_velocities": target_velocities,
        "agent_futures": target_futures,
        "agent_future_mask": target_future_masks,
        "agent_mask": agent_mask,
    }


def rasterize_agent_futures(
    centers: Tensor,
    futures: Tensor,
    future_mask: Tensor,
    agent_mask: Tensor,
    grid_shape: tuple[int, int],
    bounds: tuple[float, float, float, float],
) -> Tensor:
    """Rasterize future agent centers in ego coordinates to binary occupancy."""
    height, width = grid_shape
    x_min, x_max, y_min, y_max = bounds
    if not (x_min < x_max and y_min < y_max):
        raise ValueError("Occupancy bounds must be increasing")
    horizon = futures.shape[1]
    occupancy = torch.zeros(horizon, height, width, dtype=torch.float32)
    absolute = centers[:, None, :] + futures
    for agent_index in agent_mask.nonzero(as_tuple=False).flatten().tolist():
        for time_index in future_mask[agent_index].nonzero(as_tuple=False).flatten().tolist():
            x_coord, y_coord = absolute[agent_index, time_index].tolist()
            column = int((x_coord - x_min) / (x_max - x_min) * width)
            row = int((y_max - y_coord) / (y_max - y_min) * height)
            if 0 <= row < height and 0 <= column < width:
                occupancy[time_index, row, column] = 1.0
    return occupancy


class CachedTrajectoryDataset(Dataset):
    """Read the current cache without embedding evaluation policy in the model."""

    def __init__(
        self,
        data: Mapping[str, Mapping[str, Any]],
        tokens: Sequence[str],
        t_obs: int = 6,
        t_pred: int = 6,
        trajectory_targets: Mapping[str, Any] | None = None,
        trajectory_masks: Mapping[str, Any] | None = None,
        teacher_feature_root: str | Path | None = None,
        teacher_feature_key: str = "deep",
        camera_count: int | None = None,
        source_grid: tuple[int, int] | None = None,
        target_grid: tuple[int, int] | None = None,
        include_interaction_targets: bool = False,
        max_agents: int = 20,
        occupancy_grid: tuple[int, int] = (64, 64),
        occupancy_bounds: tuple[float, float, float, float] = (-30.0, 30.0, -10.0, 50.0),
        road_targets: Mapping[str, Any] | None = None,
        road_target_masks: Mapping[str, Any] | None = None,
    ) -> None:
        self.data = data
        self.tokens = [token for token in tokens if token in data]
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.trajectory_targets = trajectory_targets
        self.trajectory_masks = trajectory_masks
        self.teacher_feature_root = Path(teacher_feature_root) if teacher_feature_root else None
        self.teacher_feature_key = teacher_feature_key
        reduction_values = (camera_count, source_grid, target_grid)
        if any(value is not None for value in reduction_values) and not all(
            value is not None for value in reduction_values
        ):
            raise ValueError("camera_count, source_grid, and target_grid must be set together")
        self.camera_count = camera_count
        self.source_grid = source_grid
        self.target_grid = target_grid
        self.include_interaction_targets = include_interaction_targets
        self.max_agents = max_agents
        self.occupancy_grid = occupancy_grid
        self.occupancy_bounds = occupancy_bounds
        self.road_targets = road_targets
        self.road_target_masks = road_target_masks

    @classmethod
    def from_files(
        cls,
        cache_path: str | Path,
        split_path: str | Path,
        split: str,
        **kwargs: Any,
    ) -> "CachedTrajectoryDataset":
        with Path(cache_path).open("rb") as handle:
            data = pickle.load(handle)
        with Path(split_path).open("r", encoding="utf-8") as handle:
            splits = json.load(handle)
        return cls(data=data, tokens=splits[split], **kwargs)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> CanonicalBatch:
        token = self.tokens[index]
        sample = self.data[token]

        hist_traj, hist_mask = canonical_trajectory(sample["gt_ego_his_trajs"], self.t_obs, keep_last=True)
        if self.trajectory_targets is not None and token in self.trajectory_targets:
            future = torch.as_tensor(self.trajectory_targets[token], dtype=torch.float32).squeeze(0)
        else:
            future = torch.as_tensor(sample["gt_ego_fut_trajs"], dtype=torch.float32)
            if future.shape[0] >= self.t_pred + 1:
                future = future[1:self.t_pred + 1]
        gt_traj, generated_mask = canonical_trajectory(future, self.t_pred, keep_last=False)

        if self.trajectory_masks is not None and token in self.trajectory_masks:
            raw_mask = torch.as_tensor(self.trajectory_masks[token], dtype=torch.bool).squeeze(0)
            if raw_mask.ndim == 2:
                gt_mask = raw_mask[:self.t_pred]
            else:
                gt_mask = raw_mask[:self.t_pred]
        elif "gt_ego_fut_masks" in sample:
            gt_mask = torch.as_tensor(sample["gt_ego_fut_masks"], dtype=torch.bool)[:self.t_pred]
        else:
            gt_mask = generated_mask

        item: CanonicalBatch = {
            "sample_tokens": [token],
            "hist_traj": hist_traj,
            "hist_mask": hist_mask,
            "command": canonical_command(sample["gt_ego_fut_cmd"]),
            "gt_traj": gt_traj,
            "gt_traj_mask": gt_mask,
        }

        if self.include_interaction_targets:
            interaction = cached_interaction_targets(sample, self.max_agents, self.t_pred)
            item.update(interaction)
            occupancy = rasterize_agent_futures(
                interaction["agent_centers"],
                interaction["agent_futures"],
                interaction["agent_future_mask"],
                interaction["agent_mask"],
                self.occupancy_grid,
                self.occupancy_bounds,
            )
            item["future_occupancy"] = occupancy
            item["future_occupancy_mask"] = torch.ones_like(occupancy, dtype=torch.bool)
        if self.road_targets is not None:
            road_target = torch.as_tensor(self.road_targets[token], dtype=torch.float32)
            item["road_target"] = road_target
            if self.road_target_masks is not None:
                item["road_target_mask"] = torch.as_tensor(
                    self.road_target_masks[token], dtype=torch.bool
                )
            else:
                item["road_target_mask"] = torch.ones_like(road_target, dtype=torch.bool)

        if self.teacher_feature_root is not None:
            feature_path = self.teacher_feature_root / f"{token}.pt"
            features = torch.load(feature_path, map_location="cpu", weights_only=True)
            if isinstance(features, Mapping):
                visual = features[self.teacher_feature_key].float()
            elif isinstance(features, Tensor):
                visual = features.float()
            else:
                raise TypeError(f"Unsupported feature payload in {feature_path}")
            if self.camera_count is not None:
                visual, positions = reduce_camera_grid_tokens(
                    visual,
                    self.camera_count,
                    self.source_grid,
                    self.target_grid,
                )
                item["visual_positions"] = positions
            item["visual_tokens"] = visual
            item["visual_mask"] = torch.ones(visual.shape[0], dtype=torch.bool)
        return item


def _pad_sequence(tensors: list[Tensor], value: float = 0.0) -> tuple[Tensor, Tensor]:
    max_length = max(tensor.shape[0] for tensor in tensors)
    output = tensors[0].new_full((len(tensors), max_length, *tensors[0].shape[1:]), value)
    mask = torch.zeros(len(tensors), max_length, dtype=torch.bool)
    for index, tensor in enumerate(tensors):
        output[index, :tensor.shape[0]] = tensor
        mask[index, :tensor.shape[0]] = True
    return output, mask


def collate_canonical(items: list[CanonicalBatch]) -> CanonicalBatch:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    batch: CanonicalBatch = {
        "sample_tokens": [item["sample_tokens"][0] for item in items],
        "hist_traj": torch.stack([item["hist_traj"] for item in items]),
        "hist_mask": torch.stack([item["hist_mask"] for item in items]),
        "command": torch.stack([item["command"] for item in items]),
        "gt_traj": torch.stack([item["gt_traj"] for item in items]),
        "gt_traj_mask": torch.stack([item["gt_traj_mask"] for item in items]),
    }
    if "visual_tokens" in items[0]:
        visual, mask = _pad_sequence([item["visual_tokens"] for item in items])
        batch["visual_tokens"] = visual
        batch["visual_mask"] = mask
        if "visual_positions" in items[0]:
            positions, _ = _pad_sequence([item["visual_positions"] for item in items])
            batch["visual_positions"] = positions
    structured_keys = (
        "road_target",
        "road_target_mask",
        "agent_classes",
        "agent_centers",
        "agent_velocities",
        "agent_futures",
        "agent_future_mask",
        "agent_mask",
        "future_occupancy",
        "future_occupancy_mask",
    )
    for key in structured_keys:
        if key in items[0]:
            batch[key] = torch.stack([item[key] for item in items])
    return batch
