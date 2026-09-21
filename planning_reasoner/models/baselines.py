"""Controlled B0-B4 baselines for testing information contribution."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from planning_reasoner.contracts import CanonicalBatch
from planning_reasoner.models.common import TrajectoryPrediction


class ConstantVelocityBaseline:
    """B0: extrapolate the two most recent valid ego-history points."""

    def __init__(self, horizon: int = 6) -> None:
        self.horizon = horizon

    @torch.no_grad()
    def __call__(self, batch: CanonicalBatch) -> TrajectoryPrediction:
        history = batch["hist_traj"]
        mask = batch["hist_mask"]
        if history.ndim != 3 or history.shape[-1] != 2 or mask.shape != history.shape[:2]:
            raise ValueError("hist_traj and hist_mask must be (B,T,2) and (B,T)")

        indices = torch.arange(history.shape[1], device=history.device).unsqueeze(0)
        valid_indices = torch.where(mask, indices, torch.full_like(indices, -1))
        last_index = valid_indices.max(dim=1).values.clamp_min(0)
        previous_candidates = torch.where(
            mask & (indices < last_index.unsqueeze(1)), indices, torch.full_like(indices, -1)
        )
        previous_index = previous_candidates.max(dim=1).values

        gather_last = last_index[:, None, None].expand(-1, 1, 2)
        last = history.gather(1, gather_last).squeeze(1)
        gather_previous = previous_index.clamp_min(0)[:, None, None].expand(-1, 1, 2)
        previous = history.gather(1, gather_previous).squeeze(1)
        velocity = torch.where((previous_index >= 0).unsqueeze(1), last - previous, torch.zeros_like(last))

        steps = torch.arange(1, self.horizon + 1, device=history.device, dtype=history.dtype)
        mean = last.unsqueeze(1) + steps[None, :, None] * velocity.unsqueeze(1)
        return TrajectoryPrediction(mean=mean, log_scale=torch.zeros_like(mean))


@dataclass(frozen=True)
class UnstructuredPlannerConfig:
    visual_dim: int = 1280
    model_dim: int = 256
    horizon: int = 6
    max_history: int = 12
    max_visual_tokens: int = 384
    position_dim: int = 3
    num_layers: int = 2
    num_heads: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1
    use_visual: bool = True
    use_history: bool = True
    use_command: bool = True


class UnstructuredPlanner(nn.Module):
    """B1-B4 backbone without structured Road/Interaction/Future stages."""

    def __init__(self, config: UnstructuredPlannerConfig) -> None:
        super().__init__()
        if not (config.use_visual or config.use_history or config.use_command):
            raise ValueError("At least one input source must be enabled")
        self.config = config
        self.visual_projection = nn.Linear(config.visual_dim, config.model_dim)
        self.position_projection = (
            nn.Linear(config.position_dim, config.model_dim) if config.position_dim > 0 else None
        )
        self.history_projection = nn.Sequential(
            nn.Linear(2, config.model_dim), nn.GELU(), nn.Linear(config.model_dim, config.model_dim)
        )
        self.command_projection = nn.Sequential(
            nn.Linear(3, config.model_dim), nn.GELU(), nn.Linear(config.model_dim, config.model_dim)
        )
        self.visual_positions = nn.Parameter(torch.randn(config.max_visual_tokens, config.model_dim) * 0.02)
        self.history_positions = nn.Parameter(torch.randn(config.max_history, config.model_dim) * 0.02)
        self.type_embeddings = nn.Parameter(torch.randn(3, config.model_dim) * 0.02)
        self.readout_token = nn.Parameter(torch.randn(1, 1, config.model_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.output_head = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.horizon * 4),
        )

    def forward(self, batch: CanonicalBatch) -> TrajectoryPrediction:
        reference = batch["hist_traj"] if "hist_traj" in batch else batch["visual_tokens"]
        batch_size = reference.shape[0]
        tokens = [self.readout_token.expand(batch_size, -1, -1)]
        masks = [torch.ones(batch_size, 1, dtype=torch.bool, device=reference.device)]

        if self.config.use_visual:
            visual = batch["visual_tokens"]
            if visual.shape[1] > self.config.max_visual_tokens:
                raise ValueError("visual token count exceeds max_visual_tokens")
            embedded = self.visual_projection(visual)
            embedded = embedded + self.visual_positions[:visual.shape[1]] + self.type_embeddings[0]
            if self.position_projection is not None:
                if "visual_positions" not in batch:
                    raise ValueError("visual_positions are required when position_dim > 0")
                embedded = embedded + self.position_projection(batch["visual_positions"])
            tokens.append(embedded)
            masks.append(batch["visual_mask"])
        if self.config.use_history:
            history = batch["hist_traj"]
            if history.shape[1] > self.config.max_history:
                raise ValueError("history length exceeds max_history")
            embedded = self.history_projection(history)
            embedded = embedded + self.history_positions[:history.shape[1]] + self.type_embeddings[1]
            tokens.append(embedded)
            masks.append(batch["hist_mask"])
        if self.config.use_command:
            command = self.command_projection(batch["command"]).unsqueeze(1)
            tokens.append(command + self.type_embeddings[2])
            masks.append(torch.ones(batch_size, 1, dtype=torch.bool, device=reference.device))

        memory = torch.cat(tokens, dim=1)
        valid_mask = torch.cat(masks, dim=1)
        encoded = self.encoder(memory, src_key_padding_mask=~valid_mask)
        raw = self.output_head(encoded[:, 0]).view(batch_size, self.config.horizon, 4)
        return TrajectoryPrediction(mean=raw[..., :2], log_scale=raw[..., 2:].clamp(-5.0, 3.0))
