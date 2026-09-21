"""Fixed-depth progressive Road, Interaction, and Future planner."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from planning_reasoner.contracts import CanonicalBatch
from planning_reasoner.models.common import (
    FutureStageOutput,
    InteractionPrediction,
    InteractionStageOutput,
    ProgressivePlannerOutput,
    RoadStageOutput,
    TrajectoryPrediction,
)
from planning_reasoner.models.baselines import UnstructuredPlanner, UnstructuredPlannerConfig


@dataclass(frozen=True)
class ProgressivePlannerConfig:
    visual_dim: int = 1280
    position_dim: int = 3
    model_dim: int = 256
    t_obs: int = 6
    t_pred: int = 6
    max_visual_tokens: int = 384
    num_heads: int = 4
    ffn_dim: int = 1024
    context_layers: int = 2
    stage_layers: int = 1
    dropout: float = 0.1
    road_channels: int = 3
    road_height: int = 8
    road_width: int = 8
    agent_queries: int = 20
    agent_classes: int = 10
    future_height: int = 8
    future_width: int = 8
    use_motion_prior: bool = False
    zero_initialize_residual: bool = False


class ContextEncoder(nn.Module):
    def __init__(self, config: ProgressivePlannerConfig) -> None:
        super().__init__()
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
        self.history_positions = nn.Parameter(torch.randn(config.t_obs, config.model_dim) * 0.02)
        self.type_embeddings = nn.Parameter(torch.randn(3, config.model_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.context_layers)
        self.norm = nn.LayerNorm(config.model_dim)

    def forward(self, batch: CanonicalBatch) -> tuple[Tensor, Tensor]:
        visual = batch["visual_tokens"]
        history = batch["hist_traj"]
        if visual.shape[1] > self.config.max_visual_tokens:
            raise ValueError("visual token count exceeds max_visual_tokens")
        if history.shape[1] != self.config.t_obs:
            raise ValueError(f"Expected {self.config.t_obs} history steps, got {history.shape[1]}")

        visual_tokens = self.visual_projection(visual)
        visual_tokens = visual_tokens + self.visual_positions[:visual.shape[1]] + self.type_embeddings[0]
        if self.position_projection is not None:
            if "visual_positions" not in batch:
                raise ValueError("visual_positions are required when position_dim > 0")
            visual_tokens = visual_tokens + self.position_projection(batch["visual_positions"])

        history_tokens = self.history_projection(history)
        history_tokens = history_tokens + self.history_positions[:history.shape[1]] + self.type_embeddings[1]
        command_token = self.command_projection(batch["command"]).unsqueeze(1) + self.type_embeddings[2]

        tokens = torch.cat([visual_tokens, history_tokens, command_token], dim=1)
        command_mask = torch.ones(visual.shape[0], 1, dtype=torch.bool, device=visual.device)
        valid_mask = torch.cat([batch["visual_mask"], batch["hist_mask"], command_mask], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~valid_mask)
        return self.norm(encoded), valid_mask


class QueryStage(nn.Module):
    def __init__(self, config: ProgressivePlannerConfig, query_count: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, query_count, config.model_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.stage_layers)
        self.norm = nn.LayerNorm(config.model_dim)

    def forward(self, memory: Tensor, memory_valid_mask: Tensor) -> Tensor:
        queries = self.queries.expand(memory.shape[0], -1, -1)
        decoded = self.decoder(queries, memory, memory_key_padding_mask=~memory_valid_mask)
        return self.norm(decoded)


class TrajectoryExit(nn.Module):
    def __init__(
        self,
        config: ProgressivePlannerConfig,
        residual: bool,
        zero_initialize: bool = False,
    ) -> None:
        super().__init__()
        self.horizon = config.t_pred
        self.residual = residual
        input_dim = config.model_dim + (config.t_pred * 2 if residual else 0)
        self.head = nn.Sequential(
            nn.Linear(input_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.t_pred * 4),
        )
        if residual and zero_initialize:
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, latent: Tensor, previous: TrajectoryPrediction | None = None) -> TrajectoryPrediction:
        pooled = latent.mean(dim=1)
        if self.residual:
            if previous is None:
                raise ValueError("A residual trajectory exit requires the previous prediction")
            pooled = torch.cat([pooled, previous.mean.flatten(1)], dim=1)
        raw = self.head(pooled).view(latent.shape[0], self.horizon, 4)
        mean = raw[..., :2]
        if previous is not None:
            mean = previous.mean + mean
        return TrajectoryPrediction(mean=mean, log_scale=raw[..., 2:].clamp(-5.0, 3.0))


class ProgressiveStructuredPlanner(nn.Module):
    """Full fixed-depth model used before introducing any adaptive router."""

    def __init__(self, config: ProgressivePlannerConfig) -> None:
        super().__init__()
        self.config = config
        self.motion_prior_frozen = False
        self.motion_prior = None
        if config.use_motion_prior:
            self.motion_prior = UnstructuredPlanner(
                UnstructuredPlannerConfig(
                    visual_dim=config.visual_dim,
                    model_dim=config.model_dim,
                    horizon=config.t_pred,
                    max_history=12,
                    max_visual_tokens=config.max_visual_tokens,
                    position_dim=config.position_dim,
                    num_layers=config.context_layers,
                    num_heads=config.num_heads,
                    ffn_dim=config.ffn_dim,
                    dropout=config.dropout,
                    use_visual=False,
                    use_history=True,
                    use_command=True,
                )
            )
        self.context = ContextEncoder(config)

        road_queries = config.road_height * config.road_width
        self.road_stage = QueryStage(config, road_queries)
        self.road_map_head = nn.Linear(config.model_dim, config.road_channels)
        self.road_exit = TrajectoryExit(
            config,
            residual=config.use_motion_prior,
            zero_initialize=config.zero_initialize_residual,
        )

        self.interaction_stage = QueryStage(config, config.agent_queries)
        self.agent_class_head = nn.Linear(config.model_dim, config.agent_classes + 1)
        self.agent_center_head = nn.Linear(config.model_dim, 2)
        self.agent_velocity_head = nn.Linear(config.model_dim, 2)
        self.agent_future_head = nn.Linear(config.model_dim, config.t_pred * 2)
        self.interaction_exit = TrajectoryExit(
            config, residual=True, zero_initialize=config.zero_initialize_residual
        )

        future_queries = config.t_pred * config.future_height * config.future_width
        self.future_stage = QueryStage(config, future_queries)
        self.occupancy_head = nn.Linear(config.model_dim, 1)
        self.future_exit = TrajectoryExit(
            config, residual=True, zero_initialize=config.zero_initialize_residual
        )

    def freeze_motion_prior(self) -> None:
        if self.motion_prior is None:
            raise RuntimeError("The model has no motion prior to freeze")
        self.motion_prior_frozen = True
        self.motion_prior.eval()
        for parameter in self.motion_prior.parameters():
            parameter.requires_grad_(False)

    def _predict_motion_prior(self, batch: CanonicalBatch) -> TrajectoryPrediction | None:
        if self.motion_prior is None:
            return None
        if self.motion_prior_frozen:
            self.motion_prior.eval()
            with torch.no_grad():
                return self.motion_prior(batch)
        return self.motion_prior(batch)

    def forward(
        self,
        batch: CanonicalBatch,
        max_stage: str = "future",
    ) -> ProgressivePlannerOutput:
        if max_stage not in {"road", "interaction", "future"}:
            raise ValueError(f"Unsupported max_stage: {max_stage}")
        memory, memory_mask = self.context(batch)
        motion_prior = self._predict_motion_prior(batch)

        road_latent = self.road_stage(memory, memory_mask)
        batch_size = road_latent.shape[0]
        road_logits = self.road_map_head(road_latent)
        road_logits = road_logits.transpose(1, 2).reshape(
            batch_size,
            self.config.road_channels,
            self.config.road_height,
            self.config.road_width,
        )
        road_trajectory = self.road_exit(road_latent, motion_prior)
        road = RoadStageOutput(road_latent, road_logits, road_trajectory)
        if max_stage == "road":
            return ProgressivePlannerOutput(road=road)

        road_mask = torch.ones(
            batch_size, road_latent.shape[1], dtype=torch.bool, device=road_latent.device
        )
        interaction_memory = torch.cat([memory, road_latent], dim=1)
        interaction_memory_mask = torch.cat([memory_mask, road_mask], dim=1)
        interaction_latent = self.interaction_stage(interaction_memory, interaction_memory_mask)
        interaction_prediction = InteractionPrediction(
            class_logits=self.agent_class_head(interaction_latent),
            centers=self.agent_center_head(interaction_latent),
            velocities=self.agent_velocity_head(interaction_latent),
            futures=self.agent_future_head(interaction_latent).view(
                batch_size, self.config.agent_queries, self.config.t_pred, 2
            ),
        )
        interaction_trajectory = self.interaction_exit(interaction_latent, road_trajectory)
        interaction = InteractionStageOutput(
            interaction_latent, interaction_prediction, interaction_trajectory
        )
        if max_stage == "interaction":
            return ProgressivePlannerOutput(road=road, interaction=interaction)

        interaction_mask = torch.ones(
            batch_size, interaction_latent.shape[1], dtype=torch.bool, device=interaction_latent.device
        )
        future_memory = torch.cat([memory, road_latent, interaction_latent], dim=1)
        future_memory_mask = torch.cat([memory_mask, road_mask, interaction_mask], dim=1)
        future_latent = self.future_stage(future_memory, future_memory_mask)
        occupancy_logits = self.occupancy_head(future_latent).view(
            batch_size,
            self.config.t_pred,
            self.config.future_height,
            self.config.future_width,
        )
        future_trajectory = self.future_exit(future_latent, interaction_trajectory)
        future = FutureStageOutput(future_latent, occupancy_logits, future_trajectory)
        return ProgressivePlannerOutput(road=road, interaction=interaction, future=future)
