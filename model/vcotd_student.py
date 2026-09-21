"""Paper-aligned VCoTD student and hierarchical feature distillation.

The paper path uses one globally pooled visual token plus ego-motion tokens.
An explicit ``uniform`` aggregation mode is retained for the later multi-token
prototype, but it is not the architecture described by VCoT.pdf.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class VCoTDStudentConfig:
    d_vlm: int = 1280
    d: int = 256
    K: int = 4
    h: int = 4
    d_ff: int = 1024
    T_obs: int = 6
    T_pred: int = 6
    d_h: int = 64
    dropout: float = 0.1
    alpha_min: float = 0.3
    alpha_max: float = 1.0
    lambda1: float = 1.0
    lambda2: float = 1.0
    lambda3: float = 0.5
    visual_aggregation: str = "gap"
    vis_tokens: int = 1


class MotionEmbedding(nn.Module):
    def __init__(self, d: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, d)
        self.fc2 = nn.Linear(d, d)

    def forward(self, trajectory: Tensor) -> Tensor:
        return F.relu(self.fc2(F.relu(self.fc1(trajectory))))


class SceneComplexityEstimator(nn.Module):
    """Return a learned distillation-strength gate in [0, 1]."""

    def __init__(self, d: int = 256, d_h: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d, d_h)
        self.fc2 = nn.Linear(d_h, 1)

    def forward(self, features: Tensor) -> Tensor:
        return torch.sigmoid(self.fc2(F.relu(self.fc1(features.mean(dim=1)))))


class TrajectoryDecoder(nn.Module):
    def __init__(self, d: int = 256, T_pred: int = 6) -> None:
        super().__init__()
        self.T_pred = T_pred
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, T_pred * 2)

    def forward(self, visual_token: Tensor) -> Tensor:
        return self.fc2(F.relu(self.fc1(visual_token))).view(-1, self.T_pred, 2)


class VCoTDStudentModel(nn.Module):
    """Four-layer lightweight Transformer used by the VCoTD paper."""

    def __init__(
        self,
        d_vlm: int = 1280,
        d: int = 256,
        K: int = 4,
        h: int = 4,
        d_ff: int = 1024,
        T_obs: int = 6,
        T_pred: int = 6,
        d_h: int = 64,
        dropout: float = 0.1,
        alpha_min: float = 0.3,
        alpha_max: float = 1.0,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        lambda3: float = 0.5,
        visual_aggregation: str = "gap",
        vis_tokens: int = 1,
    ) -> None:
        super().__init__()
        if visual_aggregation not in {"gap", "uniform"}:
            raise ValueError("visual_aggregation must be 'gap' or 'uniform'")
        if visual_aggregation == "gap" and vis_tokens != 1:
            raise ValueError("The paper-aligned GAP path produces exactly one visual token")
        if K < 2:
            raise ValueError("Hierarchical distillation requires at least two Transformer layers")
        if not 0.0 <= alpha_min <= alpha_max:
            raise ValueError("Expected 0 <= alpha_min <= alpha_max")

        self.config = VCoTDStudentConfig(
            d_vlm=d_vlm,
            d=d,
            K=K,
            h=h,
            d_ff=d_ff,
            T_obs=T_obs,
            T_pred=T_pred,
            d_h=d_h,
            dropout=dropout,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            lambda1=lambda1,
            lambda2=lambda2,
            lambda3=lambda3,
            visual_aggregation=visual_aggregation,
            vis_tokens=vis_tokens,
        )
        self.d_vlm = d_vlm
        self.d = d
        self.K = K
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.visual_aggregation = visual_aggregation
        self.vis_tokens = vis_tokens
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.feat_proj = nn.Linear(d_vlm, d)
        self.motion_embed = MotionEmbedding(d)
        self.pos_encoding = nn.Embedding(vis_tokens + T_obs, d)
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=h,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(K)
            ]
        )
        self.complexity_estimator = SceneComplexityEstimator(d, d_h)
        self.traj_decoder = TrajectoryDecoder(d, T_pred)
        self.proj_global = nn.Linear(d, d_vlm)
        self.proj_dim = nn.Linear(d, d_vlm)
        self.final_norm = nn.LayerNorm(d)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def model_config(self) -> dict[str, object]:
        return asdict(self.config)

    @staticmethod
    def infer_config_from_state_dict(
        state_dict: Mapping[str, Tensor],
        T_obs: int = 6,
    ) -> dict[str, object]:
        """Recover shape-critical fields from legacy checkpoints."""
        visual_slots = int(state_dict["pos_encoding.weight"].shape[0]) - T_obs
        layer_indices = {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("encoder_layers.") and key.split(".")[1].isdigit()
        }
        return {
            "d_vlm": int(state_dict["feat_proj.weight"].shape[1]),
            "d": int(state_dict["feat_proj.weight"].shape[0]),
            "K": len(layer_indices),
            "d_ff": int(state_dict["encoder_layers.0.linear1.weight"].shape[0]),
            "T_obs": T_obs,
            "T_pred": int(state_dict["traj_decoder.fc2.weight"].shape[0] // 2),
            "d_h": int(state_dict["complexity_estimator.fc1.weight"].shape[0]),
            "visual_aggregation": "gap" if visual_slots == 1 else "uniform",
            "vis_tokens": visual_slots,
        }

    def _prepare_visual_tokens(
        self,
        features: Tensor,
        feature_mask: Tensor | None = None,
    ) -> Tensor:
        if features.ndim != 3 or features.shape[-1] != self.d_vlm:
            raise ValueError(
                f"Expected visual features (B,N,{self.d_vlm}), got {tuple(features.shape)}"
            )
        if feature_mask is not None:
            if feature_mask.shape != features.shape[:2]:
                raise ValueError(
                    "feature_mask must match the feature batch and token dimensions"
                )
            feature_mask = feature_mask.to(device=features.device, dtype=torch.bool)

        if self.visual_aggregation == "gap":
            return self._masked_token_mean(features, feature_mask).unsqueeze(1)

        token_count = features.shape[1]
        if feature_mask is None and token_count == self.vis_tokens:
            return features
        if feature_mask is None and token_count > self.vis_tokens:
            indices = torch.linspace(
                0, token_count - 1, self.vis_tokens, device=features.device
            ).long()
            return features[:, indices]
        if feature_mask is None:
            padding = features.new_zeros(
                features.shape[0], self.vis_tokens - token_count, features.shape[-1]
            )
            return torch.cat([features, padding], dim=1)

        sampled = []
        for sample_features, sample_mask in zip(features, feature_mask):
            valid = sample_features[sample_mask]
            if valid.shape[0] == 0:
                valid = sample_features.new_zeros(1, self.d_vlm)
            if valid.shape[0] >= self.vis_tokens:
                indices = torch.linspace(
                    0, valid.shape[0] - 1, self.vis_tokens, device=features.device
                ).long()
                valid = valid[indices]
            else:
                valid = torch.cat(
                    [
                        valid,
                        valid.new_zeros(self.vis_tokens - valid.shape[0], self.d_vlm),
                    ],
                    dim=0,
                )
            sampled.append(valid)
        return torch.stack(sampled)

    def _encode(
        self,
        feat_deep_vlm: Tensor,
        hist_traj: Tensor,
        feature_mask: Tensor | None = None,
        retain_intermediate: bool = True,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if hist_traj.shape[1:] != (self.T_obs, 2):
            raise ValueError(
                f"Expected historical trajectory (B,{self.T_obs},2), "
                f"got {tuple(hist_traj.shape)}"
            )
        visual = self.feat_proj(
            self._prepare_visual_tokens(feat_deep_vlm, feature_mask)
        )
        motion = self.motion_embed(hist_traj)
        hidden = torch.cat([visual, motion], dim=1)
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        hidden = hidden + self.pos_encoding(positions)

        student_features: dict[str, Tensor] = {}
        for index, layer in enumerate(self.encoder_layers):
            hidden = layer(hidden)
            if retain_intermediate and index == 0:
                student_features["shallow"] = hidden
            if retain_intermediate and index == 1:
                student_features["middle"] = hidden
        hidden = self.final_norm(hidden)
        if retain_intermediate:
            student_features["deep"] = hidden
        return hidden, student_features

    def forward(
        self,
        feat_deep_vlm: Tensor,
        hist_traj: Tensor,
        feature_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        hidden, student_features = self._encode(
            feat_deep_vlm,
            hist_traj,
            feature_mask=feature_mask,
            retain_intermediate=True,
        )

        gamma = self.complexity_estimator(hidden)
        prediction = self.traj_decoder(hidden[:, 0])
        return prediction, student_features, gamma

    @staticmethod
    def _masked_token_mean(features: Tensor, mask: Tensor | None) -> Tensor:
        if mask is None:
            return features.mean(dim=1)
        weights = mask.to(device=features.device, dtype=features.dtype).unsqueeze(-1)
        return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def compute_distillation_terms(
        self,
        student_features: Mapping[str, Tensor],
        teacher_features: Mapping[str, Tensor],
        teacher_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return one global/spatial/detail loss value per sample."""
        required = {"shallow", "middle", "deep"}
        if required.difference(student_features) or required.difference(teacher_features):
            raise ValueError("Student and teacher features require shallow/middle/deep levels")
        teacher_shape = teacher_features["deep"].shape[:2]
        if any(teacher_features[level].shape[:2] != teacher_shape for level in required):
            raise ValueError("Teacher feature levels must share batch and token dimensions")
        if teacher_mask is not None:
            if teacher_mask.shape != teacher_shape:
                raise ValueError("teacher_mask must match teacher batch and token dimensions")
            teacher_mask = teacher_mask.to(
                device=teacher_features["deep"].device, dtype=torch.bool
            )
            if bool(teacher_mask.all()):
                teacher_mask = None

        teacher_global = self._masked_token_mean(teacher_features["shallow"], teacher_mask)
        student_global = student_features["shallow"].mean(dim=1)
        global_loss = (
            self.proj_global(student_global) - teacher_global.detach()
        ).pow(2).sum(dim=1)

        teacher_attention = teacher_features["middle"].pow(2).sum(dim=-1)
        student_attention = student_features["middle"].pow(2).sum(dim=-1)
        if teacher_mask is None:
            student_attention = F.interpolate(
                student_attention.unsqueeze(1),
                size=teacher_attention.shape[1],
                mode="linear",
                align_corners=False,
            ).squeeze(1)
            teacher_attention = F.normalize(
                teacher_attention, p=2, dim=1, eps=1e-8
            )
            student_attention = F.normalize(
                student_attention, p=2, dim=1, eps=1e-8
            )
            spatial_loss = (
                student_attention - teacher_attention.detach()
            ).pow(2).sum(dim=1)

            student_deep = F.interpolate(
                student_features["deep"].transpose(1, 2),
                size=teacher_features["deep"].shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            student_deep = self.proj_dim(student_deep)
            detail_loss = (
                student_deep - teacher_features["deep"].detach()
            ).pow(2).mean(dim=(1, 2))
        else:
            mask = teacher_mask.to(
                device=teacher_attention.device, dtype=torch.bool
            )
            spatial_values = []
            detail_values = []
            for index in range(teacher_attention.shape[0]):
                valid_count = int(mask[index].sum())
                if valid_count == 0:
                    spatial_values.append(student_attention.new_zeros(()))
                    detail_values.append(student_attention.new_zeros(()))
                    continue

                aligned_attention = F.interpolate(
                    student_attention[index][None, None],
                    size=valid_count,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
                valid_teacher_attention = teacher_attention[index][mask[index]]
                aligned_attention = F.normalize(
                    aligned_attention, p=2, dim=0, eps=1e-8
                )
                valid_teacher_attention = F.normalize(
                    valid_teacher_attention, p=2, dim=0, eps=1e-8
                )
                spatial_values.append(
                    (aligned_attention - valid_teacher_attention.detach()).pow(2).sum()
                )

                aligned_deep = F.interpolate(
                    student_features["deep"][index].transpose(0, 1)[None],
                    size=valid_count,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).transpose(0, 1)
                aligned_deep = self.proj_dim(aligned_deep)
                valid_teacher_deep = teacher_features["deep"][index][mask[index]]
                detail_values.append(
                    (aligned_deep - valid_teacher_deep.detach()).pow(2).mean()
                )
            spatial_loss = torch.stack(spatial_values)
            detail_loss = torch.stack(detail_values)
        return {
            "L_global": global_loss,
            "L_spatial": spatial_loss,
            "L_detail": detail_loss,
        }

    def compute_distillation_loss(
        self,
        student_features: Mapping[str, Tensor],
        teacher_features: Mapping[str, Tensor],
        teacher_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        terms = self.compute_distillation_terms(
            student_features, teacher_features, teacher_mask
        )
        per_sample = (
            self.lambda1 * terms["L_global"]
            + self.lambda2 * terms["L_spatial"]
            + self.lambda3 * terms["L_detail"]
        )
        return per_sample.mean(), {
            name: float(value.mean().detach()) for name, value in terms.items()
        }

    @staticmethod
    def trajectory_loss_per_sample(
        prediction: Tensor,
        target: Tensor,
        trajectory_mask: Tensor | None = None,
    ) -> Tensor:
        """Equation (29): mean Euclidean displacement over future steps."""
        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError("prediction and target must have shape (B,T,2)")
        distance = torch.linalg.vector_norm(prediction - target, dim=-1)
        if trajectory_mask is None:
            return distance.mean(dim=1)
        mask = trajectory_mask.to(device=distance.device, dtype=torch.bool)
        if mask.ndim == 3:
            mask = mask.all(dim=-1)
        if mask.shape != distance.shape:
            raise ValueError("trajectory_mask must have shape (B,T) or (B,T,2)")
        weights = mask.to(distance.dtype)
        return (distance * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def compute_total_loss(
        self,
        pred_traj: Tensor,
        gt_traj: Tensor,
        student_features: Mapping[str, Tensor],
        teacher_features: Mapping[str, Tensor],
        gamma: Tensor,
        teacher_mask: Tensor | None = None,
        trajectory_mask: Tensor | None = None,
        adaptive: bool = True,
        fixed_alpha: float = 1.0,
    ) -> tuple[Tensor, dict[str, float]]:
        terms = self.compute_distillation_terms(
            student_features, teacher_features, teacher_mask
        )
        distill_per_sample = (
            self.lambda1 * terms["L_global"]
            + self.lambda2 * terms["L_spatial"]
            + self.lambda3 * terms["L_detail"]
        )
        prediction_per_sample = self.trajectory_loss_per_sample(
            pred_traj, gt_traj, trajectory_mask
        )
        if adaptive:
            alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * gamma.squeeze(-1)
        else:
            alpha = torch.full_like(prediction_per_sample, fixed_alpha)
        total = (prediction_per_sample + alpha * distill_per_sample).mean()
        return total, {
            "total": float(total.detach()),
            "pred": float(prediction_per_sample.mean().detach()),
            "distill": float(distill_per_sample.mean().detach()),
            "alpha": float(alpha.mean().detach()),
            "alpha_min_batch": float(alpha.min().detach()),
            "alpha_max_batch": float(alpha.max().detach()),
            **{name: float(value.mean().detach()) for name, value in terms.items()},
        }

    @torch.no_grad()
    def predict(
        self,
        feat_deep_vlm: Tensor,
        hist_traj: Tensor,
        feature_mask: Tensor | None = None,
    ) -> Tensor:
        """Inference-only path without feature taps or the adaptive loss gate."""
        hidden, _ = self._encode(
            feat_deep_vlm,
            hist_traj,
            feature_mask=feature_mask,
            retain_intermediate=False,
        )
        return self.traj_decoder(hidden[:, 0])


def build_student_from_checkpoint(checkpoint: Mapping[str, Any]) -> VCoTDStudentModel:
    """Construct a student from current or shape-compatible legacy checkpoints."""
    state_dict = checkpoint["model_state"]
    saved_config = checkpoint.get("model_config")
    if saved_config is None:
        raw_args = checkpoint.get("args") or {}
        saved_config = raw_args
    saved_config = (
        vars(saved_config)
        if hasattr(saved_config, "__dict__")
        else dict(saved_config)
    )
    has_explicit_aggregation = "visual_aggregation" in saved_config
    allowed = {field.name for field in fields(VCoTDStudentConfig)}
    config = {key: value for key, value in dict(saved_config).items() if key in allowed}
    inferred = VCoTDStudentModel.infer_config_from_state_dict(
        state_dict, T_obs=int(config.get("T_obs", 6))
    )
    if not has_explicit_aggregation:
        # Pre-reconciliation checkpoints used uniform sampling even when V=1.
        inferred["visual_aggregation"] = "uniform"
    config.update(inferred)
    model = VCoTDStudentModel(**config)
    model.load_state_dict(state_dict, strict=True)
    return model
