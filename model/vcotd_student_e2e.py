"""
VCoTD Student Model — E2E variant.

Key differences from vcotd_student.py:
  - d_enc     : encoder output dim (LightEncoder → 384), used for feat_proj input
  - d_teacher : teacher feature dim (Qwen2-VL → 1280), used for distillation projection
  - BEVDetectionHead : auxiliary head predicting BEV object positions (explicit spatial supervision)
  - compute_total_loss includes L_det
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionEmbedding(nn.Module):
    def __init__(self, d: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(2, d)
        self.fc2 = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T_obs, 2) -> (B, T_obs, d)"""
        return F.relu(self.fc2(F.relu(self.fc1(x))))


class SceneComplexityEstimator(nn.Module):
    def __init__(self, d: int = 256, d_h: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(d, d_h)
        self.fc2 = nn.Linear(d_h, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, seq, d) -> gamma: (B, 1)"""
        return torch.sigmoid(self.fc2(F.relu(self.fc1(feat.mean(dim=1)))))


class TrajectoryDecoder(nn.Module):
    def __init__(self, d: int = 256, T_pred: int = 6):
        super().__init__()
        self.T_pred = T_pred
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, T_pred * 2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d) -> (B, T_pred, 2)"""
        return self.fc2(F.relu(self.fc1(z))).view(-1, self.T_pred, 2)


class BEVDetectionHead(nn.Module):
    """
    Predicts BEV (x, y) positions of the K nearest objects in ego frame.

    Targets are normalised by perception_range → [-1, 1].
    Provides explicit spatial-awareness supervision that mirrors the
    "3D detection" stage of FSDrive's visual CoT reasoning.
    """
    def __init__(self, d: int = 256, K_det: int = 20):
        super().__init__()
        self.K_det = K_det
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, K_det * 2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d) -> (B, K_det, 2)  in normalised coords"""
        return self.fc2(F.relu(self.fc1(z))).view(-1, self.K_det, 2)


class VCoTDStudentE2E(nn.Module):
    """
    VCoTD lightweight student for end-to-end training with LightEncoder.

    Architecture (~4.5M params, same as original student):
      feat_proj     : Linear(d_enc, d)         accepts LightEncoder output
      motion_embed  : MotionEmbedding(d)
      pos_encoding  : Embedding(vis_tokens+T_obs, d)
      encoder_layers: K × TransformerEncoderLayer
      complexity_estimator : SceneComplexityEstimator
      traj_decoder  : TrajectoryDecoder
      det_head      : BEVDetectionHead          auxiliary supervision
      proj_global   : Linear(d, d_teacher)     distillation: global
      proj_dim      : Linear(d, d_teacher)     distillation: detail
    """

    def __init__(
        self,
        d_enc:     int   = 384,    # LightEncoder output channels
        d_teacher: int   = 1280,   # Qwen2-VL teacher feature dim (distillation target)
        d:         int   = 256,
        K:         int   = 4,
        h:         int   = 4,
        d_ff:      int   = 1024,
        T_obs:     int   = 6,
        T_pred:    int   = 6,
        d_h:       int   = 64,
        dropout:   float = 0.1,
        alpha_min: float = 0.3,
        alpha_max: float = 1.0,
        lambda1:   float = 1.0,
        lambda2:   float = 1.0,
        lambda3:   float = 0.5,
        vis_tokens:int   = 16,
        K_det:     int   = 20,
        lambda_det:float = 0.5,
    ):
        super().__init__()
        self.d_enc      = d_enc
        self.d_teacher  = d_teacher
        self.d          = d
        self.T_obs      = T_obs
        self.T_pred     = T_pred
        self.vis_tokens = vis_tokens
        self.alpha_min  = alpha_min
        self.alpha_max  = alpha_max
        self.lambda1    = lambda1
        self.lambda2    = lambda2
        self.lambda3    = lambda3
        self.K_det      = K_det
        self.lambda_det = lambda_det

        # Feature projection: encoder space → student space
        self.feat_proj    = nn.Linear(d_enc, d)
        self.motion_embed = MotionEmbedding(d)
        self.pos_encoding = nn.Embedding(vis_tokens + T_obs, d)

        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d, nhead=h, dim_feedforward=d_ff,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            for _ in range(K)
        ])
        self.final_norm = nn.LayerNorm(d)

        self.complexity_estimator = SceneComplexityEstimator(d, d_h)
        self.traj_decoder         = TrajectoryDecoder(d, T_pred)
        self.det_head             = BEVDetectionHead(d, K_det)

        # Distillation projections: student space → teacher space
        self.proj_global = nn.Linear(d, d_teacher)
        self.proj_dim    = nn.Linear(d, d_teacher)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _sample_vis_tokens(self, feat: torch.Tensor) -> torch.Tensor:
        """Uniform spatial sampling: (B, N, d_enc) → (B, vis_tokens, d_enc)"""
        N = feat.size(1)
        if N <= self.vis_tokens:
            return feat
        idx = torch.linspace(0, N - 1, self.vis_tokens, dtype=torch.long, device=feat.device)
        return feat[:, idx, :]

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        enc_feats: torch.Tensor,   # (B, N', d_enc) from LightEncoder
        hist_traj: torch.Tensor,   # (B, T_obs, 2)
    ):
        # 1. Sample & project visual tokens: encoder space → student space
        vis = self._sample_vis_tokens(enc_feats)   # (B, vis_tokens, d_enc)
        vis = self.feat_proj(vis)                   # (B, vis_tokens, d)

        # 2. Motion embedding
        e = self.motion_embed(hist_traj)            # (B, T_obs, d)

        # 3. Concat + positional encoding
        z = torch.cat([vis, e], dim=1)
        pos = torch.arange(self.vis_tokens + self.T_obs, device=z.device)
        z   = z + self.pos_encoding(pos)

        # 4. Transformer with intermediate feature capture
        student_features = {}
        for i, layer in enumerate(self.encoder_layers):
            z = layer(z)
            if i == 0:
                student_features['shallow'] = z
            elif i == 1:
                student_features['middle'] = z
        z = self.final_norm(z)
        student_features['deep'] = z

        # 5. Scene complexity
        gamma = self.complexity_estimator(z)        # (B, 1)

        # 6. Trajectory decoding
        z_vis     = z[:, :self.vis_tokens, :].mean(dim=1)   # (B, d)
        pred_traj = self.traj_decoder(z_vis)                 # (B, T_pred, 2)

        # 7. Auxiliary BEV detection
        det_pred = self.det_head(z_vis)             # (B, K_det, 2)

        return pred_traj, student_features, gamma, det_pred

    # ── Losses ────────────────────────────────────────────────────────────────

    def compute_distillation_loss(self, student_features, teacher_features):
        """
        Three-level hierarchical distillation in teacher feature space.

        Structural  (shallow): global scene layout alignment
        Relational  (middle) : spatial attention distribution alignment
        Semantic    (deep)   : fine-grained per-token feature alignment
        """
        V = self.vis_tokens
        N = teacher_features['shallow'].size(1)

        # Structural: align global scene representation
        g_T = teacher_features['shallow'].mean(dim=1)
        g_S = student_features['shallow'][:, :V, :].mean(dim=1)
        L_global = F.mse_loss(self.proj_global(g_S), g_T.detach())

        # Relational: align spatial attention energy distributions
        A_T = (teacher_features['middle'] ** 2).sum(dim=-1)
        A_S = (student_features['middle'][:, :V, :] ** 2).sum(dim=-1)
        A_T_n = A_T / (A_T.norm(dim=1, keepdim=True) + 1e-8)
        A_S_n = A_S / (A_S.norm(dim=1, keepdim=True) + 1e-8)
        A_S_up = F.interpolate(
            A_S_n.unsqueeze(1), size=N, mode='linear', align_corners=False,
        ).squeeze(1)
        L_spatial = F.mse_loss(A_S_up, A_T_n.detach())

        # Semantic: align per-token feature details
        F_s = student_features['deep'][:, :V, :]
        F_t = teacher_features['deep'].detach()
        F_s_up = F.interpolate(
            F_s.permute(0, 2, 1), size=N, mode='linear', align_corners=False,
        ).permute(0, 2, 1)
        L_detail = F.mse_loss(self.proj_dim(F_s_up), F_t)

        L_distill = self.lambda1 * L_global + self.lambda2 * L_spatial + self.lambda3 * L_detail
        return L_distill, {
            'L_global':  L_global.item(),
            'L_spatial': L_spatial.item(),
            'L_detail':  L_detail.item(),
        }

    def compute_detection_loss(
        self,
        det_pred: torch.Tensor,   # (B, K_det, 2)
        det_gt:   torch.Tensor,   # (B, K_det, 2) normalised to [-1, 1]
        det_mask: torch.Tensor,   # (B, K_det) bool
    ) -> torch.Tensor:
        if det_mask.sum() == 0:
            return torch.tensor(0.0, device=det_pred.device)
        mask = det_mask.unsqueeze(-1).expand_as(det_pred)
        return F.mse_loss(det_pred[mask], det_gt[mask])

    def compute_total_loss(
        self,
        pred_traj:        torch.Tensor,
        gt_traj:          torch.Tensor,
        student_features: dict,
        teacher_features: dict,
        gamma:            torch.Tensor,
        det_pred:         torch.Tensor = None,
        det_gt:           torch.Tensor = None,
        det_mask:         torch.Tensor = None,
    ):
        L_pred    = F.mse_loss(pred_traj, gt_traj)
        L_distill, dist_comps = self.compute_distillation_loss(student_features, teacher_features)
        alpha   = self.alpha_min + (self.alpha_max - self.alpha_min) * gamma.mean()
        L_total = L_pred + alpha * L_distill

        L_det = torch.tensor(0.0, device=pred_traj.device)
        if det_pred is not None and det_gt is not None and det_mask is not None:
            L_det   = self.compute_detection_loss(det_pred, det_gt, det_mask)
            L_total = L_total + self.lambda_det * L_det

        return L_total, {
            'total':   L_total.item(),
            'pred':    L_pred.item(),
            'distill': L_distill.item(),
            'det':     L_det.item(),
            'alpha':   alpha.item(),
            **dist_comps,
        }

    @torch.no_grad()
    def predict(self, enc_feats: torch.Tensor, hist_traj: torch.Tensor) -> torch.Tensor:
        pred_traj, *_ = self.forward(enc_feats, hist_traj)
        return pred_traj
