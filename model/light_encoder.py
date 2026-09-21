"""
Lightweight visual encoder for VCoTD E2E inference (no Qwen2-VL needed).

Uses EfficientViT-M4 (~8.4M params) from timm.
Input:  (B, 6, 3, H, W)  surround-view images, values in [0, 1]
Output: (B, 6*H'*W', d_enc)  visual token sequence
        For M4 at 224x224: (B, 96, 384)
"""

import torch
import torch.nn as nn
import timm

CAMERA_TYPES = [
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
]

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class LightEncoder(nn.Module):
    """
    EfficientViT-M4 surround-view encoder.

    Processes N_cam camera images in parallel through the backbone,
    flattens each image's spatial feature map into tokens, then
    concatenates all cameras into a single sequence.
    """

    def __init__(
        self,
        model_name: str = 'efficientvit_m4',
        image_size: int = 224,
        pretrained: bool = True,
        freeze: bool = False,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[-1],          # last stage only
        )

        # Derive output shape from a dummy forward (no grad, no data)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            feat  = self.backbone(dummy)[-1]   # (1, d_enc, H', W')
        self.d_enc          = feat.shape[1]    # 384 for M4
        self.H_out          = feat.shape[2]    # 4   for M4 @224
        self.W_out          = feat.shape[3]    # 4   for M4 @224
        self.tokens_per_cam = self.H_out * self.W_out   # 16
        self.total_tokens   = 6 * self.tokens_per_cam   # 96

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # ImageNet normalisation (registered as buffers → moved with .to(device))
        self.register_buffer(
            'pixel_mean',
            torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            'pixel_std',
            torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, N_cam, 3, H, W)  float32 in [0, 1]
        Returns:
            feats:  (B, N_cam * H'*W', d_enc)
        """
        B, N_cam, C, H, W = images.shape
        imgs = images.view(B * N_cam, C, H, W)

        # ImageNet normalisation (vectorised, no Python loop)
        imgs = (imgs - self.pixel_mean) / self.pixel_std

        feat = self.backbone(imgs)[-1]               # (B*N_cam, d_enc, H', W')
        feat = feat.flatten(2).transpose(1, 2)       # (B*N_cam, H'*W', d_enc)
        feat = feat.reshape(B, N_cam * self.tokens_per_cam, self.d_enc)
        return feat
