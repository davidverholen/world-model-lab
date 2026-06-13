"""Frozen DINOv2 encoder for rung 3 (Crafter) — see experiments/0022 (validated) + the
frozen-encoder-lean memory.

DINO-WM-style recipe (arxiv:2411.04983): a FROZEN DINOv2 vision backbone supplies
game-agnostic features; only a small dynamics model is learned on top. Frozen ⇒ it cannot
collapse (so SIGReg is dropped), embeddings are cacheable (encode once at collection), and it
is immune to primacy/drift by construction ([[retention]]). Drop-in for ConvEncoder: consumes
CHW float32 obs in [0, 1] (our env-wrapper format) and exposes `.latent_dim`.
"""

import torch
from torch import nn

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class FrozenDinoEncoder(nn.Module):
    """Frozen DINOv2 backbone → flat embedding. pool: 'cls' | 'patch_mean' | 'cls+patch'."""

    def __init__(self, model: str = "dinov2_vits14", img_size: int = 98, pool: str = "cls"):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", model, verbose=False)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.img_size = img_size  # must be divisible by the patch size (14)
        self.pool = pool
        dim = self.backbone.embed_dim  # 384 for vits14
        self.latent_dim = 2 * dim if pool == "cls+patch" else dim
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):  # keep the frozen backbone in eval always
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, 3, H, W) float in [0, 1] → (B, latent_dim) embedding."""
        x = nn.functional.interpolate(obs, size=self.img_size, mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std
        feats = self.backbone.forward_features(x)
        cls = feats["x_norm_clstoken"]
        if self.pool == "cls":
            return cls
        patch = feats["x_norm_patchtokens"].mean(dim=1)
        if self.pool == "patch_mean":
            return patch
        return torch.cat([cls, patch], dim=-1)
