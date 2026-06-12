"""Image encoder: observation -> latent state.

The encoder maps pixels into the latent space where all prediction happens.
This is the shared starting point for both research directions we want to compare
(see knowledge/concepts/world-models.md):

- JEPA-style: predict future *latents*, never reconstruct pixels.
- Dreamer-style: latent dynamics + decoder, trained with reconstruction.
"""

import torch
from torch import nn


class ConvEncoder(nn.Module):
    """Small CNN encoder for low-resolution game observations.

    Works on any input size >= 32x32; spatial dims are collapsed with adaptive pooling
    so the same encoder runs on MiniGrid at different tile sizes.
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 128, width: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width * 2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(width * 2, width * 4, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(2),
        )
        self.head = nn.Linear(width * 4 * 2 * 2, latent_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) float in [0, 1] -> (B, latent_dim)."""
        feat = self.conv(obs).flatten(1)
        return self.head(feat)
