"""Action-conditioned latent dynamics predictor.

Given the current latent state and an action, predict the next latent state:

    z_{t+1} ≈ f(z_t, a_t)

This is the minimal "world model" core: trained against the encoder's embedding of the
actually observed next frame (JEPA-style latent regression). Stop-gradient / EMA target
encoders, multi-step rollout losses, and stochastic latents are deliberate later
experiments — see knowledge/concepts/jepa.md for why naive latent regression can collapse.
"""

import torch
from torch import nn


class LatentDynamicsPredictor(nn.Module):
    """MLP forward model over latents with discrete-action conditioning."""

    def __init__(self, latent_dim: int = 128, num_actions: int = 7, hidden_dim: int = 256):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """(B, latent_dim), (B,) int64 -> (B, latent_dim) predicted next latent."""
        a = self.action_embed(action)
        return self.net(torch.cat([latent, a], dim=-1))
