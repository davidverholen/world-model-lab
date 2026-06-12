"""Reward head: (latent, action) -> predicted reward.

Lets the agent plan in latent space without ever seeing pixels at planning time:
imagined rollouts through the dynamics predictor are scored by this head. Trained
jointly with encoder/predictor; nonzero rewards are upweighted because MiniGrid
rewards are sparse (goal-reaching transitions are <1% of random data).
"""

import torch
from torch import nn


class RewardHead(nn.Module):
    def __init__(self, latent_dim: int = 128, num_actions: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """(B, latent_dim), (B,) int64 -> (B,) predicted reward."""
        a = self.action_embed(action)
        return self.net(torch.cat([latent, a], dim=-1)).squeeze(-1)


def reward_loss(
    pred: torch.Tensor, target: torch.Tensor, pos_weight: float = 100.0
) -> torch.Tensor:
    """MSE with nonzero-reward transitions upweighted (sparse-reward imbalance)."""
    weight = 1.0 + pos_weight * (target.abs() > 1e-6).float()
    return (weight * (pred - target) ** 2).mean()
