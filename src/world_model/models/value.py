"""Value head: belief state -> expected discounted return-to-go.

The fix for exp 0005's horizon problem: receding-horizon planning can only see
rewards inside the imagination window, but DoorKey's reward sits ~25+ actions deep.
V(s) summarizes everything beyond the horizon, so the planner score becomes

    sum_t gamma^t * r_hat(s_t, a_t)  +  gamma^H * V(s_H)

Trained on Monte-Carlo return-to-go from real episodes (computed in the replay
buffer) — denser than the sparse reward itself, since every step of a successful
episode carries a nonzero target.
"""

import torch
from torch import nn


class ValueHead(nn.Module):
    def __init__(self, state_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """(B, state_dim) -> (B,) predicted return-to-go."""
        return self.net(state).squeeze(-1)


def value_loss(pred: torch.Tensor, target: torch.Tensor, pos_weight: float = 20.0) -> torch.Tensor:
    """Weighted MSE; positive returns upweighted (success episodes are the minority)."""
    weight = 1.0 + pos_weight * (target.abs() > 1e-6).float()
    return (weight * (pred - target) ** 2).mean()
