"""Actor: belief state -> action logits (the Mode-1 reactive policy).

The amortized counterpart to CEM planning (Mode-2): one forward pass instead of
hundreds of imagined rollouts per action. Distilled from the planner (exp 0016) or
trained in imagination (Dreamer-style, later). The base for a hierarchical actor —
see knowledge/concepts/hierarchy-and-credit.md.
"""

import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, state_dim: int = 256, num_actions: int = 7, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """(B, state_dim) -> (B, num_actions) logits."""
        return self.net(state)
