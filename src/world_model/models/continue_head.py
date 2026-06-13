"""Continue predictor: belief state -> P(episode continues).

Dreamer's omitted-by-us component (exp 0017 → 0018). Imagination has no environment
termination, so without this the actor farms reward past the (unmodeled) episode
end. c(s) ∈ [0,1] is trained on real done flags; imagined returns are discounted by
the cumulative product of predicted continue probs, zeroing post-termination reward.
"""

import torch
from torch import nn


class ContinueHead(nn.Module):
    def __init__(self, state_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """(B, state_dim) -> (B,) logit for P(continue)."""
        return self.net(state).squeeze(-1)
