"""Actor: belief state -> action logits (the Mode-1 reactive policy).

The amortized counterpart to CEM planning (Mode-2): one forward pass instead of
hundreds of imagined rollouts per action. Distilled from the planner (exp 0016) or
trained in imagination (Dreamer-style, later). The base for a hierarchical actor —
see knowledge/concepts/hierarchy-and-credit.md.
"""

import torch
from torch import nn

from world_model.models.manual_conditioning import ManualConditioner


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


class ConditionedActor(nn.Module):
    """OPTIONAL manual-conditioned actor (rung-4 §6.2 fallback) — flag-gated, OFF by default.

    ANTI-BAKING WARNING. Conditioning the policy directly is the baking-RISKIER choice the
    rung-4 design deliberately AVOIDS by default (knowledge/design/rung4-manual-conditioned-agent.md
    §2, §3d): a text→action policy can learn ``token X → action Y`` shortcuts that bypass reading.
    The design pre-authorizes this path ONLY as a fallback "if planning through the m-conditioned WM
    is too weak, and re-verify the swap test if we do" (§6.2). The guard is UNCHANGED: the EXISTING
    swap test (swap_follow + swapped≪correct on HELD-OUT manuals) in ``evaluate_rtfm`` remains the
    acceptance gate. The actor gets NO privileged channel — only the same frozen manual TOKEN
    embeddings the world model already sees (no manual_facts, no displayed gesture).

    Mechanism: the belief (query) cross-attends over the manual token embeddings (reusing
    ``ManualConditioner``) → a context vector; the base ``Actor`` then runs on ``concat(belief,
    ctx)``. The base Actor is reused unchanged (just with a wider input), so the non-conditioned
    path is untouched.
    """

    def __init__(
        self,
        belief_dim: int,
        text_dim: int,
        num_actions: int = 7,
        ctx_dim: int = 128,
        n_heads: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.conditioner = ManualConditioner(belief_dim, text_dim, ctx_dim, n_heads)
        self.actor = Actor(belief_dim + ctx_dim, num_actions, hidden_dim)

    def forward(
        self,
        belief: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """belief (B, belief_dim); tokens (B, L, text_dim); mask (B, L) bool, True = real token.
        Returns (B, num_actions) logits."""
        ctx = self.conditioner(belief, tokens, mask)
        return self.actor(torch.cat([belief, ctx], dim=-1))
