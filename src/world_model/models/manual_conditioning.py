"""Rung-4 manual conditioning: the manual conditions the WORLD-MODEL DYNAMICS via cross-attention.

The agent reads a per-episode manual describing that episode's dynamics; reading must change what
the model *predicts will happen*. So the manual conditions the RSSM **prior** (the imagined
dynamics), NOT the policy — the strongest anti-baking lever (a WM can only "use" the manual by
predicting the right per-episode dynamics from it, which IS reading). The pre-check
(world_model.text_probe) showed a pooled sentence vector loses the ordered recipe, so we attend
over the frozen manual **token embeddings** (order/token-aware), not a single pooled code.

See knowledge/design/rung4-manual-conditioned-agent.md.
"""

import torch
from torch import nn

from world_model.models.rssm import RSSM


class ManualConditioner(nn.Module):
    """Cross-attention: the RSSM deterministic state (query) attends over the manual TOKEN
    embeddings (keys/values) → a context vector that modulates the prior. Token-level + attention
    so the ordered recipe survives (the pooled-vector pre-check failed)."""

    def __init__(self, query_dim: int, text_dim: int, ctx_dim: int = 128, n_heads: int = 4):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.q_proj = nn.Linear(query_dim, ctx_dim)
        self.kv_proj = nn.Linear(text_dim, ctx_dim)
        self.attn = nn.MultiheadAttention(ctx_dim, n_heads, batch_first=True)

    def forward(self, query: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor | None = None):
        """query (B, query_dim); tokens (B, L, text_dim); mask (B, L) bool, True = real token.
        Returns context (B, ctx_dim)."""
        q = self.q_proj(query).unsqueeze(1)  # (B, 1, ctx)
        kv = self.kv_proj(tokens)  # (B, L, ctx)
        # nn.MultiheadAttention key_padding_mask: True = IGNORE, so invert the valid-token mask.
        key_padding_mask = (~mask) if mask is not None else None
        out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        return out.squeeze(1)  # (B, ctx)


class ConditionedRSSM(RSSM):
    """RSSM whose DETERMINISTIC STATE h is conditioned on the manual via cross-attention, so the
    manual propagates to the prior (dynamics), the posterior, the belief (→ the actor sees it at
    act-time), and imagination — all from one injection point. prior_net/post_net keep their
    original dims (the manual is already folded into h). obs_step/img_step take the episode's
    manual token embeddings (+ mask); the rest of the rung-3 machinery reuses the interface."""

    def __init__(self, *, text_dim: int, ctx_dim: int = 128, n_heads: int = 4, **kw):
        super().__init__(**kw)
        self.conditioner = ManualConditioner(self.deter_dim, text_dim, ctx_dim, n_heads)
        self.ctx_to_h = nn.Linear(ctx_dim, self.deter_dim)

    def _deter_cond(self, h, z, prev_action, tokens, mask):
        """GRU step, then fold the manual cross-attention context into the deterministic state."""
        h = self._deter(h, z, prev_action)
        return h + self.ctx_to_h(self.conditioner(h, tokens, mask))

    def deter_noctx(self, state, prev_action):
        """The deterministic state of the *current* step WITHOUT the manual cross-attention
        injection — i.e. the GRU output before ``+ ctx_to_h(conditioner(...))`` is added.

        This is the anti-baking hook for the masked-manual-reconstruction aux loss
        ([[dynalang-2023]] option A / option C): the conditioner folds the manual into ``h`` at
        every step (``_deter_cond``), so the ordinary belief is a trivial copy-through of the
        cross-attention input. Reconstructing the manual from ``deter_noctx`` instead forces the
        signal through the RECURRENT carry — the manual content the GRU has *internalised* from
        prior steps — not the current-step attention shortcut. See
        knowledge/design/rung4-manual-conditioned-agent.md §2–3.
        """
        h, z = state
        return self._deter(h, z, prev_action)

    def obs_step(self, state, prev_action, embed, tokens, mask=None):
        """One step WITH observation. Returns (state, prior, post). h carries the manual."""
        h, z = state
        h = self._deter_cond(h, z, prev_action, tokens, mask)
        prior = self._dist(self.prior_net(h))
        post = self._dist(self.post_net(torch.cat([h, embed], dim=-1)))
        return (h, post.rsample()), prior, post

    def img_step(self, state, prev_action, tokens, mask=None):
        """One imagination step (no obs) → manual-conditioned prior. Returns (state, prior)."""
        h, z = state
        h = self._deter_cond(h, z, prev_action, tokens, mask)
        prior = self._dist(self.prior_net(h))
        return (h, prior.rsample()), prior
