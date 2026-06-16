"""Two-hot distributional value head (DreamerV3) — bounded value learning.

The MSE critic regresses an unbounded scalar, so on dense/large-scale rewards (Crafter) the
imagined value blows up (exp 0023 motivation: imagined_return → 21, critic_loss → 35). A
distributional critic predicts a categorical over a FIXED grid of symlog-spaced bins and is
trained with a two-hot cross-entropy target, so its output is bounded by construction and its
gradient scale is decoupled from the target magnitude. `forward` returns the scalar (symexp)
expectation, a drop-in for ValueHead in the imagination actor-critic. See
knowledge/papers/dreamerv3-2023.md.
"""

import torch
import torch.nn.functional as F
from torch import nn


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(x.abs())


class TwoHotValueHead(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_bins: int = 255,
        vmin: float = -20.0,
        vmax: float = 20.0,
        hidden: int = 256,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.SiLU(), nn.Linear(hidden, num_bins)
        )
        self.register_buffer("bins", torch.linspace(vmin, vmax, num_bins))
        self.vmin, self.vmax = vmin, vmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scalar value = symexp(E[bin]) under the predicted categorical."""
        probs = F.softmax(self.net(x), dim=-1)
        return symexp((probs * self.bins).sum(-1))

    def _twohot(self, y: torch.Tensor) -> torch.Tensor:
        """Two-hot encode symlog-space targets y (...,) → (..., num_bins)."""
        y = y.clamp(self.vmin, self.vmax)
        below = torch.clamp((self.bins <= y[..., None]).long().sum(-1) - 1, 0, len(self.bins) - 1)
        above = torch.clamp(below + 1, 0, len(self.bins) - 1)
        b, a = self.bins[below], self.bins[above]
        w_above = torch.where(a > b, (y - b) / (a - b), torch.zeros_like(y))
        target = torch.zeros(*y.shape, len(self.bins), device=y.device)
        target.scatter_add_(-1, below.unsqueeze(-1), (1 - w_above).unsqueeze(-1))
        target.scatter_add_(-1, above.unsqueeze(-1), w_above.unsqueeze(-1))
        return target

    def twohot_loss(self, x: torch.Tensor, target_scalar: torch.Tensor) -> torch.Tensor:
        """Cross-entropy between the predicted categorical and two-hot(symlog(target))."""
        logits = self.net(x)
        tgt = self._twohot(symlog(target_scalar.detach()))
        return -(tgt * F.log_softmax(logits, dim=-1)).sum(-1).mean()


def _twohot_encode(y: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Two-hot encode (already symlog-space, clamped) y (...,) → (..., len(bins))."""
    y = y.clamp(bins[0], bins[-1])
    below = torch.clamp((bins <= y[..., None]).long().sum(-1) - 1, 0, len(bins) - 1)
    above = torch.clamp(below + 1, 0, len(bins) - 1)
    b, a = bins[below], bins[above]
    w_above = torch.where(a > b, (y - b) / (a - b), torch.zeros_like(y))
    target = torch.zeros(*y.shape, len(bins), device=y.device)
    target.scatter_add_(-1, below.unsqueeze(-1), (1 - w_above).unsqueeze(-1))
    target.scatter_add_(-1, above.unsqueeze(-1), w_above.unsqueeze(-1))
    return target


class TwoHotRewardHead(nn.Module):
    """Action-conditioned two-hot reward predictor (DreamerV3-style, bounded). Drop-in for
    RewardHead: forward(latent, action) returns the symexp expectation (scalar). Bounds per-step
    reward so the imagination AC can't exploit an over-predicted action (exp 0023 diagnostic: the
    agent spammed mk_iron_sword because the MSE reward head over-predicted it)."""

    def __init__(self, state_dim, num_actions, num_bins=255, vmin=-20.0, vmax=20.0, hidden=256):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, hidden)
        self.net = nn.Sequential(
            nn.Linear(state_dim + hidden, hidden), nn.SiLU(), nn.Linear(hidden, num_bins)
        )
        self.register_buffer("bins", torch.linspace(vmin, vmax, num_bins))

    def _logits(self, latent, action):
        return self.net(torch.cat([latent, self.action_embed(action)], dim=-1))

    def forward(self, latent, action):
        probs = F.softmax(self._logits(latent, action), dim=-1)
        return symexp((probs * self.bins).sum(-1))

    def twohot_loss(self, latent, action, target):
        tgt = _twohot_encode(symlog(target.detach()), self.bins)
        return -(tgt * F.log_softmax(self._logits(latent, action), dim=-1)).sum(-1).mean()


class EnsembleRewardHead(nn.Module):
    """Deep ensemble of K action-conditioned two-hot reward heads (exp 0055), for MOPO/MOReL-style
    epistemic-uncertainty pessimism. Members differ by init + per-member bootstrap masks, so they
    AGREE in-distribution and DISAGREE off-distribution. That disagreement (std across members) is
    the epistemic signal: the imagination actor optimises ``mean - coef*std``, penalising the OOD
    over-prediction exp0054 measured WITHOUT crushing the true gesture's learned value (the exp0047
    blanket-CROP failure mode). See knowledge/experiments/0054-rtfm-reward-gap.md.

    Drop-in for ``TwoHotRewardHead``: ``forward`` = ensemble-mean scalar reward; ``mean_std`` =
    (mean, std) for the pessimistic imagined reward; ``twohot_loss`` trains all members
    (bootstrapped). ``action_embed`` aliases member 0 so the exp0047 num_embeddings path works."""

    def __init__(
        self, n_members, state_dim, num_actions, num_bins=255, vmin=-20.0, vmax=20.0, hidden=256
    ):
        super().__init__()
        self.members = nn.ModuleList(
            [
                TwoHotRewardHead(state_dim, num_actions, num_bins, vmin, vmax, hidden)
                for _ in range(n_members)
            ]
        )
        self.n_members = n_members

    @property
    def action_embed(self):  # compat shim: num_embeddings is identical across members
        return self.members[0].action_embed

    def forward(self, latent, action):
        """Ensemble-mean scalar reward (drop-in for TwoHotRewardHead.forward)."""
        return torch.stack([m(latent, action) for m in self.members], 0).mean(0)

    def mean_std(self, latent, action):
        """(mean, std) of the per-member scalar reward predictions — std = epistemic uncertainty.
        ``unbiased=False`` so a degenerate K=1 ensemble yields std 0 (not nan)."""
        preds = torch.stack([m(latent, action) for m in self.members], 0)
        return preds.mean(0), preds.std(0, unbiased=False)

    def twohot_loss(self, latent, action, target, bootstrap=True):
        """Mean over members of each member's two-hot CE. With ``bootstrap``, each member trains on
        an independent Bernoulli(0.5) subset of the batch elements (Osband bootstrapped ensemble) so
        members stay diverse → meaningful OOD disagreement. Averaged over members so the loss
        magnitude matches a single head (keeps the WM loss balance unchanged)."""
        tgt = _twohot_encode(symlog(target.detach()), self.members[0].bins)
        total = latent.new_zeros(())
        for m in self.members:
            ce = -(tgt * F.log_softmax(m._logits(latent, action), dim=-1)).sum(-1)  # (B,)
            if bootstrap:
                mask = (torch.rand_like(ce) < 0.5).float()
                total = total + (ce * mask).sum() / mask.sum().clamp_min(1.0)
            else:
                total = total + ce.mean()
        return total / self.n_members
