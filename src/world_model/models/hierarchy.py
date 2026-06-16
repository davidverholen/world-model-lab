"""Director-style manager/worker hierarchy over the RSSM, trained in imagination (exp 0050).

The flat reactive actor hit a ~0.10 execution ceiling on the rung-4 length-2 gesture (exp 0048/9):
it *follows* the displayed manual but cannot reliably *sequence* the 2-step gesture — the
exp0043/0044 multi-step credit-assignment wall. Director ([[director-2022]]) supplies the temporal
abstraction: a MANAGER proposes a subgoal every K steps, a WORKER reaches it, both trained in
imagination. See knowledge/experiments/0050-rtfm-hierarchy.md.

Minimal first cut (Phase A, state-only — no manual conditioning yet):
- Goal space = the RSSM belief space directly (no VQ-VAE / learned goal autoencoder yet). A goal is
  a centroid from a K-means **codebook** over visited beliefs; the worker's reward is the cosine
  similarity between its current belief and that goal (Director §4).
- Worker: a goal-conditioned policy π_w(a | belief, goal). Manager: π_m(code | belief) → a codebook
  index every K steps, trained on the (accumulated) task reward (Director §5, exploration bonus
  omitted for the first probe).
"""

import torch
import torch.nn.functional as F
from torch import nn

from world_model.models.actor import Actor
from world_model.models.twohot import TwoHotValueHead


class GoalCodebook(nn.Module):
    """A fixed-size set of goal vectors (centroids) in belief space, fit by K-means over visited
    beliefs. The manager's discrete action space (Director §2, minus the VQ-VAE — centroids stand in
    for learned codes for the first probe). Reachable by construction: every centroid is the mean of
    real visited beliefs."""

    def __init__(self, n_codes: int, dim: int):
        super().__init__()
        self.n_codes = n_codes
        self.dim = dim
        # registered (not a parameter — set by K-means, not gradient); starts at small noise so an
        # unfit codebook still yields distinct, finite goals in smoke tests.
        self.register_buffer("centroids", torch.randn(n_codes, dim) * 0.01)
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def fit(self, beliefs: torch.Tensor, iters: int = 10) -> None:
        """Lloyd's K-means over ``beliefs`` (N, dim) → set ``centroids``. Cosine geometry (normalize
        before assigning) to match the worker's cosine similarity reward."""
        n = beliefs.shape[0]
        if n < self.n_codes:  # too few points: pad by sampling with replacement
            idx = torch.randint(0, n, (self.n_codes,), device=beliefs.device)
            self.centroids.copy_(beliefs[idx])
            self.fitted.fill_(True)
            return
        x = F.normalize(beliefs, dim=-1)
        c = x[torch.randperm(n, device=x.device)[: self.n_codes]].clone()
        for _ in range(iters):
            assign = (x @ c.t()).argmax(dim=-1)  # cosine nearest centroid
            for k in range(self.n_codes):
                m = assign == k
                if m.any():
                    c[k] = F.normalize(x[m].mean(0), dim=-1)
        self.centroids.copy_(c)
        self.fitted.fill_(True)

    def vecs(self, idx: torch.Tensor) -> torch.Tensor:
        """idx (...) long → (..., dim) goal vectors."""
        return self.centroids[idx]

    @torch.no_grad()
    def nearest(self, belief: torch.Tensor) -> torch.Tensor:
        """(B, dim) → (B,) nearest centroid index by cosine (for HAC hindsight relabeling)."""
        sims = F.normalize(belief, dim=-1) @ F.normalize(self.centroids, dim=-1).t()
        return sims.argmax(dim=-1)


def goal_similarity(belief: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Worker reward (Director §4): cosine similarity between the current belief and the goal, in
    belief space. (..., dim),(..., dim) → (...). Bounded [-1, 1], so the worker value head stays
    well-scaled without the task-reward magnitude."""
    return F.cosine_similarity(belief, goal, dim=-1)


class GoalWorker(nn.Module):
    """Goal-conditioned worker policy π_w(a | belief, goal) — the existing Actor over concat(belief,
    goal). Trained on goal_similarity reward only (no task reward; Director §4)."""

    def __init__(self, state_dim: int, goal_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.actor = Actor(state_dim + goal_dim, num_actions, hidden_dim)

    def forward(self, belief: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """(B, state_dim),(B, goal_dim) → (B, num_actions) logits."""
        return self.actor(torch.cat([belief, goal], dim=-1))


class Manager(nn.Module):
    """Manager policy π_m(code | belief) → logits over the codebook. Picks a subgoal every K steps,
    trained on the accumulated task reward (Director §5). Phase A is state-only; Phase B will
    concat a manual embedding (exp 0050)."""

    def __init__(self, state_dim: int, n_codes: int, hidden_dim: int = 256, ctx_dim: int = 0):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + ctx_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, n_codes)
        )

    def forward(self, belief: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        """(B, state_dim)[, (B, ctx_dim)] → (B, n_codes) logits over codebook indices."""
        x = belief if (ctx is None or self.ctx_dim == 0) else torch.cat([belief, ctx], dim=-1)
        return self.net(x)


def make_hierarchy(state_dim: int, n_actions: int, n_codes: int, manager_ctx_dim: int = 0):
    """Build worker + manager + their two-hot value heads. Goal space = belief space (goal_dim =
    state_dim) for the first cut. Returns (worker, worker_critic, manager, manager_critic,
    codebook).
    Worker value is goal-conditioned (state+goal); manager value is over the belief only."""
    worker = GoalWorker(state_dim, state_dim, n_actions)
    worker_critic = TwoHotValueHead(state_dim=state_dim + state_dim)
    manager = Manager(state_dim, n_codes, ctx_dim=manager_ctx_dim)
    manager_critic = TwoHotValueHead(state_dim=state_dim)
    codebook = GoalCodebook(n_codes, state_dim)
    return worker, worker_critic, manager, manager_critic, codebook
