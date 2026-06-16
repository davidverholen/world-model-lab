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


class GoalVQVAE(nn.Module):
    """Director's goal autoencoder (§2): a VQ-VAE over RSSM beliefs that learns a SEPARABLE discrete
    goal space. The minimal K-means-in-raw-belief cut failed (exp0050b/c: raw-belief cosine is
    ~uninformative — worker_sim flat ~0.6 = ambient, so goals are meaningless and the manager
    collapses). The VQ-VAE fixes this: the encoder learns a latent that RECONSTRUCTS the belief, so
    distinct states get distinct latents — a goal space where reaching a goal is meaningful.

    - The manager's action space = the codebook (``n_codes`` discrete codes).
    - A code → a goal vector ``goal_of(idx)`` (the code embedding) in the encoder latent space.
    - The worker's similarity reward lives in that latent: ``cosine(encode(next_belief), goal)``.
    - ``recon_error`` (per-state) feeds the optional manager exploration bonus (Director §5: novel =
      high autoencoder error).
    """

    def __init__(
        self,
        belief_dim: int,
        code_dim: int = 32,
        n_codes: int = 64,
        beta: float = 0.25,
        hidden: int = 256,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(belief_dim, hidden), nn.SiLU(), nn.Linear(hidden, code_dim)
        )
        self.codebook = nn.Embedding(n_codes, code_dim)
        self.codebook.weight.data.uniform_(-1.0 / n_codes, 1.0 / n_codes)
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, hidden), nn.SiLU(), nn.Linear(hidden, belief_dim)
        )
        self.n_codes, self.code_dim, self.beta = n_codes, code_dim, beta
        self.goal_dim = code_dim  # worker goal-space dimension (the encoder latent)

    def encode(self, belief: torch.Tensor) -> torch.Tensor:
        """(..., belief_dim) → (..., code_dim) latent (the worker's goal-reward space)."""
        return self.encoder(belief)

    def quantize(self, z_e: torch.Tensor):
        """Nearest-codebook quantization. (B, code_dim) → (z_q, idx)."""
        d = (
            z_e.pow(2).sum(-1, keepdim=True)
            - 2 * z_e @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(-1)
        )
        idx = d.argmin(-1)
        return self.codebook(idx), idx

    def goal_of(self, idx: torch.Tensor) -> torch.Tensor:
        """Manager code index (...) → (..., code_dim) goal vector (the code embedding)."""
        return self.codebook(idx)

    def forward(self, belief: torch.Tensor):
        """VQ-VAE pass → (recon_loss, vq_loss, idx). Straight-through estimator so the encoder
        gets gradients through the quantizer."""
        z_e = self.encode(belief)
        z_q, idx = self.quantize(z_e)
        z_st = z_e + (z_q - z_e).detach()  # straight-through
        recon = self.decoder(z_st)
        recon_loss = F.mse_loss(recon, belief)
        vq_loss = F.mse_loss(z_q, z_e.detach()) + self.beta * F.mse_loss(z_e, z_q.detach())
        return recon_loss, vq_loss, idx

    @torch.no_grad()
    def recon_error(self, belief: torch.Tensor) -> torch.Tensor:
        """Per-state reconstruction error (B,) — the Director §5 novelty/exploration signal."""
        z_q, _ = self.quantize(self.encode(belief))
        return (self.decoder(z_q) - belief).pow(2).mean(-1)

    # Uniform goal-module interface (shared with GoalCodebook) so the hierarchy loop is goal-kind-
    # agnostic: ``project`` maps a belief into the goal space the worker reward lives in.
    def project(self, belief: torch.Tensor) -> torch.Tensor:
        return self.encode(belief)


class GoalCodebook(nn.Module):
    """A fixed-size set of goal vectors (centroids) in belief space, fit by K-means over visited
    beliefs. The manager's discrete action space (Director §2, minus the VQ-VAE — centroids stand in
    for learned codes for the first probe). Reachable by construction: every centroid is the mean of
    real visited beliefs."""

    def __init__(self, n_codes: int, dim: int):
        super().__init__()
        self.n_codes = n_codes
        self.dim = dim
        self.goal_dim = dim  # K-means goal space == belief space
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

    # Uniform goal-module interface (shared with GoalVQVAE): the K-means goal space IS belief space,
    # so project is the identity and goal_of aliases vecs.
    def goal_of(self, idx: torch.Tensor) -> torch.Tensor:
        return self.centroids[idx]

    def project(self, belief: torch.Tensor) -> torch.Tensor:
        return belief

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


def make_hierarchy(
    state_dim: int,
    n_actions: int,
    n_codes: int,
    manager_ctx_dim: int = 0,
    goal_kind: str = "vq",
    code_dim: int = 32,
):
    """Build worker + manager + their two-hot value heads + the goal module. Returns
    (worker, worker_critic, manager, manager_critic, goal_module).

    ``goal_kind='vq'`` (exp0050 chosen path): a learned VQ-VAE goal autoencoder — goal space = its
    ``code_dim`` latent; the worker reaches a code embedding, reward = cosine in the encoder latent.
    ``goal_kind='kmeans'`` (the failed minimal cut, kept for reference): raw-belief centroids. The
    worker value is goal-conditioned (belief + goal); the manager value is over the belief only."""
    goal_dim = code_dim if goal_kind == "vq" else state_dim
    worker = GoalWorker(state_dim, goal_dim, n_actions)
    worker_critic = TwoHotValueHead(state_dim=state_dim + goal_dim)
    manager = Manager(state_dim, n_codes, ctx_dim=manager_ctx_dim)
    manager_critic = TwoHotValueHead(state_dim=state_dim)
    if goal_kind == "vq":
        goal_module = GoalVQVAE(state_dim, code_dim=code_dim, n_codes=n_codes)
    else:
        goal_module = GoalCodebook(n_codes, state_dim)
    return worker, worker_critic, manager, manager_critic, goal_module
