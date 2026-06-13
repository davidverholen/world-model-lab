"""Stochastic recurrent state-space model (RSSM) — Gaussian latents (exp 0019).

The exploitation fix after exps 0017/0018: a DETERMINISTIC world model is maximally
exploitable — the imagination actor-critic drives the model to a single fake-reward
state. RSSM makes the latent STOCHASTIC: imagination samples z from a prior p(z|h)
trained (via KL) to match the posterior q(z|h,obs) over real data, so imagined
rollouts stay near the real distribution and no single exploitable trajectory exists.
Gaussian latents (PlaNet/DreamerV1 lineage); DreamerV3 uses categoricals.
See knowledge/papers/dreamerv3-2023.md, knowledge/concepts/generative-vs-predictive.md.

State = (h: deterministic GRU, z: stochastic sample). Belief consumed by all heads =
concat(h, z). obs_step uses the posterior (real observation available); img_step uses
the prior (imagination, no observation).
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal


class RSSM(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_actions: int = 7,
        deter_dim: int = 256,
        stoch_dim: int = 32,
        hidden_dim: int = 256,
        min_std: float = 0.1,
    ):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.state_dim = deter_dim + stoch_dim  # belief dim for the heads
        self.no_action = num_actions
        self.min_std = min_std
        self.action_embed = nn.Embedding(num_actions + 1, 32)
        self.cell = nn.GRUCell(stoch_dim + 32, deter_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2 * stoch_dim)
        )
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )

    def initial(self, batch: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch, self.deter_dim, device=device),
            torch.zeros(batch, self.stoch_dim, device=device),
        )

    def _dist(self, params: torch.Tensor) -> Normal:
        mean, std = params.chunk(2, dim=-1)
        return Normal(mean, F.softplus(std) + self.min_std)

    def _deter(self, h: torch.Tensor, z: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, self.action_embed(prev_action)], dim=-1)
        return self.cell(x, h)

    def obs_step(self, state, prev_action, embed):
        """One step WITH observation → posterior sample. Returns (state, prior, post)."""
        h, z = state
        h = self._deter(h, z, prev_action)
        prior = self._dist(self.prior_net(h))
        post = self._dist(self.post_net(torch.cat([h, embed], dim=-1)))
        return (h, post.rsample()), prior, post

    def img_step(self, state, prev_action):
        """One step WITHOUT observation → prior sample (imagination). Returns (state, prior)."""
        h, z = state
        h = self._deter(h, z, prev_action)
        prior = self._dist(self.prior_net(h))
        return (h, prior.rsample()), prior

    @staticmethod
    def belief(state) -> torch.Tensor:
        """concat(h, z) — the 'belief' vector every head/actor/planner consumes."""
        return torch.cat(state, dim=-1)
