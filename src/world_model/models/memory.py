"""Recurrent world model: GRU belief state over partial observations.

Under partial observability a single frame doesn't determine the world state (the
key may be behind you). The belief state s_t summarizes the observation history:

    s_t = GRU([z_t, emb(a_{t-1})], s_{t-1})          # closed-loop update (real obs)
    ẑ_{t+1} = predict_next(s_t, a_t)                 # latent prediction
    ŝ_{t+1} = GRU([ẑ_{t+1}, emb(a_t)], ŝ_t)          # open-loop update (imagination)

Planning rolls the open-loop form: imagined observations feed back into the belief.
This is a deterministic RSSM-lite (PlaNet's RSSM adds stochastic latents — a later
experiment if deterministic memory plateaus). Reward is predicted from (s, a), so it
can depend on remembered facts (e.g. "I picked up the key five steps ago").
"""

import torch
from torch import nn


class RecurrentDynamics(nn.Module):
    def __init__(self, latent_dim: int = 128, num_actions: int = 7, state_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_embed = nn.Embedding(num_actions + 1, 32)  # +1: "no previous action"
        self.no_action = num_actions
        self.cell = nn.GRUCell(latent_dim + 32, state_dim)
        self.next_latent = nn.Sequential(
            nn.Linear(state_dim + 32, 256), nn.SiLU(), nn.Linear(256, latent_dim)
        )

    def initial_state(self, batch: int, device) -> torch.Tensor:
        return torch.zeros(batch, self.state_dim, device=device)

    def update(self, z: torch.Tensor, prev_action: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Belief update with an (observed or imagined) latent. (B,D),(B,),(B,S)->(B,S)."""
        x = torch.cat([z, self.action_embed(prev_action)], dim=-1)
        return self.cell(x, s)

    def predict_next(self, s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict the next observation latent from belief and action. ->(B,D)."""
        return self.next_latent(torch.cat([s, self.action_embed(action)], dim=-1))
