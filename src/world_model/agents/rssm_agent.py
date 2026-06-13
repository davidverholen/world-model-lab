"""RSSMActorAgent: act from a reactive policy over the RSSM belief (h, z).

Like ActorAgent but tracks the stochastic RSSM state via obs_step (posterior z from
each real observation); the action comes from a one-shot Actor forward on the belief
concat(h, z). epsilon mixes random actions for exploration during collection.
"""

import numpy as np
import torch

from world_model.models import ConvEncoder
from world_model.models.actor import Actor
from world_model.models.rssm import RSSM


class RSSMActorAgent:
    def __init__(
        self,
        encoder: ConvEncoder,
        rssm: RSSM,
        actor: Actor,
        num_actions: int,
        device: str,
        epsilon: float = 0.0,
        seed: int = 0,
    ):
        self.encoder = encoder.to(device).eval()
        self.rssm = rssm.to(device).eval()
        self.actor = actor.to(device).eval()
        self.num_actions = num_actions
        self.device = device
        self.epsilon = epsilon
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.reset()

    def reset(self) -> None:
        self._state = self.rssm.initial(1, self.device)
        self._prev_action = torch.full((1,), self.rssm.no_action, device=self.device)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        embed = self.encoder(torch.as_tensor(obs, device=self.device).unsqueeze(0))
        self._state, _, _ = self.rssm.obs_step(self._state, self._prev_action, embed)
        if self.epsilon > 0 and torch.rand((), generator=self.rng).item() < self.epsilon:
            action = int(torch.randint(self.num_actions, (1,), generator=self.rng))
        else:
            action = int(self.actor(self.rssm.belief(self._state)).argmax(dim=-1))
        self._prev_action = torch.tensor([action], device=self.device)
        return action
