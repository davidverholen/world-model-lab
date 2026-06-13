"""ActorAgent: act from a reactive policy over the GRU belief state.

Same belief-tracking as RecurrentMPCAgent (encode obs, update GRU belief online),
but the action comes from a one-shot Actor forward instead of CEM planning — the
Mode-1 fast path. Call reset() at episode start.
"""

import numpy as np
import torch

from world_model.models import ConvEncoder, RecurrentDynamics
from world_model.models.actor import Actor


class ActorAgent:
    def __init__(
        self,
        encoder: ConvEncoder,
        dynamics: RecurrentDynamics,
        actor: Actor,
        num_actions: int,
        device: str,
        sample: bool = False,
        seed: int = 0,
    ):
        self.encoder = encoder.to(device).eval()
        self.dynamics = dynamics.to(device).eval()
        self.actor = actor.to(device).eval()
        self.num_actions = num_actions
        self.device = device
        self.sample = sample
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.reset()

    def reset(self) -> None:
        self._state = self.dynamics.initial_state(1, self.device)
        self._prev_action = torch.full((1,), self.dynamics.no_action, device=self.device)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        z = self.encoder(torch.as_tensor(obs, device=self.device).unsqueeze(0))
        self._state = self.dynamics.update(z, self._prev_action, self._state)
        logits = self.actor(self._state)
        if self.sample:
            probs = torch.softmax(logits, dim=-1).cpu()
            action = int(torch.multinomial(probs, 1, generator=self.rng))
        else:
            action = int(logits.argmax(dim=-1))
        self._prev_action = torch.tensor([action], device=self.device)
        return action
