"""MPC agent: plans by imagining futures inside the learned world model.

At every environment step:
1. encode the current observation into a latent z;
2. imagine `candidates` action sequences of length `horizon`, rolled out entirely in
   latent space with the dynamics predictor (no environment access);
3. score each imagined future with the reward head (discounted sum);
4. CEM refinement: keep the top-`elite_frac` sequences, refit per-timestep action
   probabilities to the elites, resample, repeat `iters` times (iters=1 = plain
   random shooting). Needed because with sparse rewards a uniformly random sequence
   almost never reaches the goal in imagination (learned in exp 0004);
5. execute the first action of the best imagined sequence; re-plan next step.
"""

import numpy as np
import torch

from world_model.models import ConvEncoder, LatentDynamicsPredictor
from world_model.models.reward import RewardHead


class MPCAgent:
    def __init__(
        self,
        encoder: ConvEncoder,
        predictor: LatentDynamicsPredictor,
        reward_head: RewardHead,
        num_actions: int,
        device: str,
        horizon: int = 16,
        candidates: int = 512,
        iters: int = 3,
        elite_frac: float = 0.1,
        discount: float = 0.98,
        seed: int = 0,
    ):
        self.encoder = encoder.to(device).eval()
        self.predictor = predictor.to(device).eval()
        self.reward_head = reward_head.to(device).eval()
        self.num_actions = num_actions
        self.device = device
        self.horizon = horizon
        self.candidates = candidates
        self.iters = iters
        self.num_elites = max(1, int(elite_frac * candidates))
        self.discount = discount
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

    @classmethod
    def from_checkpoint(cls, path: str, device: str, **kwargs) -> "MPCAgent":
        ckpt = torch.load(path, map_location=device, weights_only=True)
        num_actions = int(ckpt["config"]["num_actions"])
        encoder = ConvEncoder()
        predictor = LatentDynamicsPredictor(num_actions=num_actions)
        reward_head = RewardHead(num_actions=num_actions)
        encoder.load_state_dict(ckpt["encoder"])
        predictor.load_state_dict(ckpt["predictor"])
        reward_head.load_state_dict(ckpt["reward_head"])
        return cls(encoder, predictor, reward_head, num_actions, device, **kwargs)

    @torch.no_grad()
    def _rollout_returns(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        z = z0.expand(actions.shape[0], -1).contiguous()
        returns = torch.zeros(actions.shape[0], device=self.device)
        for t in range(actions.shape[1]):
            returns += (self.discount**t) * self.reward_head(z, actions[:, t])
            z = self.predictor(z, actions[:, t])
        return returns

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        z0 = self.encoder(torch.as_tensor(obs, device=self.device).unsqueeze(0))
        probs = torch.full((self.horizon, self.num_actions), 1.0 / self.num_actions)

        best_action, best_return = 0, -torch.inf
        for _ in range(self.iters):
            actions = torch.multinomial(
                probs, self.candidates, replacement=True, generator=self.rng
            ).T.to(self.device)  # (candidates, horizon)
            returns = self._rollout_returns(z0, actions)
            elites = actions[returns.topk(self.num_elites).indices].cpu()  # (E, horizon)

            top = int(returns.argmax())
            if float(returns[top]) > best_return:
                best_return = float(returns[top])
                best_action = int(actions[top, 0])

            # refit per-timestep categorical to elites, smoothed, floored
            onehot = torch.zeros(self.horizon, self.num_actions)
            for t in range(self.horizon):
                onehot[t] = torch.bincount(elites[:, t], minlength=self.num_actions).float()
            probs = 0.5 * probs + 0.5 * (onehot / onehot.sum(dim=1, keepdim=True))
            probs = probs.clamp_min(0.02)
            probs = probs / probs.sum(dim=1, keepdim=True)

        return best_action
