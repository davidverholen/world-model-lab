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

from world_model.models import ConvEncoder, LatentDynamicsPredictor, RecurrentDynamics
from world_model.models.reward import RewardHead


def _cem_plan(rollout_fn, num_actions, horizon, candidates, iters, num_elites, rng, device):
    """Shared CEM loop: refine per-timestep action distributions toward elites."""
    probs = torch.full((horizon, num_actions), 1.0 / num_actions)
    best_action, best_return = 0, -torch.inf
    for _ in range(iters):
        actions = torch.multinomial(probs, candidates, replacement=True, generator=rng).T.to(device)
        returns = rollout_fn(actions)
        elites = actions[returns.topk(num_elites).indices].cpu()
        top = int(returns.argmax())
        if float(returns[top]) > best_return:
            best_return = float(returns[top])
            best_action = int(actions[top, 0])
        onehot = torch.zeros(horizon, num_actions)
        for t in range(horizon):
            onehot[t] = torch.bincount(elites[:, t], minlength=num_actions).float()
        probs = 0.5 * probs + 0.5 * (onehot / onehot.sum(dim=1, keepdim=True))
        probs = probs.clamp_min(0.02)
        probs = probs / probs.sum(dim=1, keepdim=True)
    return best_action


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
        return _cem_plan(
            lambda actions: self._rollout_returns(z0, actions),
            self.num_actions,
            self.horizon,
            self.candidates,
            self.iters,
            self.num_elites,
            self.rng,
            self.device,
        )


class RecurrentMPCAgent:
    """Belief-state MPC for partial observability.

    Maintains the GRU belief online (closed-loop with real observations); plans by
    imagining open-loop continuations where predicted latents feed the belief.
    Call reset() at episode start. epsilon > 0 mixes in random actions — used when
    this agent collects training data (exploration), zero for evaluation.
    """

    def __init__(
        self,
        encoder: ConvEncoder,
        dynamics: RecurrentDynamics,
        reward_head: RewardHead,
        num_actions: int,
        device: str,
        horizon: int = 20,
        candidates: int = 512,
        iters: int = 3,
        elite_frac: float = 0.1,
        discount: float = 0.98,
        epsilon: float = 0.0,
        seed: int = 0,
    ):
        self.encoder = encoder.to(device).eval()
        self.dynamics = dynamics.to(device).eval()
        self.reward_head = reward_head.to(device).eval()
        self.num_actions = num_actions
        self.device = device
        self.horizon = horizon
        self.candidates = candidates
        self.iters = iters
        self.num_elites = max(1, int(elite_frac * candidates))
        self.discount = discount
        self.epsilon = epsilon
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.reset()

    @classmethod
    def from_checkpoint(cls, path: str, device: str, **kwargs) -> "RecurrentMPCAgent":
        ckpt = torch.load(path, map_location=device, weights_only=True)
        num_actions = int(ckpt["config"]["num_actions"])
        encoder = ConvEncoder()
        dynamics = RecurrentDynamics(num_actions=num_actions)
        reward_head = RewardHead(latent_dim=dynamics.state_dim, num_actions=num_actions)
        encoder.load_state_dict(ckpt["encoder"])
        dynamics.load_state_dict(ckpt["dynamics"])
        reward_head.load_state_dict(ckpt["reward_head"])
        return cls(encoder, dynamics, reward_head, num_actions, device, **kwargs)

    def reset(self) -> None:
        self._state = self.dynamics.initial_state(1, self.device)
        self._prev_action = torch.full((1,), self.dynamics.no_action, device=self.device)

    @torch.no_grad()
    def _imagined_returns(self, s0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        s = s0.expand(actions.shape[0], -1).contiguous()
        returns = torch.zeros(actions.shape[0], device=self.device)
        for t in range(actions.shape[1]):
            returns += (self.discount**t) * self.reward_head(s, actions[:, t])
            z_hat = self.dynamics.predict_next(s, actions[:, t])
            s = self.dynamics.update(z_hat, actions[:, t], s)
        return returns

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        z = self.encoder(torch.as_tensor(obs, device=self.device).unsqueeze(0))
        self._state = self.dynamics.update(z, self._prev_action, self._state)

        if self.epsilon > 0 and torch.rand((), generator=self.rng).item() < self.epsilon:
            action = int(torch.randint(self.num_actions, (1,), generator=self.rng))
        else:
            action = _cem_plan(
                lambda actions: self._imagined_returns(self._state, actions),
                self.num_actions,
                self.horizon,
                self.candidates,
                self.iters,
                self.num_elites,
                self.rng,
                self.device,
            )
        self._prev_action = torch.tensor([action], device=self.device)
        return action
