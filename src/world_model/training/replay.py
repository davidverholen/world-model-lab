"""Trajectory storage for world-model training.

A world model is trained on (obs, action, next_obs, reward, done) transitions —
for multi-step latent rollouts we will later need whole trajectory segments, so the
buffer stores episodes contiguously and can sample both single transitions and segments.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    """Flat ring buffer of transitions with uniform sampling."""

    def __init__(self, capacity: int, obs_shape: tuple[int, ...], seed: int | None = None):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.size = 0
        self.pos = 0
        self.rng = np.random.default_rng(seed)

    def add(self, t: Transition) -> None:
        i = self.pos
        self.obs[i] = t.obs
        self.next_obs[i] = t.next_obs
        self.actions[i] = t.action
        self.rewards[i] = t.reward
        self.dones[i] = t.done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "done": self.dones[idx],
        }

    def __len__(self) -> int:
        return self.size
