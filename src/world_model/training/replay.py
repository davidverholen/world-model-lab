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
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.size = 0
        self.pos = 0
        self.total_adds = 0
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
        self.total_adds += 1

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "done": self.dones[idx],
        }

    def compute_returns(self, gamma: float) -> None:
        """Fill `returns` with discounted return-to-go, reset at episode ends.

        Walks the (unwrapped) buffer backwards: G_i = r_i + gamma * G_{i+1}, restarting
        at dones. The trailing partial episode of a collection round is not
        done-terminated, so the next round's first-episode return leaks backward into
        it (mild overestimate, one partial episode per round boundary) — acceptable
        as a value target.
        """
        g = 0.0
        for i in range(self.size - 1, -1, -1):
            if self.dones[i]:
                g = 0.0
            g = float(self.rewards[i]) + gamma * g
            self.returns[i] = g

    def sample_sequences(self, batch_size: int, length: int) -> dict[str, np.ndarray]:
        """Sample time-contiguous windows for multi-step rollout training.

        Windows never cross an episode boundary (no `done` inside, except possibly at
        the final transition). Assumes the buffer hasn't wrapped (our usage: capacity
        == collected steps); rejection-samples valid start indices.

        Returns obs (B, L, ...), action/reward (B, L), next_obs (B, ...) — the
        observation after the window's last transition.
        """
        if self.total_adds > self.capacity:
            raise NotImplementedError("sequence sampling assumes an unwrapped buffer")
        starts = np.empty(batch_size, dtype=np.int64)
        found = 0
        while found < batch_size:
            cand = self.rng.integers(0, self.size - length, size=2 * batch_size)
            # episode boundary inside the window (excluding final transition) → invalid
            valid = cand[~self.dones[cand[:, None] + np.arange(length - 1)].any(axis=1)]
            take = min(len(valid), batch_size - found)
            starts[found : found + take] = valid[:take]
            found += take
        idx = starts[:, None] + np.arange(length)
        return {
            "obs": self.obs[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "return": self.returns[idx],
            "next_obs": self.next_obs[starts + length - 1],
        }

    def __len__(self) -> int:
        return self.size
