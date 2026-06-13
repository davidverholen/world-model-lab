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

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        seed: int | None = None,
        obs_dtype: np.dtype = np.uint8,
    ):
        # Pixel obs are stored as uint8 (originate from uint8 pixels scaled to [0,1], so
        # round-trip via round(x*255) is exact) — 4x less RAM than float32 (exp 0010 ops
        # lesson: 6 parallel float32 buffers paged the desktop to death). obs_dtype=float32
        # stores embeddings verbatim (no [0,1] assumption) — for the frozen-encoder cache
        # (train_crafter: store DINO embeddings, not pixels; skip the encoder in WM updates).
        self._u8 = np.dtype(obs_dtype) == np.uint8
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
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
        self.obs[i] = np.round(t.obs * 255.0) if self._u8 else t.obs
        self.next_obs[i] = np.round(t.next_obs * 255.0) if self._u8 else t.next_obs
        self.actions[i] = t.action
        self.rewards[i] = t.reward
        self.dones[i] = t.done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_adds += 1

    def _to_float(self, obs: np.ndarray) -> np.ndarray:
        return obs.astype(np.float32) / 255.0 if self._u8 else obs.astype(np.float32, copy=False)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self._to_float(self.obs[idx]),
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "next_obs": self._to_float(self.next_obs[idx]),
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

    def _valid_starts(self, cand: np.ndarray, length: int) -> np.ndarray:
        """Filter window starts: in range, no episode boundary before the final step."""
        cand = cand[(cand >= 0) & (cand <= self.size - length)]
        if len(cand) == 0:
            return cand
        return cand[~self.dones[cand[:, None] + np.arange(length - 1)].any(axis=1)]

    def success_starts(self, length: int) -> np.ndarray:
        """Valid window starts whose final transition carries a nonzero reward.

        Terminal rewards end episodes, so a reward transition can only sit at a
        window's last position — one canonical window per reward event.
        """
        reward_idx = np.flatnonzero(np.abs(self.rewards[: self.size]) > 1e-6)
        return self._valid_starts(reward_idx - length + 1, length)

    def sample_sequences(
        self, batch_size: int, length: int, success_frac: float = 0.0
    ) -> dict[str, np.ndarray]:
        """Sample time-contiguous windows for multi-step rollout training.

        Windows never cross an episode boundary (no `done` inside, except possibly at
        the final transition). Assumes the buffer hasn't wrapped (our usage: capacity
        == collected steps); rejection-samples valid start indices.

        success_frac > 0 oversamples windows that END in a reward transition (exp
        0009 ignition fix: success episodes must not drown in the replay) — that
        fraction of the batch is drawn from `success_starts`, the rest uniformly.

        Returns obs (B, L, ...), action/reward/return (B, L), next_obs (B, ...).
        """
        if self.total_adds > self.capacity:
            raise NotImplementedError("sequence sampling assumes an unwrapped buffer")
        starts = np.empty(batch_size, dtype=np.int64)
        found = 0
        if success_frac > 0:
            pool = self.success_starts(length)
            if len(pool) > 0:
                want = int(batch_size * success_frac)
                starts[:want] = pool[self.rng.integers(0, len(pool), size=want)]
                found = want
        while found < batch_size:
            cand = self.rng.integers(0, self.size - length, size=2 * batch_size)
            valid = self._valid_starts(cand, length)
            take = min(len(valid), batch_size - found)
            starts[found : found + take] = valid[:take]
            found += take
        idx = starts[:, None] + np.arange(length)
        return {
            "obs": self._to_float(self.obs[idx]),
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "return": self.returns[idx],
            "done": self.dones[idx],  # terminations (windows may end in one)
            "next_obs": self._to_float(self.next_obs[starts + length - 1]),
        }

    def __len__(self) -> int:
        return self.size
