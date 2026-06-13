"""Crafter environment factory — rung 3 (see knowledge/decisions/0006-crafter-vs-craftax.md).

Crafter (Hafner 2021) is a 2D open-world survival game: 64x64 RGB observations, 17 discrete
actions, and a 22-achievement tech tree (wood->table->pickaxe->stone->...->diamond). It is
our rung-3 target, where hierarchy and long-horizon credit assignment start to matter.

Crafter predates gymnasium: it uses the old-gym 4-tuple `step`, a no-arg `reset`, and its
own space classes. This adapter exposes the gymnasium interface the rest of our stack
expects and returns CHW float32 observations in [0, 1] exactly like make_minigrid_env, so
the same encoder / agents / training loops consume it unchanged. Per ADR 0006 we use the
original (PyTorch-native) Crafter with PIXEL observations (the frozen-encoder thesis), not
Craftax (JAX) and not the symbolic view.

`info` carries `achievements` (the 22-achievement dict — the basis of the Crafter score =
geometric mean of per-achievement unlock rates) and `discount` (0.0 on death), which we use
to split the old-gym `done` into gymnasium's terminated (death) vs truncated (length limit).
"""

import crafter
import gymnasium as gym
import numpy as np


class CrafterEnv(gym.Env):
    """Gymnasium adapter over crafter.Env: CHW float obs, 5-tuple step, seeded reset."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, size: int = 64, length: int = 10000, seed: int | None = None):
        super().__init__()
        self._kwargs = {"size": (size, size), "length": length}
        self._env = crafter.Env(seed=seed, **self._kwargs)
        self.action_space = gym.spaces.Discrete(self._env.action_space.n)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (3, size, size), dtype=np.float32)

    @staticmethod
    def _chw(obs: np.ndarray) -> np.ndarray:
        return obs.transpose(2, 0, 1).astype(np.float32) / 255.0

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # reproducible world for this seed (eval); construction is ~0.02s, so cheap
            self._env = crafter.Env(seed=seed, **self._kwargs)
        return self._chw(self._env.reset()), {}

    def step(self, action):
        obs, reward, done, info = self._env.step(int(action))
        terminated = bool(done and info.get("discount", 1.0) == 0.0)  # death
        truncated = bool(done and not terminated)  # length limit
        return self._chw(obs), float(reward), terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        pass


def make_crafter_env(size: int = 64, length: int = 10000, seed: int | None = None) -> gym.Env:
    """Create a Crafter env emitting CHW float observations (gymnasium interface)."""
    return CrafterEnv(size=size, length=length, seed=seed)
