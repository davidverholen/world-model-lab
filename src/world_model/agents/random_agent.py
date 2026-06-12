"""Random agent: the data-collection baseline.

World-model training starts from trajectories. A uniform-random policy is the simplest
exploration strategy and is sufficient for small MiniGrid rooms; it stops being
sufficient exactly when environments get interesting (sparse rewards, doors, keys) —
which is the point where the learned world model should start paying for itself.
"""

import gymnasium as gym
import numpy as np


class RandomAgent:
    def __init__(self, action_space: gym.Space, seed: int | None = None):
        self.action_space = action_space
        self.action_space.seed(seed)

    def act(self, obs: np.ndarray) -> int:
        return self.action_space.sample()
