"""MiniGrid environment factory.

MiniGrid is our first rung on the environment ladder (see knowledge/environments/minigrid.md):
small, fast, partially observable gridworlds — cheap enough to iterate on world-model
architectures in minutes instead of hours.

Observations are returned as CHW float tensors in [0, 1] so they can feed a conv encoder
directly. We use the fully-observable RGB wrapper for the first experiments; partial
observability is a deliberate later step (it is where world models start to matter).
"""

import gymnasium as gym
import numpy as np
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper, RGBImgPartialObsWrapper


class ChwFloatObs(gym.ObservationWrapper):
    """Convert HWC uint8 image observations to CHW float32 in [0, 1]."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs.transpose(2, 0, 1).astype(np.float32) / 255.0


def make_minigrid_env(
    env_id: str = "MiniGrid-Empty-5x5-v0",
    *,
    fully_observable: bool = True,
    tile_size: int = 8,
    **kwargs,
) -> gym.Env:
    """Create a MiniGrid env emitting CHW float image observations.

    Args:
        env_id: any registered MiniGrid id, e.g. "MiniGrid-Empty-5x5-v0",
            "MiniGrid-DoorKey-6x6-v0".
        fully_observable: if True, render the full grid as RGB; if False, keep the
            agent's egocentric partial view.
        tile_size: pixels per grid tile (controls observation resolution).
    """
    env = gym.make(env_id, **kwargs)
    if fully_observable:
        env = RGBImgObsWrapper(env, tile_size=tile_size)
    else:
        # egocentric 7x7-tile RGB view — pixels, like the fully observable path
        env = RGBImgPartialObsWrapper(env, tile_size=tile_size)
    env = ImgObsWrapper(env)
    return ChwFloatObs(env)
