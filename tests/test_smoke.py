import numpy as np
import torch

from world_model.agents import RandomAgent
from world_model.envs import make_minigrid_env
from world_model.models import ConvEncoder, LatentDynamicsPredictor
from world_model.training import ReplayBuffer, Transition


def test_env_obs_format():
    env = make_minigrid_env("MiniGrid-Empty-5x5-v0")
    obs, _ = env.reset(seed=0)
    assert obs.dtype == np.float32
    assert obs.ndim == 3 and obs.shape[0] == 3  # CHW
    assert obs.min() >= 0.0 and obs.max() <= 1.0


def test_random_episode_runs():
    env = make_minigrid_env("MiniGrid-Empty-5x5-v0")
    agent = RandomAgent(env.action_space, seed=0)
    obs, _ = env.reset(seed=0)
    for _ in range(20):
        obs, _, terminated, truncated, _ = env.step(agent.act(obs))
        if terminated or truncated:
            obs, _ = env.reset()


def test_encoder_predictor_shapes():
    env = make_minigrid_env("MiniGrid-Empty-5x5-v0")
    obs, _ = env.reset(seed=0)
    encoder = ConvEncoder(latent_dim=128)
    predictor = LatentDynamicsPredictor(latent_dim=128, num_actions=int(env.action_space.n))
    batch = torch.as_tensor(np.stack([obs, obs]))
    z = encoder(batch)
    assert z.shape == (2, 128)
    z_next = predictor(z, torch.tensor([0, 1]))
    assert z_next.shape == (2, 128)


def test_replay_roundtrip():
    buf = ReplayBuffer(capacity=10, obs_shape=(3, 4, 4), seed=0)
    obs = np.zeros((3, 4, 4), dtype=np.float32)
    for i in range(15):  # overfill to exercise the ring
        buf.add(Transition(obs, i % 3, 1.0, obs, False))
    assert len(buf) == 10
    batch = buf.sample(4)
    assert batch["obs"].shape == (4, 3, 4, 4)
    assert batch["action"].dtype == np.int64
