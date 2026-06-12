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


def test_sigreg_discriminates_collapse():
    from world_model.models import SIGReg

    torch.manual_seed(0)
    sigreg = SIGReg(num_slices=128)
    gaussian = torch.randn(256, 32)
    collapsed = torch.zeros(256, 32) + 0.1
    loss_gaussian = sigreg(gaussian)
    loss_collapsed = sigreg(collapsed)
    assert loss_collapsed > 10 * loss_gaussian  # collapse must be heavily penalized
    # gradients flow
    z = torch.randn(64, 16, requires_grad=True)
    sigreg(z).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_mpc_agent_acts():
    from world_model.agents import MPCAgent
    from world_model.models import RewardHead

    env = make_minigrid_env("MiniGrid-Empty-5x5-v0")
    n = int(env.action_space.n)
    agent = MPCAgent(
        ConvEncoder(),
        LatentDynamicsPredictor(num_actions=n),
        RewardHead(num_actions=n),
        num_actions=n,
        device="cpu",
        horizon=3,
        candidates=16,
    )
    obs, _ = env.reset(seed=0)
    action = agent.act(obs)
    assert 0 <= action < n


def test_sequence_sampling():
    buf = ReplayBuffer(capacity=50, obs_shape=(3, 4, 4), seed=0)
    obs = np.zeros((3, 4, 4), dtype=np.float32)
    for i in range(50):
        buf.add(Transition(obs, i % 3, 0.0, obs, done=(i % 10 == 9)))
    batch = buf.sample_sequences(8, length=4)
    assert batch["obs"].shape == (8, 4, 3, 4, 4)
    assert batch["action"].shape == (8, 4)
    assert batch["next_obs"].shape == (8, 3, 4, 4)


def test_recurrent_world_model_and_agent():
    from world_model.agents import RecurrentMPCAgent
    from world_model.models import RecurrentDynamics, RewardHead

    env = make_minigrid_env("MiniGrid-DoorKey-5x5-v0", fully_observable=False)
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] == 3  # RGB partial view
    n = int(env.action_space.n)
    dyn = RecurrentDynamics(num_actions=n)
    agent = RecurrentMPCAgent(
        ConvEncoder(),
        dyn,
        RewardHead(latent_dim=dyn.state_dim, num_actions=n),
        num_actions=n,
        device="cpu",
        horizon=3,
        candidates=8,
        iters=1,
    )
    for _ in range(3):
        a = agent.act(obs)
        assert 0 <= a < n
        obs, *_ = env.step(a)
    agent.reset()  # belief reset between episodes
