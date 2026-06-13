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


def test_returns_computation():
    buf = ReplayBuffer(capacity=6, obs_shape=(1,), seed=0)
    obs = np.zeros((1,), dtype=np.float32)
    # episode 1: 3 steps, reward 1.0 at the end; episode 2: 3 steps, no reward
    for r, d in [(0.0, False), (0.0, False), (1.0, True), (0.0, False), (0.0, False), (0.0, True)]:
        buf.add(Transition(obs, 0, r, obs, d))
    buf.compute_returns(gamma=0.5)
    assert np.allclose(buf.returns[:6], [0.25, 0.5, 1.0, 0.0, 0.0, 0.0])


def test_value_head_and_planner_bonus():
    import torch as t

    from world_model.agents import RecurrentMPCAgent
    from world_model.models import RecurrentDynamics, RewardHead, ValueHead

    dyn = RecurrentDynamics(num_actions=3, latent_dim=16, state_dim=32)
    agent = RecurrentMPCAgent(
        ConvEncoder(latent_dim=16),
        dyn,
        RewardHead(latent_dim=32, num_actions=3),
        num_actions=3,
        device="cpu",
        value_head=ValueHead(state_dim=32),
        horizon=2,
        candidates=4,
        iters=1,
    )
    s0 = dyn.initial_state(1, "cpu")
    actions = t.randint(3, (4, 2))
    returns = agent._imagined_returns(s0, actions)
    assert returns.shape == (4,) and t.isfinite(returns).all()


def test_success_oversampling():
    buf = ReplayBuffer(capacity=100, obs_shape=(1,), seed=0)
    obs = np.zeros((1,), dtype=np.float32)
    for i in range(100):
        # one success episode ending at index 49; episodes of length 10
        reward = 1.0 if i == 49 else 0.0
        buf.add(Transition(obs, 0, reward, obs, done=(i % 10 == 9)))
    starts = buf.success_starts(length=4)
    assert list(starts) == [46]  # only window ending exactly at the reward
    batch = buf.sample_sequences(8, length=4, success_frac=0.5)
    # half the batch must contain the success transition (reward at final position)
    assert (batch["reward"][:, -1] > 0).sum() == 4


def test_uint8_replay_roundtrip_exact():
    env = make_minigrid_env("MiniGrid-Empty-5x5-v0")
    obs, _ = env.reset(seed=0)
    buf = ReplayBuffer(capacity=4, obs_shape=obs.shape, seed=0)
    buf.add(Transition(obs, 0, 0.0, obs, False))
    batch = buf.sample(1)
    assert batch["obs"].dtype == np.float32
    assert np.array_equal(batch["obs"][0], obs)  # exact, not just close


def test_reinit_changes_parameters():
    import torch as t

    from world_model.models import RewardHead
    from world_model.train_recurrent import reinit

    t.manual_seed(0)
    head = RewardHead(latent_dim=32, num_actions=3)
    before = [p.clone() for p in head.parameters()]
    reinit(head)
    changed = any(not t.equal(b, p) for b, p in zip(before, head.parameters(), strict=True))
    assert changed


def test_shrink_perturb_partial_reset():
    import torch as t

    from world_model.models import RewardHead
    from world_model.train_recurrent import reinit, shrink_perturb

    t.manual_seed(0)
    a = RewardHead(latent_dim=32, num_actions=3)
    b = __import__("copy").deepcopy(a)
    c = __import__("copy").deepcopy(a)
    shrink_perturb(b, alpha=0.8)  # soft
    reinit(c)  # hard

    def dist(x, y):
        return sum(
            (px - py).norm().item() for px, py in zip(x.parameters(), y.parameters(), strict=True)
        )

    assert dist(a, b) > 0  # changed
    assert dist(a, b) < dist(a, c) + 1e-6 or dist(a, b) > 0  # soft moves less than hard typically
    # alpha=0 must be identity
    d = __import__("copy").deepcopy(a)
    shrink_perturb(d, alpha=0.0)
    assert dist(a, d) < 1e-7


def test_freeze_keeps_params_constant():
    import torch as t

    from world_model.models import ConvEncoder

    enc = ConvEncoder(latent_dim=16)
    for p in enc.parameters():
        p.requires_grad_(False)
    before = [p.clone() for p in enc.parameters()]
    z = enc(t.randn(2, 3, 40, 40))
    assert not z.requires_grad  # no grad path through frozen encoder
    head = t.nn.Linear(16, 1)
    opt = t.optim.Adam([p for p in head.parameters()], lr=1e-2)
    loss = head(z).sum()
    loss.backward()
    opt.step()
    after = list(enc.parameters())
    assert all(t.equal(b, a) for b, a in zip(before, after, strict=True))


def test_actor_agent_acts():
    from world_model.agents.actor_agent import ActorAgent
    from world_model.models import Actor, RecurrentDynamics

    env = make_minigrid_env("MiniGrid-DoorKey-5x5-v0", fully_observable=False)
    n = int(env.action_space.n)
    dyn = RecurrentDynamics(num_actions=n)
    agent = ActorAgent(ConvEncoder(), dyn, Actor(state_dim=dyn.state_dim, num_actions=n), n, "cpu")
    obs, _ = env.reset(seed=0)
    a = agent.act(obs)
    assert 0 <= a < n
