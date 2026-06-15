"""CPU smoke tests for the rung-4 CEM-MPC planner + agent (exp 0044).

See knowledge/design/hierarchical-imagination-agent.md §5,§7 and agents.rtfm_mpc. Fast (<30s):
tiny ConditionedRSSM + TwoHotRewardHead, no real DINO/MiniLM encoders (the planner only consumes
the WM + reward head + token tensors, so fakes suffice)."""

import numpy as np
import torch

from world_model.agents.rtfm_mpc import RTFMMPCAgent, plan_action
from world_model.models.manual_conditioning import ConditionedRSSM
from world_model.models.twohot import TwoHotRewardHead

TD, ED, N_ACT = 384, 64, 17


def _tiny_wm():
    torch.manual_seed(0)
    rssm = ConditionedRSSM(embed_dim=ED, num_actions=N_ACT, text_dim=TD, ctx_dim=64)
    rew = TwoHotRewardHead(state_dim=rssm.state_dim, num_actions=N_ACT)
    return rssm.eval(), rew.eval()


def test_plan_action_returns_valid_int():
    rssm, rew = _tiny_wm()
    state = rssm.initial(1, "cpu")
    tokens = torch.randn(1, 6, TD)
    mask = torch.ones(1, 6, dtype=torch.bool)
    rng = torch.Generator(device="cpu").manual_seed(0)
    a = plan_action(
        rssm,
        rew,
        state,
        tokens,
        mask,
        horizon=5,
        n_samples=64,
        n_iters=3,
        n_elites=8,
        n_actions=N_ACT,
        device="cpu",
        gamma=0.99,
        rng=rng,
    )
    assert isinstance(a, int) and 0 <= a < N_ACT


def test_plan_action_no_mask():
    rssm, rew = _tiny_wm()
    state = rssm.initial(1, "cpu")
    tokens = torch.randn(1, 4, TD)
    a = plan_action(
        rssm,
        rew,
        state,
        tokens,
        None,
        horizon=3,
        n_samples=32,
        n_iters=2,
        n_elites=4,
        n_actions=N_ACT,
        device="cpu",
        gamma=0.99,
    )
    assert 0 <= a < N_ACT


class _FakeEnc:
    """Stand-in for the frozen DINO encoder: any CHW image tensor → (1, ED) embedding."""

    def __call__(self, e):
        return torch.zeros(e.shape[0], ED)


class _FakeTextEnc:
    """Stand-in for the frozen MiniLM text encoder: returns fixed token embeddings + mask."""

    def encode(self, manuals):
        b = len(manuals)
        return torch.randn(b, 5, TD), torch.ones(b, 5, dtype=torch.bool)


def test_mpc_agent_act_over_fake_steps():
    rssm, rew = _tiny_wm()
    agent = RTFMMPCAgent(
        _FakeEnc(),
        _FakeTextEnc(),
        rssm,
        rew,
        N_ACT,
        "cpu",
        horizon=3,
        n_samples=32,
        n_iters=2,
        n_elites=4,
    )
    obs = {"image": np.zeros((9, 9, 3), dtype=np.uint8), "manual": "go to the tree"}
    info = {}
    agent.reset(obs, info)
    for _ in range(3):
        a = agent.act(obs, info)
        assert isinstance(a, int) and 0 <= a < N_ACT
