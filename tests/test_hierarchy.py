"""CPU tests for the Director-style hierarchy models (exp 0050). The two-level imagination loop is
exercised by a separate integration smoke; here we pin the model pieces."""

import torch

from world_model.models.hierarchy import (
    GoalCodebook,
    GoalWorker,
    Manager,
    goal_similarity,
    make_hierarchy,
)


def test_codebook_fit_and_nearest():
    torch.manual_seed(0)
    dim, K = 32, 8
    cb = GoalCodebook(K, dim)
    assert not bool(cb.fitted)
    # three well-separated clusters → fit should place centroids near them
    centers = (torch.zeros(dim), 5 * torch.ones(dim), -5 * torch.ones(dim))
    beliefs = torch.cat([torch.randn(60, dim) + c for c in centers])
    cb.fit(beliefs)
    assert bool(cb.fitted) and cb.centroids.shape == (K, dim)
    assert torch.isfinite(cb.centroids).all()
    idx = cb.nearest(beliefs[:5])
    assert idx.shape == (5,) and idx.dtype == torch.long and (idx < K).all()
    assert cb.vecs(idx).shape == (5, dim)


def test_codebook_fit_fewer_points_than_codes():
    cb = GoalCodebook(16, 8)
    cb.fit(torch.randn(4, 8))  # N < n_codes → pad-sample path, must not crash
    assert bool(cb.fitted) and cb.centroids.shape == (16, 8)


def test_goal_similarity_range_and_sign():
    g = torch.randn(4, 16)
    assert torch.allclose(goal_similarity(g, g), torch.ones(4), atol=1e-5)  # self-sim = 1
    assert goal_similarity(g, -g).max() < -0.99  # opposite = -1


def test_goal_worker_forward_and_grad():
    torch.manual_seed(0)
    B, sd, na = 4, 48, 17
    w = GoalWorker(sd, sd, na)
    belief, goal = torch.randn(B, sd), torch.randn(B, sd)
    logits = w(belief, goal)
    assert logits.shape == (B, na)
    logits.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in w.parameters())
    # the goal must actually influence the policy
    assert not torch.allclose(w(belief, goal), w(belief, torch.randn(B, sd)), atol=1e-5)


def test_manager_forward_state_only_and_with_ctx():
    torch.manual_seed(0)
    B, sd, K = 4, 48, 32
    m = Manager(sd, K)
    assert m(torch.randn(B, sd)).shape == (B, K)
    mc = Manager(sd, K, ctx_dim=16)  # Phase B: manual-conditioned manager
    out = mc(torch.randn(B, sd), torch.randn(B, 16))
    assert out.shape == (B, K)
    out.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mc.parameters())


def test_make_hierarchy_shapes():
    sd, na, K = 48, 17, 32
    worker, wc, manager, mc, cb = make_hierarchy(sd, na, K)
    assert isinstance(worker, GoalWorker) and isinstance(manager, Manager)
    assert isinstance(cb, GoalCodebook) and cb.centroids.shape == (K, sd)
    # worker critic consumes concat(belief, goal); manager critic consumes belief
    assert wc(torch.randn(2, sd + sd)).shape == (2,)
    assert mc(torch.randn(2, sd)).shape == (2,)


# ---- integration: the two-level imagination loop runs end-to-end ----


class _FakeBuf:
    def __init__(self, ed, na):
        self.ed, self.na = ed, na

    def sample_sequences(self, b, length, success_frac=0.0):
        import numpy as np

        return {
            "obs": np.random.randn(b, length, self.ed).astype("float32"),
            "action": np.random.randint(0, self.na, (b, length)),
            "reward": np.random.rand(b, length).astype("float32"),
            "return": np.random.rand(b, length).astype("float32"),
            "done": np.zeros((b, length), dtype="float32"),
            "tag": np.zeros(b, dtype="int64"),
        }


class _FakeReg:
    def __init__(self, td):
        self.td = td

    def tokens(self, tags):
        n = len(tags)
        return torch.randn(n, 5, self.td), torch.ones(n, 5, dtype=torch.bool)


def test_imagine_hierarchy_runs_end_to_end():
    """The load-bearing integration test: the two-level loop (manager picks codes, worker reaches
    them, both train) executes one+ update on a tiny real RSSM and returns finite stats — pins the
    shape/flatten/lambda-return wiring of imagine_hierarchy_rtfm."""
    import copy

    from world_model.models.continue_head import ContinueHead
    from world_model.models.manual_conditioning import ConditionedRSSM
    from world_model.models.twohot import TwoHotRewardHead
    from world_model.train_rtfm import imagine_hierarchy_rtfm

    torch.manual_seed(0)
    ED, NA, TD = 32, 17, 384
    rssm = ConditionedRSSM(embed_dim=ED, num_actions=NA, text_dim=TD, ctx_dim=32)
    rew = TwoHotRewardHead(state_dim=rssm.state_dim, num_actions=NA)
    cont = ContinueHead(rssm.state_dim)
    worker, wcrit, manager, mcrit, cb = make_hierarchy(rssm.state_dim, NA, n_codes=16)
    cb.fit(torch.randn(64, rssm.state_dim))
    wtgt, mtgt = copy.deepcopy(wcrit), copy.deepcopy(mcrit)
    opt = torch.optim.Adam(
        [
            *worker.parameters(),
            *wcrit.parameters(),
            *manager.parameters(),
            *mcrit.parameters(),
        ],
        lr=1e-3,
    )
    stats = imagine_hierarchy_rtfm(
        _FakeBuf(ED, NA),
        _FakeReg(TD),
        rssm,
        rew,
        cont,
        worker,
        wcrit,
        wtgt,
        manager,
        mcrit,
        mtgt,
        cb,
        opt,
        updates=2,
        seq_batch=4,
        window=6,
        burn_in=2,
        horizon=4,
        hier_k=2,
        gamma=0.99,
        lam=0.95,
        ent_coef=0.01,
        device="cpu",
        hindsight_frac=0.5,
    )
    # (worker_loss, manager_loss, macro_r, worker_sim, code_entropy, codes_used)
    assert len(stats) == 6 and all(v == v for v in stats)  # no NaN
    assert -1.01 <= stats[3] <= 1.01  # worker goal-similarity is a cosine in [-1, 1]
    assert stats[4] >= 0.0 and 1 <= stats[5] <= 16  # code entropy ≥0; codes_used in [1, n_codes]
