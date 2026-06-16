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
