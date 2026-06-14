"""Rung-4 manual-conditioning model tests. The load-bearing one: swapping the manual must CHANGE
the predicted prior — if the prior is invariant to the manual, conditioning is broken/ignored."""

import torch

from world_model.models.manual_conditioning import ConditionedRSSM, ManualConditioner


def test_manual_conditioner_shapes_and_mask():
    B, L, qd, td, cd = 4, 7, 32, 384, 64
    cond = ManualConditioner(query_dim=qd, text_dim=td, ctx_dim=cd, n_heads=4)
    q = torch.randn(B, qd)
    tokens = torch.randn(B, L, td)
    mask = torch.ones(B, L, dtype=torch.bool)
    out = cond(q, tokens, mask)
    assert out.shape == (B, cd)
    # masking out all-but-first token must change the context (the masked tokens were attended)
    mask2 = mask.clone()
    mask2[:, 1:] = False
    assert not torch.allclose(out, cond(q, tokens, mask2), atol=1e-5)


def test_conditioned_rssm_steps_and_belief():
    B, L, td = 3, 6, 384
    rssm = ConditionedRSSM(embed_dim=128, num_actions=5, text_dim=td, ctx_dim=64)
    state = rssm.initial(B, "cpu")
    prev_a = torch.zeros(B, dtype=torch.long)
    tokens = torch.randn(B, L, td)
    mask = torch.ones(B, L, dtype=torch.bool)
    embed = torch.randn(B, 128)
    s2, prior, post = rssm.obs_step(state, prev_a, embed, tokens, mask)
    assert rssm.belief(s2).shape == (B, rssm.state_dim)
    assert prior.mean.shape == (B, rssm.stoch_dim) == post.mean.shape
    s3, prior_i = rssm.img_step(s2, prev_a, tokens, mask)
    assert rssm.belief(s3).shape == (B, rssm.state_dim)


def test_swapping_manual_changes_the_prior():
    # THE anti-baking property at the model level: the predicted dynamics (prior) must depend on
    # the manual. Same state + action, two different manuals → different prior.
    torch.manual_seed(0)
    B, L, td = 2, 5, 384
    rssm = ConditionedRSSM(embed_dim=64, num_actions=4, text_dim=td, ctx_dim=64)
    state = rssm.initial(B, "cpu")
    prev_a = torch.zeros(B, dtype=torch.long)
    mask = torch.ones(B, L, dtype=torch.bool)
    manual_a = torch.randn(B, L, td)
    manual_b = torch.randn(B, L, td)
    _, prior_a = rssm.img_step(state, prev_a, manual_a, mask)
    _, prior_b = rssm.img_step(state, prev_a, manual_b, mask)
    assert not torch.allclose(prior_a.mean, prior_b.mean, atol=1e-4)


def test_gradient_flows_to_conditioner():
    B, L, td = 2, 5, 384
    rssm = ConditionedRSSM(embed_dim=64, num_actions=4, text_dim=td, ctx_dim=64)
    state = rssm.initial(B, "cpu")
    tokens = torch.randn(B, L, td)
    mask = torch.ones(B, L, dtype=torch.bool)
    _, prior = rssm.img_step(state, torch.zeros(B, dtype=torch.long), tokens, mask)
    prior.mean.sum().backward()
    grads = [p.grad for p in rssm.conditioner.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
