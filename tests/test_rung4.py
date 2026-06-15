"""Rung-4 manual-conditioning model tests. The load-bearing one: swapping the manual must CHANGE
the predicted prior — if the prior is invariant to the manual, conditioning is broken/ignored."""

import torch

from world_model.models.actor import Actor, ConditionedActor
from world_model.models.manual_aux import (
    MaskedManualHead,
    manual_aux_loss,
    manual_invariance_diagnostic,
    random_span_mask,
)
from world_model.models.manual_conditioning import ConditionedRSSM, ManualConditioner
from world_model.models.twohot import TwoHotRewardHead


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


# ---- Masked-manual-reconstruction aux loss (Dynalang option A) ----


def test_random_span_mask_fraction_and_padding():
    torch.manual_seed(0)
    valid = torch.tensor([[True, True, True, True, False], [True, True, False, False, False]])
    masked = random_span_mask(valid, frac=0.5)
    # never mask a padding token
    assert not (masked & ~valid).any()
    # at least one real token masked per non-empty row
    assert masked[0].sum() >= 1 and masked[1].sum() >= 1
    # ~half of the 4 real tokens in row 0
    assert masked[0].sum().item() == 2


def test_manual_aux_loss_forward_backward():
    torch.manual_seed(0)
    B, L, td = 4, 6, 384
    rssm = ConditionedRSSM(embed_dim=64, num_actions=4, text_dim=td, ctx_dim=64)
    head = MaskedManualHead(belief_dim=rssm.state_dim, text_dim=td)
    state = rssm.initial(B, "cpu")
    prev_a = torch.zeros(B, dtype=torch.long)
    tokens = torch.randn(B, L, td)
    mask = torch.ones(B, L, dtype=torch.bool)
    loss, masked = manual_aux_loss(rssm, head, state, prev_a, tokens, mask, mask_frac=0.5)
    assert loss.ndim == 0 and torch.isfinite(loss) and loss.item() > 0
    assert masked.shape == (B, L)
    loss.backward()
    # gradient reaches the aux head AND the RSSM recurrent dynamics (the internalisation path)...
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in rssm.cell.parameters())
    # ...but NOT the conditioner: the anti-baking belief blocks the cross-attention shortcut.
    assert all(p.grad is None for p in rssm.conditioner.parameters())


def test_anti_baking_belief_excludes_conditioner():
    # deter_noctx must NOT equal the conditioned h (the aux belief really drops cross-attention).
    torch.manual_seed(0)
    B, L, td = 2, 5, 384
    rssm = ConditionedRSSM(embed_dim=64, num_actions=4, text_dim=td, ctx_dim=64)
    state = rssm.initial(B, "cpu")
    prev_a = torch.zeros(B, dtype=torch.long)
    tokens = torch.randn(B, L, td)
    mask = torch.ones(B, L, dtype=torch.bool)
    h_noctx = rssm.deter_noctx(state, prev_a)
    h_cond = rssm._deter_cond(state[0], state[1], prev_a, tokens, mask)
    assert not torch.allclose(h_noctx, h_cond, atol=1e-5)


# ---- OPTIONAL actor-conditioning (rung-4 §6.2 fallback, flag-gated OFF by default) ----


def test_conditioned_actor_forward_and_gradients():
    # Forward shape + gradient flows to BOTH the conditioner and the wrapped base actor.
    torch.manual_seed(0)
    B, L, bd, td, n_act = 4, 6, 288, 384, 17
    actor = ConditionedActor(belief_dim=bd, text_dim=td, num_actions=n_act, ctx_dim=64)
    belief = torch.randn(B, bd)
    tokens = torch.randn(B, L, td)
    mask = torch.ones(B, L, dtype=torch.bool)
    logits = actor(belief, tokens, mask)
    assert logits.shape == (B, n_act)
    logits.sum().backward()
    cond_ps = actor.conditioner.parameters()
    base_ps = actor.actor.parameters()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in cond_ps)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in base_ps)


def test_conditioned_actor_depends_on_manual():
    # The whole point of the fallback: the policy logits must change when the manual changes.
    torch.manual_seed(0)
    B, L, bd, td = 3, 5, 288, 384
    actor = ConditionedActor(belief_dim=bd, text_dim=td, num_actions=7, ctx_dim=64)
    belief = torch.randn(B, bd)
    mask = torch.ones(B, L, dtype=torch.bool)
    logits_a = actor(belief, torch.randn(B, L, td), mask)
    logits_b = actor(belief, torch.randn(B, L, td), mask)
    assert not torch.allclose(logits_a, logits_b, atol=1e-4)


def test_actor_logits_off_path_byte_for_byte():
    # The dispatch helper must leave the base-Actor (flag-OFF) path identical to a direct call:
    # tokens/mask are ignored, output equals actor(belief) exactly.
    from world_model.train_rtfm import actor_logits

    torch.manual_seed(0)
    B, bd, n_act = 4, 288, 17
    base = Actor(state_dim=bd, num_actions=n_act)
    belief = torch.randn(B, bd)
    tok = torch.randn(B, 6, 384)
    mask = torch.ones(B, 6, dtype=torch.bool)
    out_helper = actor_logits(base, belief, tok, mask)
    out_direct = base(belief)
    assert torch.equal(out_helper, out_direct)


def test_actor_logits_imagination_broadcast_matches_per_step():
    # The imagine_ac broadcast trick: tiling tok/mask over the horizon axis then flatten(0,1) must
    # line up row-for-row with bel_s.flatten(0,1) so each imagined step sees its OWN episode manual.
    from world_model.train_rtfm import actor_logits

    torch.manual_seed(0)
    horizon, B, bd, L, td = 3, 4, 288, 5, 384
    actor = ConditionedActor(belief_dim=bd, text_dim=td, num_actions=7, ctx_dim=64)
    bel_s = torch.randn(horizon, B, bd)
    tok = torch.randn(B, L, td)
    mask = torch.ones(B, L, dtype=torch.bool)
    flat = bel_s.flatten(0, 1)
    flat_tok = tok.unsqueeze(0).expand(horizon, *tok.shape).flatten(0, 1)
    flat_mask = mask.unsqueeze(0).expand(horizon, *mask.shape).flatten(0, 1)
    flat_logits = actor_logits(actor, flat, flat_tok, flat_mask)
    # Each (step h, episode i) row must equal a single-row call with episode i's manual.
    for h in range(horizon):
        for i in range(B):
            row = actor_logits(actor, bel_s[h, i : i + 1], tok[i : i + 1], mask[i : i + 1])
            assert torch.allclose(flat_logits[h * B + i], row[0], atol=1e-5)


def test_manual_invariance_diagnostic_runs():
    torch.manual_seed(0)
    B, L, td = 4, 6, 384
    rssm = ConditionedRSSM(embed_dim=64, num_actions=4, text_dim=td, ctx_dim=64)
    head = MaskedManualHead(belief_dim=rssm.state_dim, text_dim=td)
    state = rssm.initial(B, "cpu")
    prev_a = torch.zeros(B, dtype=torch.long)
    tokens = torch.randn(B, L, td)
    mask = torch.ones(B, L, dtype=torch.bool)
    diag = manual_invariance_diagnostic(rssm, head, state, prev_a, tokens, mask, mask_frac=0.5)
    assert set(diag) == {"correct", "wrong", "ratio"}
    assert diag["correct"] > 0 and diag["wrong"] > 0
    # untrained head: both losses finite; ratio is a finite number (not asserting direction yet —
    # that's the thing TRAINING is supposed to produce; here we just exercise the wiring).
    assert all(v == v for v in diag.values())  # no NaN


def test_conservative_reward_penalty_pushes_ood_reward_down():
    """exp 0047 (CROP/CQL): optimizing ONLY the conservative penalty must drive the reward head's
    predicted reward on random (OOD) actions down. Validates sign + gradient flow of the term used
    in train_rtfm.wm_train_rtfm. The penalty is one-sided (relu) so it only suppresses positive
    overestimates."""
    torch.manual_seed(0)
    B, SD, NA, M = 8, 96, 17, 16
    rew = TwoHotRewardHead(state_dim=SD, num_actions=NA)
    belief = torch.randn(B, SD)  # detached fixed beliefs (the train loop detaches the belief)

    @torch.no_grad()
    def ood_mean():
        rand_a = torch.randint(0, NA, (B, M))
        bel_exp = belief.unsqueeze(1).expand(B, M, SD)
        return float(rew(bel_exp, rand_a).mean())

    # Induce a CONTROLLED (non-saturating) overestimate: regress predicted reward on random actions
    # toward +1.5 via the head's own twohot loss, so there is a positive OOD value to push down.
    warm = torch.optim.Adam(rew.parameters(), lr=5e-3)
    for _ in range(60):
        rand_a = torch.randint(0, NA, (B, M))
        bel_exp = belief.unsqueeze(1).expand(B, M, SD)
        tgt = torch.full((B, M), 1.5)
        loss = rew.twohot_loss(bel_exp, rand_a, tgt)
        warm.zero_grad()
        loss.backward()
        warm.step()
    before = ood_mean()
    assert before > 0.5  # the head now overrates OOD actions

    # Apply the conservative penalty (the exact term from wm_train_rtfm) and confirm it pushes down.
    opt = torch.optim.Adam(rew.parameters(), lr=5e-3)
    for _ in range(120):
        rand_a = torch.randint(0, NA, (B, M))
        bel_exp = belief.unsqueeze(1).expand(B, M, SD)
        cons = rew(bel_exp, rand_a).clamp_min(0.0).mean()
        opt.zero_grad()
        cons.backward()
        opt.step()
    after = ood_mean()
    assert after < before  # OOD predicted reward was pushed down


def test_validated_reading_reward_marginal_sign():
    """exp 0048: intrinsic validated-reading reward = clip≥0(err_without_manual − err_with_manual)
    against the REAL next embedding. If reality matches the WITH-manual prediction, the manual
    helped → reward > 0; if reality matches the no-manual (deter_noctx) prediction, the manual did
    not help → reward clips to 0. Pins the sign + the marginal (deter_noctx) baseline."""
    from world_model.train_rtfm import validated_reading_reward

    torch.manual_seed(0)
    ED, NA, TD = 64, 5, 384
    rssm = ConditionedRSSM(embed_dim=ED, num_actions=NA, text_dim=TD, ctx_dim=64)
    recon = torch.nn.Linear(rssm.state_dim, ED)
    state = rssm.initial(1, "cpu")
    action = torch.zeros(1, dtype=torch.long)
    tok = torch.randn(1, 5, TD)
    mask = torch.ones(1, 5, dtype=torch.bool)

    # Rebuild the two predicted next embeddings exactly as the reward helper does.
    h, z = state
    h_with = rssm._deter_cond(h, z, action, tok, mask)
    pred_with = recon(rssm.belief((h_with, rssm._dist(rssm.prior_net(h_with)).mean)))
    h_no = rssm.deter_noctx(state, action)
    pred_no = recon(rssm.belief((h_no, rssm._dist(rssm.prior_net(h_no)).mean)))
    assert not torch.allclose(pred_with, pred_no)  # the manual actually changes the prediction

    r_helps = validated_reading_reward(rssm, recon, state, action, tok, mask, pred_with.detach())
    r_nohelp = validated_reading_reward(rssm, recon, state, action, tok, mask, pred_no.detach())
    assert isinstance(r_helps, float) and isinstance(r_nohelp, float)
    assert r_helps > 0.0  # reality = with-manual pred → err_with=0, err_no>0 → positive marginal
    assert r_nohelp == 0.0  # reality = no-manual pred → err_no=0, err_with≥0 → clipped to 0
