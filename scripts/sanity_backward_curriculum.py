"""Sanity check for the exp-0066 imagination backward curriculum (default-off training lever).

knowledge/design/hierarchical-imagination-agent.md §3a. Self-contained, CPU-ok, < ~1 min. Asserts:
  (a) the collection loop TAGS buffered transitions with gesture-prefix progress (`gesture_pos`),
      and prefix-done transitions are identifiable;
  (b) with --backward-curriculum the imagination START sampler draws a HIGHER fraction of
      prefix-done starts than the baseline (uniform) distribution — the core oversampling claim;
  (c) the AC update still runs under the curriculum with no NaNs/Infs.

Part (b) is verified on a SYNTHETIC buffer with a controlled prefix-done rate (deterministic, does
not depend on a tiny random policy actually completing step-1), and the live collection+AC path is
smoke-tested separately on a tiny crafter-rtfm config.

Run: uv run python scripts/sanity_backward_curriculum.py
"""

import math

import numpy as np
import torch

from world_model.training import ReplayBuffer, Transition


def _emb(d, v):
    return np.full((d,), float(v), dtype=np.float32)


def part_b_synthetic_oversampling() -> None:
    """Controlled buffer: ~20% of imagination-start positions are prefix-done. Assert the
    curriculum sampler oversamples them well above that baseline, and the baseline sampler matches
    the underlying rate."""
    d, burn_in, length = 4, 4, 8
    off = burn_in - 2  # reviewer-confirmed imagination-start offset (see prefix_done_starts)
    buf = ReplayBuffer(capacity=20_000, obs_shape=(d,), seed=0, obs_dtype=np.float32)
    rng = np.random.default_rng(0)
    n_eps, ep_len = 200, 16  # contiguous episodes (done at the last step) so windows are valid
    for _ in range(n_eps):
        # Per episode, pick whether its tail is "prefix-done" (gesture_pos>=1) with prob ~0.2.
        done_ep = rng.random() < 0.2
        for t in range(ep_len):
            gp = 1 if (done_ep and t >= ep_len // 2) else 0
            last = t == ep_len - 1
            buf.add(Transition(_emb(d, t), 0, 0.0, _emb(d, t + 1), last, gesture_pos=gp))
    buf.compute_returns(0.99)

    pool = buf.prefix_done_starts(length, burn_in, min_pos=1)
    # all valid starts, to get the true baseline prefix-done rate
    all_cand = np.arange(0, buf.size - length + 1)
    valid = buf._valid_starts(all_cand, length)
    baseline_rate = float(np.mean(buf.gesture_pos[valid + off] >= 1))
    assert len(pool) > 0, "expected a non-empty prefix-done pool in the synthetic buffer"
    assert 0.02 < baseline_rate < 0.4, f"unexpected baseline prefix rate {baseline_rate:.3f}"

    # curriculum sampler with bc_frac=0.5 should realize ~>= 0.5 prefix-done starts
    realized = []
    for _ in range(20):
        _batch, frac = buf.sample_sequences_prefix_curriculum(
            batch_size=64, length=length, burn_in=burn_in, bc_frac=0.5, min_pos=1
        )
        realized.append(frac)
    cur_frac = float(np.mean(realized))
    # baseline (uniform) realized prefix-done rate, for the A/B contrast: sample many uniform valid
    # starts directly (the distribution sample_sequences draws from at success_frac=0)
    uni = buf._valid_starts(rng.integers(0, buf.size - length, size=20_000), length)
    base_frac = float(np.mean(buf.gesture_pos[uni + off] >= 1))

    assert cur_frac > base_frac + 0.15, (
        f"curriculum did NOT oversample: cur={cur_frac:.3f} vs baseline={base_frac:.3f}"
    )
    assert cur_frac >= 0.45, f"curriculum realized frac {cur_frac:.3f} below bc_frac~0.5"
    # empty-pool fallback: a buffer with no prefix-done transitions must not crash, frac ~0
    empty = ReplayBuffer(capacity=5000, obs_shape=(d,), seed=1, obs_dtype=np.float32)
    for _ in range(50):
        for t in range(ep_len):
            empty.add(
                Transition(_emb(d, t), 0, 0.0, _emb(d, t + 1), t == ep_len - 1, gesture_pos=0)
            )
    empty.compute_returns(0.99)
    _b, ef = empty.sample_sequences_prefix_curriculum(64, length, burn_in, bc_frac=0.5, min_pos=1)
    assert ef == 0.0, f"empty pool should realize 0 prefix-done starts, got {ef}"
    print(
        f"(b) OK: curriculum oversamples — baseline prefix frac {base_frac:.3f} "
        f"-> curriculum {cur_frac:.3f}; empty-pool fallback realized {ef:.3f}"
    )


def part_ac_collection_smoke() -> None:
    """Tiny live crafter-rtfm collection + AC update: assert (a) the collection loop populates
    `gesture_pos` (tracking works) and (c) the curriculum AC update runs with no NaNs."""
    import crafter_rtfm as C  # noqa: F401  (import here so part (b) runs without the rtfm extra)
    from crafter_rtfm import ManualMode

    from world_model.models import Actor, FrozenDinoEncoder
    from world_model.models.continue_head import ContinueHead
    from world_model.models.manual_conditioning import ConditionedRSSM
    from world_model.models.text_encoder import FrozenTextEncoder
    from world_model.models.twohot import TwoHotRewardHead, TwoHotValueHead
    from world_model.train_rtfm import (
        ManualRegistry,
        RTFMAgent,
        collect_rtfm,
        imagine_ac_rtfm,
    )

    device = "cpu"
    torch.manual_seed(0)
    np.random.seed(0)
    enc = FrozenDinoEncoder(pool="cls+patch").to(device)
    text_enc = FrozenTextEncoder(device=device)
    ed, td = enc.latent_dim, text_enc.text_dim
    n_act = 17
    rssm = ConditionedRSSM(embed_dim=ed, num_actions=n_act, text_dim=td).to(device)
    sd = rssm.state_dim
    rew = TwoHotRewardHead(state_dim=sd, num_actions=n_act).to(device)
    cont = ContinueHead(state_dim=sd).to(device)
    actor = Actor(state_dim=sd, num_actions=n_act).to(device)
    critic = TwoHotValueHead(state_dim=sd).to(device)
    target_critic = __import__("copy").deepcopy(critic)
    opt_ac = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=3e-4)

    registry = ManualRegistry(text_enc)
    buf = ReplayBuffer(20_000, (ed,), seed=0, obs_dtype=np.float32)
    agent = RTFMAgent(enc, text_enc, rssm, actor, n_act, device, epsilon=0.5, seed=0)
    seeds = list(range(8))

    # A high epsilon + a few episodes; path_reward_coef on so the pointer is exercised. Tracking is
    # independent of use_path, but turning it on also exercises the reward path.
    for rnd in range(3):
        collect_rtfm(
            buf,
            registry,
            agent,
            n_episodes=4,
            length=2,
            seeds=seeds,
            mode=ManualMode.CORRECT,
            device=device,
            use_agent=(rnd > 0),
            max_steps=12,
            one_shot=True,
            path_reward_coef=1.0,
        )
    buf.compute_returns(0.99)

    gp = buf.gesture_pos[: buf.size]
    tracked = int(np.sum(gp >= 0))
    assert tracked > 0, "collection did not populate gesture_pos at all"
    # gesture_pos must be either -1 (untracked) or a valid in-order progress count 0..len(gesture)
    assert set(np.unique(gp)).issubset({-1, 0, 1, 2}), (
        f"unexpected gesture_pos values {np.unique(gp)}"
    )
    print(
        f"(a) OK: collection tagged {tracked}/{buf.size} transitions; "
        f"prefix-done (>=1) count = {int(np.sum(gp >= 1))}; values seen = {sorted(np.unique(gp))}"
    )

    # (c) AC update runs under the curriculum without NaN/Inf. burn_in small to keep windows valid.
    al, cl, ir, bc_realized = imagine_ac_rtfm(
        buf,
        registry,
        rssm,
        rew,
        cont,
        actor,
        critic,
        target_critic,
        opt_ac,
        updates=3,
        seq_batch=8,
        window=8,
        burn_in=2,
        horizon=4,
        gamma=0.99,
        lam=0.95,
        ent_coef=3e-3,
        device=device,
        backward_curriculum=True,
        bc_frac=0.5,
    )
    for name, v in [("actor_loss", al), ("critic_loss", cl), ("imagined_return", ir)]:
        assert math.isfinite(v), f"{name} is not finite: {v}"
    print(
        f"(c) OK: curriculum AC update ran — actor_loss={al:.3f} critic_loss={cl:.3f} "
        f"imagined_return={ir:.3f} bc_realized_frac={bc_realized:.2f} (no NaNs)"
    )


def part_d_offset_pin() -> None:
    """Discriminating test for the imagination-START offset (the exp0066 reviewer-caught bug). Tag
    gesture_pos=1 at a SINGLE known position per episode; the corrected offset (burn_in-2) must land
    the prefix-done pool exactly on those tags, and the OLD offset (burn_in-1) must NOT — so a
    regression to burn_in-1 fails here (the original whole-tail test passed under either)."""
    d, burn_in, length = 4, 4, 8
    ep_len, n_eps, p_local = 16, 60, 8  # one tag per episode at local index 8
    buf = ReplayBuffer(capacity=20_000, obs_shape=(d,), seed=0, obs_dtype=np.float32)
    for _ in range(n_eps):
        for t in range(ep_len):
            gp = 1 if t == p_local else 0
            buf.add(Transition(_emb(d, t), 0, 0.0, _emb(d, t + 1), t == ep_len - 1, gesture_pos=gp))
    buf.compute_returns(0.99)

    pool = buf.prefix_done_starts(length, burn_in, min_pos=1)
    assert len(pool) > 0, "expected a non-empty single-tag prefix-done pool"
    # The corrected offset (burn_in-2) lands every selected start on a tagged position:
    assert np.all(buf.gesture_pos[pool + (burn_in - 2)] == 1), "pool not aligned to burn_in-2"
    # ...and the OLD offset (burn_in-1) would land one step PAST the tag (gesture_pos 0 there):
    assert np.all(buf.gesture_pos[pool + (burn_in - 1)] == 0), (
        "pool aligns to burn_in-1 — the off-by-one regressed"
    )
    print(
        f"(d) OK: offset pinned to burn_in-2 — {len(pool)} starts at S+{burn_in - 2}==1, "
        f"none at S+{burn_in - 1}"
    )


def main() -> None:
    part_b_synthetic_oversampling()
    part_d_offset_pin()
    part_ac_collection_smoke()
    print("\nALL SANITY CHECKS PASSED")


if __name__ == "__main__":
    main()
