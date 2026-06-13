---
status: draft
owner: human
scope: local
sources: [arxiv:2301.04104, arxiv:1906.08253]
verified: true
last_reviewed: 2026-06-13
---

# 0021: Critic-on-replay — grounding the imagined value [CONFIRMED, kept as recipe default]

## Hypothesis

Exp 0019 found RSSM fixed policy *quality* but not value *calibration*
(`imagined_return` 0.7–4.3 vs real max ~1.0); exp 0020 refuted the behaviour-layer-reset
explanation and confirmed value miscalibration is the load-bearing problem (resetting the
critic made inflation *worse*, s2→7.3). Rather than reinvent a fix, reuse DreamerV3's
established answer to exactly this — **the critic trains on real replay returns too**
(β_repval 0.3), not just imagined λ-returns. The actor still trains only on imagination,
but its value baseline (the EMA target critic) is now anchored to real returns, so the
policy cannot chase a purely hallucinated value.

**Predict:** with `--repval 0.3`, (i) `imagined_return` falls from 2–4 toward the real
scale (~0.5–1.0) as the real-grounded EMA target pulls the λ-return bootstrap down, and
(ii) eval sustains/improves and ignition broadens beyond 1/3 seeds. Null result
(imagined_return stays inflated) would mean the burn-in grounding signal is too weak →
escalate to b2 (short branched rollouts, MBPO) or b3 (percentile return-norm + two-hot
critic).

## Setup

Single treatment arm vs the 0019 baseline (the `--repval` flag defaults 0.0, so baseline
≡ 0019 exactly). Implementation reuses the burn-in beliefs already computed in
`imagine_ac` (near-zero extra compute): capture beliefs over the real burn-in states,
regress `critic` on their real MC returns with weight 0.3, add to the imagined critic
loss. `real_bel` is detached so gradient flows to the critic only (not the world model).
3 seeds, DoorKey-6x6, all other 0019 settings identical. Held in reserve if weak: b2
MBPO short/branched rollouts; b3 DreamerV3 percentile-norm + two-hot critic.

Command per seed:
`python -m world_model.train_rssm --env-id MiniGrid-DoorKey-6x6-v0 --rounds 7
--round0-steps 20000 --actor-steps 15000 --updates-per-round 4000 --ac-updates-per-round
4000 --success-frac 0.25 --ignition-events 5 --freeze-round 2 --eval-episodes 20
--repval 0.3 --save runs/dk6_repval.pt --seed {0,1,2}`

## Result

**Prediction (i) confirmed — value calibrated; (ii) nuanced — ignition delayed, not
broadened.** 3 seeds, DoorKey-6x6, commit 7c6d1bd, `--repval 0.3`. Per-round eval +
`imagined_return`:

| seed | eval r0–6 | best | imagined_return trend |
|---|---|---|---|
| s0 | 0,0,0,0,0,0,**0.15** | 0.15 | 0.6→0.4→0.2→…→**1.04** |
| s1 | 0,0,0,0,0,0,0 | 0.00 | **3.2→1.6→1.3→1.4→1.1→1.0** |
| s2 | 0,0,0,0,0,0,**0.55** | 0.55 | 1.9→0.55→…→**1.24** |

`imagined_return` was pulled from 0019's inflated 2–7 down to **~1.0 (real scale)** —
s1's clean 3.2→1.0 is the mechanism working exactly as designed. Ignition still
occurred (s2 0.55, s0 0.15) but **late** (round 6 vs 0019's rounds 3–5), and crucially
*at honest value* (s2 0.55 with imagined_return 1.24, not inflated). Endpoint
trajectories are **opposite**: 0019's inflated runs peaked mid then collapsed at round 6
(0.55→0.00); 0021's calibrated runs are *rising* at round 6 (0.00→0.55). Aggregate best
is a wash and seed-reshuffled (0019: 0/0.55/0.35; 0021: 0.15/0/0.55).

## Lesson

**Critic-on-replay (DreamerV3 β_repval) works as advertised — independent confirmation
of the Dreamer design on our stack** (sparse MiniGrid, from scratch): it grounds the
critic in real returns and calibrates imagined value (inflated→~1). Validates both the
recipe and our implementation. Caveat: confirmed the *core mechanism* on a sparse task;
DreamerV3 pairs it with two-hot + percentile-norm on denser rewards (full recipe still
to harden — next, on the way to Crafter).

The trade is **early-ignition speed for honest value + a healthier endpoint**: less
imagined optimism → slower bootstrap (ignition delayed) but NOT prevented, and the
calibrated runs climb at the end where inflated runs collapse. For Crafter — denser
rewards, deep tree, where value inflation would be far more damaging — calibrated value
is what we want, so **critic-on-replay is kept as the recipe default** (`--repval` 0.0→0.3).
DECISION (Dave): bank the MiniGrid imagination loop as good-enough-and-now-calibrated;
don't over-polish a stepping stone (the marginal DoorKey seed isn't our bottleneck);
move to recipe-hardening → Crafter.

**Process meta-lesson (2nd time this thread):** an interim read at rounds 0–5 showed flat
zero and I concluded "calibration killed ignition" — the final round flipped it (late
ignition). Partial-data conclusions burned us again (cf. the 0019 smoke-test promotion).
Standing rule reinforced: this flywheel can ignite at the *last* round — wait for the
full run before concluding.

## Links

[[0019-stochastic-latents]] · [[0020-ac-reset]] · [[dreamerv3-2023]] ·
[[imagination-training]] · [[generative-vs-predictive]]
