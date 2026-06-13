---
status: draft
owner: human
scope: local
sources: [arxiv:2301.04104, arxiv:1906.08253]
verified: true
last_reviewed: 2026-06-13
---

# 0021: Critic-on-replay — grounding the imagined value (pre-registered) [reuse DreamerV3]

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

_pending (dispatched)_

## Lesson

_pending_

## Links

[[0019-stochastic-latents]] · [[0020-ac-reset]] · [[dreamerv3-2023]] ·
[[imagination-training]] · [[generative-vs-predictive]]
