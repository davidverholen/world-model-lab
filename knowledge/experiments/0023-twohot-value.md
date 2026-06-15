---
status: draft
owner: world-model
scope: local
sources: [arxiv:2301.04104]
verified: true
last_reviewed: 2026-06-13
---

# 0023: Two-hot distributional critic — bound the Crafter value scale

## Hypothesis

First Crafter run (train_crafter, frozen DINO + RSSM + critic-on-replay) LEARNS
(reward 0.1→1.1) but the value INFLATES at Crafter's dense/large-scale rewards:
imagined_return → 21, critic_loss → 35 (vs ~1 on MiniGrid) — model exploitation (cf
0017/0018) the MSE critic can't bound. DreamerV3's fix: a **two-hot distributional
critic** (categorical over symlog-spaced bins, two-hot CE target) is bounded by
construction and decouples gradient scale from target magnitude. Predict: imagined_return
falls to a stable O(1–5) range, critic_loss bounded, and — over a LONG run — the
calibrated agent climbs the achievement tree instead of exploiting the model.

## Setup

`models/twohot.py` TwoHotValueHead (255 bins, symlog [-20,20]); `forward` = symexp
expectation (drop-in for ValueHead). `imagine_ac` gated: `_critic_loss` uses `twohot_loss`
if the critic supports it, else MSE — so MiniGrid (ValueHead) is byte-identical, Crafter
(TwoHot) gets the distributional loss. Critic-on-replay (β_repval 0.3) kept. WM `val` head
stays MSE (separate from the imagination critic). Isolation-tested (symlog/symexp inverse,
two-hot sums to 1, fits target 18 bounded).

## Result

**Success — the rung-3 pipeline LEARNS and climbs.** Overnight desktop run, 2 seeds, 10
rounds (round0+actor 10k steps/round, 2k WM + 2k AC updates, eval 8 eps, horizon 15,
repval 0.3, commit 26f5693). Per-round mean eval achievements (random floor = 1):

| round | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | best reward |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s0 | 1.25 | 1.62 | 1.50 | 2.25 | **3.00** | 2.62 | 3.00 | 2.38 | 2.00 | 2.75 | **2.10** @r6 |
| s1 | 1.50 | 2.25 | **3.00** | 2.62 | 2.50 | 2.62 | 1.88 | 1.62 | 2.25 | 2.50 | **1.97** @r2 |

Both climb off the random floor to **~2.5–3 achievements / reward ~2** (20× the random
reward ~0.1), no collapse, and — key — **no ignition stall** (the 0021 calibration-kills-
exploration risk did NOT materialize; Crafter's denser reward gives enough signal). Then
both **plateau at the shallow tree (~2.5–3 achievements)**, oscillating.

`imagined_return` is the weak spot: bounded-critic-LOSS held (1.5–2.7 vs MSE's 35), but the
λ-return stayed **inflated and noisy** — s0 trended down (20→5–10, self-calibrating) while
s1 stayed 17–22 with a one-round collapse to 0.56. The two-hot critic fixed the *critic*;
the **reward head** still over-predicts in imagination (accumulates over horizon 15).

## Lesson

**Two-hot critic delivered a learning rung-3 agent** (bounded critic, no blow-up, climbs
to ~3 achievements) — the value-scale fix works and the recipe (frozen DINO + RSSM +
critic-on-replay + two-hot + embedding cache) is sound. But it's a *partial* value fix:
imagined_return stays inflated/noisy because the **reward head** is the remaining
over-prediction source → next is a symlog/distributional **reward** head. Separately, the
**plateau at the shallow tree** is the headline open problem — likely representation
(global CLS can't localize resources to navigate to) and/or no temporal abstraction for the
deep tree. Backlog: (a) richer features (DINO patch tokens, exp 0024) for navigation/
gathering; (b) symlog reward head for clean value; (c) hierarchy for deep achievements.

## Links

[[0021-critic-on-replay]] · [[0019-stochastic-latents]] · [[dreamerv3-2023]] · [[crafter]]
