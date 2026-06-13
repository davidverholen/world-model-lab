---
status: draft
owner: human
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

Local-GPU smoke (rounds 0→1, same config that gave MSE imagined_return 21): two-hot gives
**imagined_return 6.11→5.39 (stable)**, **critic_loss 2.5→2.2** — the blow-up is gone and
stable across rounds (MSE grew 9.95→21.67). Eval flat (0.10/1.0) in 2 noisy rounds — the
calibration-delays-ignition pattern (cf 0021); the climb test needs a long run.
_Overnight desktop run pending → fill curve + achievement trajectory._

## Lesson

_pending (overnight run)_

## Links

[[0021-critic-on-replay]] · [[0019-stochastic-latents]] · [[dreamerv3-2023]] · [[crafter]]
