---
status: draft
owner: human
scope: local
sources: [arxiv:2301.04104]
verified: true
last_reviewed: 2026-06-13
---

# 0018: Continue predictor fixes imagination reward-farming (planned) [autonomous]

## Hypothesis

Single-variable fix for exp 0017's model exploitation. Adding Dreamer's continue
predictor c(s)∈[0,1] — trained on real done flags, used to discount imagined
λ-returns by the per-step gamma·c_t (zeroing reward past a predicted termination) —
stops the actor farming the goal reward across the un-terminated horizon. Expect:
`imagined_return` drops to ≤ ~1.0 (the real per-episode max) AND `eval_success`
rises off 0 (target: ≥ the planner's ~25-35% on DoorKey-6x6, 20 eps).

Diagnostic forks if it doesn't fully work: (a) imagined_return ≤1 but eval still 0
→ off-distribution problem (the actor reaches goal in imagination but not reality;
world-model accuracy / exploration, not termination); (b) imagined_return
over-suppressed (too low) → continue head conflates truncation with termination
(known: our buffer stores done = term OR trunc; Dreamer separates them) → exp 0019
separates them.

## Setup

`train_dreamer.py` + ContinueHead (train_continue on real dones, success_frac 0.5
for termination examples; cumulative-continue discount in imagine_ac). 3 seeds,
DoorKey-6x6, same protocol as 0017. Desktop. Baseline: exp 0017 (imagined_return
2-3, eval 0).

## Result

_pending_

## Lesson

_pending_

## Links

[[0017-imagination-actor-critic]] · [[dreamerv3-2023]] · [[imagination-training]]
