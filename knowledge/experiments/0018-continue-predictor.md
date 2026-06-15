---
status: draft
owner: world-model
scope: local
sources: [arxiv:2301.04104]
verified: true
last_reviewed: 2026-06-13
---

# 0018: Continue predictor — partial; exploitation is deeper (off-distribution) (run 2026-06-13)

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
(known: our buffer stores done = term OR trunc; Dreamer separates them) → [[0019-stochastic-latents]]
separates them.

## Setup

`train_dreamer.py` + ContinueHead (train_continue on real dones, success_frac 0.5
for termination examples; cumulative-continue discount in imagine_ac). 3 seeds,
DoorKey-6x6, same protocol as 0017. Desktop. Baseline: exp 0017 (imagined_return
2-3, eval 0).

## Result

3 seeds, desktop. Continue predictor PARTIALLY tamed exploitation but did not solve
it (vs 0017: 2.06/3.01/0.48, all eval 0):

| seed | imagined_return | eval |
|---|---|---|
| 0 | 1.127 (was 2.06) | 0.00 |
| 1 | **5.416 (was 3.01 — WORSE)** | 0.20 |
| 2 | 1.793 | 0.00 |

s0 dropped toward the real max (~1.0); s1 got worse; eval still ~0 everywhere.
Pre-registered fork (a) fires: returns partly bounded, but the actor still fails in
reality.

## Lesson

The continue predictor patched termination-farming, but the actor found OTHER model
inaccuracies to exploit. Deeper root cause: the continue/reward heads (and dynamics)
are unreliable OFF the real data distribution, and the actor drives imagination
*into* those hallucinated regions (s1's 5.4 = a new exploit the continue head, trained
on real beliefs, fails to flag). Underneath that: our DETERMINISTIC world model can't
accurately imagine DoorKey's ~25-step success chain (compounding error — exp 0004's
old finding), so there is no honest gradient toward real success, only toward
exploitation. Dreamer's defenses we LACK: stochastic latents (RSSM — a deterministic
model is maximally exploitable) and return normalization.

**Strategic fork (open call):**
(a) Deepen the imagination AC toward real-Dreamer: stochastic latents / shorter
    horizon (H=5, cheap one-variable test of the compounding-error hypothesis) /
    off-distribution uncertainty penalty. Bigger build, the "faithful" path.
(b) On-policy distillation (DAgger): reuse the WORKING CEM planner (~25-35%) as
    teacher, label the actor's OWN visited states, retrain — directly fixes exp
    0016's off-policy compounding error AND sidesteps imagination exploitation
    entirely (no reward-head imagination). Pragmatic, likely cheaper/robuster; keeps
    the planner in the training loop but deploys a fast actor.
Recommendation: the cheap H=5 check is worth one run (tests the compounding-error
diagnosis); the robust route to a deployable fast actor is (b) DAgger.

## Links

[[0017-imagination-actor-critic]] · [[dreamerv3-2023]] · [[imagination-training]]
