---
status: draft
owner: world-model
scope: local
verified: false
last_reviewed: 2026-06-17
---

# 0058: escalating per-step path reward — reward later gesture steps MORE

**Status: DONE — CONFOUNDED NEGATIVE (base coef too low).** The escalation arms scored BELOW baseline
(step-1 collapsed to 0.13–0.17 vs 0.28), but the cause is a parameterization confound, not a refutation
of back-loading. Lesson + the fix below. The maintainer's lever: *increase the reward for each step by
a factor.* Targets the [[0057-rtfm-path-reward]] finding that a UNIFORM per-step reward leaves step-2
under-optimized.

## Result — confounded negative: I under-fed step-1

`--path-reward-coef 0.2 --path-reward-factor {3,5,10} --path-reward-decay 0.5`, last-5 length-2:

| factor | step-1 | exact | P(step-2\|step-1) | imagined_return |
|---|---|---|---|---|
| 3 | 0.126 | 0.022 | 0.17 | 1.4 |
| 5 | 0.140 | 0.008 | 0.06 | 1.3 |
| 10 | 0.174 | 0.024 | 0.14 | 1.4 |

All below baseline. **Confound:** base `coef 0.2` gave step-1 a reward of only 0.2, but
[[0057-rtfm-path-reward]] showed step-1 needs ~**0.5** (floor 0.5 → step1 0.36; floor 0.2 → starts to
work). Plus `decay 0.5` halved repeats. So step-1 was **starved** (0.13–0.17), and you can't complete a
chain you rarely start — the low exact follows mechanically. imagined_return ~1.4 (vs baseline ~8–12)
confirms the arms were under-rewarded overall. **NOT a clean test of back-loading.**

**Lesson: step-1 reward magnitude is load-bearing** — match the working base (~0.5) before escalating.
Fix (deferred, lower priority since uniform floor-0.5 already works): re-run with
`--path-reward-coef 0.5 --path-reward-factor {2,3} --path-reward-decay 1.0` so step-1 gets the working
0.5 and step-2 gets 0.5·factor on top. Prioritised instead: multi-seed confirm of the floor-0.5 winner
([[0059]]) and coverage×floor combination ([[0060]]).

## Original pre-registration

## Why (the satisficing argument)

With a uniform per-step path reward, each step's contribution to the objective is
P(reach it)·coef. Measured: step-1 ≈ 0.42·coef; step-2 ≈ 0.42·0.17·coef ≈ **0.07·coef** — so step-2
contributes ~**6× less** than step-1. The actor rationally satisfices on the easy, high-contribution
step-1 (exp0057: keeping a uniform path reward lifted step-1 0.28→0.42 but left conditional step-2 at
~0.17). **Reward step-k with `coef·factor**k` (factor>1)** → step-2's contribution rises to ~match
step-1's (factor ≈ 6 equalises), giving the actor a reason to invest in the chain's tail. Distinct
from [[0057-rtfm-path-reward]] (which raises both steps equally and keeps the step-1 satisfaction).

## Setup (BUILT)

- `collect_rtfm`: escalating path reward computed on OUR side — track an in-order pointer over the
  displayed gesture; reward `coef·factor**gptr` on each correct next action, reset on a miss (mirrors
  the env's `reading_shaping` but escalating). When active, env `reading_shaping` is disabled (no
  double count). Flags `--path-reward-coef`, `--path-reward-factor` (0 = off, byte-for-byte unchanged).
- Honest: collection-only, CORRECT-mode (displayed == true recipe); eval stays swapped/held-out.
- Reviewer (opus): SHIP (faithful escalating analog, no double-count, no eval leak, backward-compat).

## Reward design (maintainer-refined)

Two reward channels, cleanly separated:
- **Tutorial-execution (path reward, intrinsic, repeatable):** `coef · factor**k · decay**n` for the
  k-th in-order gesture step on its n-th in-episode payout. `factor>1` = later steps worth more
  (escalation, the core lever); `decay<1` = repeated in-episode executions worth progressively less
  (diminishing return — the skill stays rewarded every time but re-runs can't farm unbounded reward).
  `decay 1.0` = reward every time; `0.0` = once per episode. Rewarded EVERY correct execution.
- **Achievement (`tutorial_newly`, extrinsic):** one-time, only on actual game progress. (Base Crafter
  reward is discarded.)

## Pre-registered sweep

`--path-reward-coef 0.2 --path-reward-factor ∈ {3, 5, 10} --path-reward-decay 0.5` (step-2 worth 0.6 /
1.0 / 2.0 on first execution; factor≈6 equalises the per-step contribution; decay 0.5 = each repeat
worth half), baseline else (folded VR 0.3, n-train 400, length 2). Watch **conditional
P(step-2|step-1)** and exact swap_follow.

- Conditional step-2 **rises** with factor → satisficing was the limiter; the chain's tail just needed
  to be worth optimizing.
- Conditional step-2 **flat** while step-1 holds → step-2 is **exposure-starved** (rarely reached), not
  magnitude-starved → next lever = practice step-2 directly (backward curriculum / imagine from
  step-1-done states).

## Standing bar — diagnostic/LADDER (tag: `EXTRA-RIGOR`)

Moves the conditional step-2 (the real nut) above ~0.17, or definitively rules out reward-magnitude as
the lever (→ exposure).

## Links

[[0057-rtfm-path-reward]] · [[0055-rtfm-ensemble-pessimism]] · [[0056-rtfm-coverage-sweep]] · [[0043-rtfm-execution-wall]] · [[mentored-learning-loop]]

## exp0061 — back-loading done RIGHT (base 0.5): REFUTED

Re-ran at the working base (`--path-reward-coef 0.5 --path-reward-factor {1,2,3} --path-reward-decay
1.0`, 2 seeds each), last-5 length-2:

| factor | step-1 | exact | conditional |
|---|---|---|---|
| **1 (uniform 0.5)** | 0.35 | **0.115** | **0.33** |
| 2 | 0.39 | 0.081 | 0.21 |
| 3 | 0.49 | 0.097 | 0.20 |

**Back-loading HURTS the conditional** (0.33→0.21→0.20). Bigger step-2 reward makes the agent attempt
step-1 MORE (chasing the payoff → step-1 up to 0.49) but follow through WORSE. So the conditional is NOT
step-2-reward-magnitude-limited → the satisficing hypothesis is **refuted**; the bottleneck is
EXPOSURE / execution (data/coverage axis). **Uniform 0.5, every time (factor 1) is the robust reward
winner** (exact 0.115, cond 0.33) — a 2nd confirmation of the floor-0.5 lever. → pivot the conditional
push to exposure (coverage confirm exp0062 / backward curriculum), not reward shape.
