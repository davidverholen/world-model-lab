---
status: draft
owner: world-model
scope: local
verified: false
last_reviewed: 2026-06-16
---

# 0058: escalating per-step path reward — reward later gesture steps MORE

**Status: BUILT + reviewer-passed (SHIP), CPU-smoked; dispatch when a GPU slot frees.** The
maintainer's lever: *increase the reward for each step by a factor.* Targets the
[[0057-rtfm-path-reward]] finding that a UNIFORM per-step reward leaves step-2 under-optimized.

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

## Pre-registered sweep

`--path-reward-coef 0.2 --path-reward-factor ∈ {3, 5, 10}` (step-2 worth 0.6 / 1.0 / 2.0; factor≈6
equalises the per-step contribution), baseline else (folded VR 0.3, n-train 400, length 2). Watch
**conditional P(step-2|step-1)** and exact swap_follow.

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
