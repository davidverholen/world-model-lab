---
status: draft
owner: world-model
scope: local
verified: false
last_reviewed: 2026-06-17
---

# 0064: path-reward-coef sweep — does reward STRENGTH (at factor=1) move the step-2 conditional?

**Status: PRE-REGISTERED — DISPATCHING (3 coeffs × 3 seeds, in PARALLEL).** First experiment to exploit
the [[rtfm-rollout-perf]] finding: collection is CPU-world-gen-bound with the GPU idle, so 3 configs ×
3 seeds run concurrently in the wall-time of a single config (~9 procs; GPU is never the limit).

## Why

The uniform path reward at **coef 0.5** is the confirmed lever for the step-2 conditional
(P(step-2 | step-1)): [[0057-rtfm-path-reward]] lifted it 0.17→~0.30, robust across seeds in
[[0059-rtfm-floor-confirm]] / [[0061-rtfm-backloading-confirm]]. The reward *shape* axis is otherwise
exhausted — **back-loading (factor>1) was REFUTED** ([[0058-rtfm-escalating-path-reward]] /
[[0061-rtfm-backloading-confirm]]: making step-2 worth more *lowers* the conditional). What was never
swept multi-seed is the one remaining reward knob at fixed factor=1: the **magnitude** (`--path-reward-coef`).
0.5 is the established working point but its neighbours (0.25, 1.0) were never tested head-to-head.

## Hypothesis / fork

Sweep `--path-reward-coef ∈ {0.25, 0.5, 1.0}` at factor=1, decay=1.0, n400, 3 seeds each. Read **exact
swap_follow** and the **conditional** (`swap_follow / swap_follow_s1`) on held-out length-2 at the last-5
of 30 rounds, multi-seed.

- **Conditional peaks at 0.5** (0.25 and 1.0 both lower) → 0.5 is a genuine sweet spot; reward strength
  is tuned and over-rewarding (1.0) re-inflates the step-1 satisficing problem (exp0057's mechanism).
- **Monotone increasing** (1.0 ≥ 0.5 ≥ 0.25) → higher coef is better → push past 1.0 in a follow-up.
- **Flat across coeffs** → the conditional is NOT reward-magnitude-sensitive at factor=1 (consistent with
  exp0061's "not step-2-magnitude-limited") → the reward axis is done; the lever is elsewhere
  (coverage / backward curriculum / architecture).
- **Sanity anchor:** the coef-0.5 / 3-seed arm should reproduce [[0061-rtfm-backloading-confirm]] f1
  (exact ~0.115, conditional ~0.33) — a built-in check that 30-round n400 is stable.

## Setup

Three parallel dispatches, 3 seeds each, **30 rounds** (10 curriculum + 20 length-2), n400, matching the
exp0061 protocol so coef-0.5 is directly comparable:

```
scripts/dispatch_rtfm.sh exp0064a 3 --rounds 30 --curriculum-rounds 10 --length 2 \
  --manual-aux-coef 1.0 --validated-reading-coef 0.3 --n-train-seeds 400 \
  --path-reward-coef 0.25 --path-reward-factor 1 --path-reward-decay 1.0   # a = 0.25
scripts/dispatch_rtfm.sh exp0064b 3 ... --path-reward-coef 0.5  ...        # b = 0.5 (reference)
scripts/dispatch_rtfm.sh exp0064c 3 ... --path-reward-coef 1.0  ...        # c = 1.0
```

All launched concurrently (background). Read last-5 length-2 exact / conditional / step-1 per arm.

## Standing bar — diagnostic (tag: `EXTRA-RIGOR`)

Does coef magnitude move the conditional at factor=1, and where is the peak? Either result picks the next
step cleanly: tune reward magnitude further, or conclude the reward axis is exhausted and move to
coverage/curriculum/architecture.

## Links

[[0057-rtfm-path-reward]] · [[0059-rtfm-floor-confirm]] · [[0061-rtfm-backloading-confirm]] · [[0058-rtfm-escalating-path-reward]] · [[rtfm-rollout-perf]] · [[mentored-learning-loop]]
