---
status: draft
owner: world-model
scope: local
verified: true
last_reviewed: 2026-06-17
---

# 0064: path-reward-coef sweep — does reward STRENGTH (at factor=1) move the step-2 conditional?

**Status: DONE (3 coeffs × 3 seeds, 30 rounds) — NULL / NOISE-DOMINATED.** Reward magnitude shows **no
resolvable effect** on the step-2 conditional above a large seed-to-seed variance. The reward
shape+magnitude axis now looks **exhausted** → the lever is coverage / backward-curriculum / architecture,
not reward tuning. Also a methods finding: 30-round / 3-seed n400 is too noisy to resolve effects this
small (now cheap to fix with more parallel seeds). Ran as the first [[rtfm-rollout-perf]] exploit — 9
procs concurrent in one config's wall-time, GPU ~10%.

## Result — flat within heavy seed noise (last-5 of 30 rounds, mean of 3 seeds)

| coef | step1 (swap_follow_s1) | exact (swap_follow) | conditional | swapped | grounding |
|---|---|---|---|---|---|
| 0.25 | 0.433 | 0.118 | 0.27 | 0.021 | 0.104 |
| **0.5** (ref) | 0.410 | 0.083 | 0.20 | 0.004 | 0.093 |
| 1.0 | 0.311 | 0.078 | 0.25 | 0.014 | 0.097 |

Per-seed `exact` spread (why this is a null, not an ordering): 0.25 → [0.150, 0.124, 0.080]; 0.5 →
[0.124, 0.084, 0.040]; 1.0 → [0.128, 0.018, 0.088]. The arms **overlap completely**; the conditional
(0.27 / 0.20 / 0.25) has no clean peak or monotone trend.

**Reading the fork → the "flat" branch.** None of {peak-at-0.5, monotone} holds; within noise the
conditional is flat in coef. Consistent with [[0061-rtfm-backloading-confirm]]'s "not
step-2-magnitude-limited" — now extended from *shape* (back-loading) to *overall magnitude*. All arms
read the manual (grounding ~0.09–0.10, `swapped ≈ 0`) and do step-1 at a real rate (0.31–0.43); the
2-step conditional is the wall and reward magnitude doesn't move it.

**Two caveats that keep this honest:**
1. **Sanity anchor under-reproduced.** coef-0.5 here = exact 0.083 / cond 0.20, vs the exp0061f1
   reference ~0.115 / ~0.33. Within the (large) variance band, but it confirms the protocol is
   noise-dominated — a single 2–3-seed draw at 30 rounds is not a reliable point estimate.
2. **Weak, unresolved hint only:** coef-1.0 has the lowest step-1 (0.311) with one near-collapsed seed
   (step1 0.236 / exact 0.018) — *weakly* consistent with over-rewarding destabilising step-1, but well
   inside noise. Not a claim.

**Methods lesson (now actionable cheaply):** to resolve effects of this size, beat down variance with
**more parallel seeds** (6–8/arm — still one wall-time window per [[rtfm-rollout-perf]]) rather than more
rounds. Future small-effect comparisons at n400 should not run at 3 seeds.

## Original pre-registration

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
