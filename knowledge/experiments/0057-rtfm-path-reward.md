---
status: draft
owner: world-model
scope: local
verified: false
last_reviewed: 2026-06-16
---

# 0057: reward the PATH — keep the per-step gesture reward instead of annealing it away

**Status: IN FLIGHT.** Follows the [[0055-rtfm-ensemble-pessimism]] reframe (wall = 2-step
composition, not reading/reward) and the maintainer's framing: *the path isn't rewarded — but
path-following is exactly what we want to train.* Objective↔reward alignment — we MEASURE
path-following (swap_follow), so we should REWARD it.

## Key realization: the path reward already exists, and we turn it off

The env's `reading_shaping` (HO-0007) is *already* a per-step path reward: +coef each time the agent
performs the NEXT action of the displayed gesture (pointer advances; resets on a wrong action) — so
step-1 AND step-2 are rewarded, reading-gated. **But we anneal coef 1.0→0 over the run**, and the
length-2 phase (rounds 10–30) is exactly where it fades. We built the path reward and switch it off
when step-2 needs it. Honesty is guaranteed by the SWAPPED/held-out eval (always coef 0), not by
withholding the path reward — so we can keep it dense and stay honest.

## Motivating re-analysis (exp0049 floor 0.2, via the new swap_follow_s1)

| shaping floor | step-1 | exact | P(step-2 \| step-1) |
|---|---|---|---|
| 0 (anneal to 0, baseline) | 0.28 | 0.05 | 0.18 |
| 0.2 (path kept, exp0049, 4 seeds) | **0.42** | 0.07 | 0.17 |

Keeping the path reward at 0.2 **lifted step-1 reading (0.28→0.42)** — "reward the path" works for getting
ON the path — but the **conditional step-2 (~0.17) did not move.** Read: a *uniform* per-step reward
raises how often the agent is on the path (step-1, lots of exposure) but not step-2's conditional
reliability — plausibly because step-2 is only ever practiced AFTER step-1 (a rare state), so it stays
exposure-starved even when rewarded.

## Hypothesis / sweep

Push the path reward harder — `--reading-shaping-floor ∈ {0.5, 1.0}` (1.0 = no anneal, full path
reward throughout), baseline else (folded VR 0.3, n-train 400, length 2). Watch swap_follow_s1, exact,
and the **conditional P(step-2|step-1)**.

- If conditional step-2 rises with a stronger/kept path reward → reward-magnitude was the limiter.
- If step-1 keeps rising but conditional step-2 stays ~0.17 → step-2 is **exposure-starved**, not
  reward-starved → next lever targets step-2 *practice* directly (weight later gesture steps higher,
  or imagine/reset from step-1-done states — a backward curriculum), rather than more uniform reward.

## Standing bar — diagnostic (tag: `EXTRA-RIGOR`)

Succeeds by moving (or definitively not moving) the conditional step-2 with a kept/stronger path
reward — picking between "reward magnitude" and "step-2 exposure" as the next lever.

## Links

[[0055-rtfm-ensemble-pessimism]] · [[0056-rtfm-coverage-sweep]] · [[0049-rtfm-sustained-vr]] · [[0043-rtfm-execution-wall]] · [[mentored-learning-loop]]
