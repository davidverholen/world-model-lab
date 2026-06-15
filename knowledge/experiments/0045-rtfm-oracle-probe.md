---
status: draft
owner: world-model
scope: local
verified: true
last_reviewed: 2026-06-15
---

# 0045: oracle probe — the WM reads+values the gesture; the wall is reward-head OOD calibration

## Hypothesis

exp0044 found MPC ≈ reactive at length-2 (neither cracks it), relocating the wall to the WM's
multi-step rollout/reward fidelity — but couldn't separate "the WM can't imagine the gesture's payoff"
(fidelity) from "MPC's search/objective is the limit". The oracle-gesture probe (`--oracle-probe`,
commit 692c233) forks it: per eval episode, score the **true** 2-action gesture (privileged
`manual_facts`, eval-only) vs `n_random=200` random sequences with the SAME planner objective
(`_rollout_returns`), and report the oracle's **percentile**. pct≈1 ⇒ the reward head ranks the
gesture top ⇒ the gap is MPC search; pct≈0.5 ⇒ uninformative ⇒ WM-fidelity wall. Same curriculum+aux
WM as exp0044, no MPC.

## Result — oracle pct ≈ 0.88 (NOT 0.5), oracle return ~5–6× random

Final rounds (length-2):

| seed | oracle pct | oracle_ret | rand_ret | oracle/random |
|---|---|---|---|---|
| s0 | 0.89 | 0.23 | 0.04 | ~5× |
| s1 | 0.82 | 0.12 | 0.03 | ~4× |
| s2 | 0.92 | 0.25 | 0.03 | ~8× |
| s3 | 0.93 | 0.26 | 0.04 | ~6× |

Stable/rising across the length-2 phase (s0: 0.83 → 0.90). **The WM genuinely reads AND values the
true gesture** — it ranks it well above random (≈88th percentile, 5–6× the random return). This is
**not** a fundamental WM-fidelity wall (that would be pct≈0.5).

## Lesson — the wall is reward-head OOD calibration; exp 0017–0025 at the planning layer

The decisive detail is that pct ≈ 0.88, **not 1.0**: the gesture is ranked *high but not highest* —
~12% of random sequences are scored *above* the only sequence that actually earns reward. Those are
**reward-head false positives: overestimates on out-of-distribution action sequences.** Naive MPC
maximizes predicted reward, so it is pulled to exactly those overrated wrong sequences rather than the
gesture — which is precisely why exp 0044's MPC matched the reactive actor (both chase reward-head
overestimates).

**This is the exp 0017–0025 imagination-exploitation problem resurfacing at the planning layer** — the
planner exploits reward-head overestimation in OOD regions, just as the imagination actor did before.
Same dragon, known toolkit. The architecture's read→imagine→**value** chain is intact; the gap is
planner robustness to reward-head OOD false positives. This also empirically motivates the
calibrated-uncertainty / confidence-gating the design flagged as load-bearing
([[hierarchical-imagination-agent]] §5).

→ next (a calibration / robust-planning lever, not a new WM):
1. **Reward-head calibration** — cut OOD overestimation so the gesture becomes (near) the argmax
   (exp 0017–0025 lineage: distributional head — have it; add in-distribution regularization / penalize
   actions the WM is uncertain about).
2. **Robust planning** — value/continue-aware scoring, pessimism (penalize predicted-reward by WM
   uncertainty), or sampled-rollout averaging instead of prior-mean argmax. Re-run the MPC A/B and
   the oracle pct should rise toward 1 while MPC `correct` lifts off the floor.

## Reproduction

- Commit 692c233@world-model (`--oracle-probe`).
- `dispatch_rtfm.sh exp0045 4 --rounds 30 --curriculum-rounds 10 ... --length 2 --one-shot
  --reading-shaping-coef 1.0 --manual-aux-coef 1.0 --oracle-probe` (no MPC).
- Artifacts: `runs/exp0045/` (`ORACLE pct=.. oracle_ret=.. rand_ret=..` per length-2 round).

## Links

[[0044-rtfm-mpc-execution]] · [[hierarchical-imagination-agent]] · [[imagination-training]] · [[0021-critic-on-replay]] · [[0007-crafter-mastery-milestone]]
