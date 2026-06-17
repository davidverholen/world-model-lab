---
status: current
owner: world-model
scope: local
verified: true
last_reviewed: 2026-06-17
---

# 0059: multi-seed confirm of the floor-0.5 path reward

**Status: DONE (4 seeds) — floor-0.5 robustly lifts step-1 + conditional; exact ~0.10 at round 30 but
STILL RISING (not a plateau — the single seed was high, but the ceiling claim was premature).**

Multi-seed confirmation of the [[0057-rtfm-path-reward]] floor-0.5 single-seed result. The rung-4
wall is 2-step composition: `swap_follow` = exact full-chain match; `swap_follow_s1` = step-1 match;
conditional = P(step-2|step-1) = exact/step1. Baseline: step1 0.28 / exact 0.05 / conditional 0.17.

## Setup

`--rounds 30 --curriculum-rounds 10 --length 2 --manual-aux-coef 1.0 --validated-reading-coef 0.3
--n-train-seeds 400 --reading-shaping-floor 0.5`, 4 seeds (0–3).

## Result

Last-5 length-2 swap_follow metrics:

| seed | step-1 | exact | conditional |
|---|---|---|---|
| 0 | 0.364 | 0.078 | 0.21 |
| 1 | 0.354 | 0.144 | 0.41 |
| 2 | 0.404 | 0.120 | 0.30 |
| 3 | 0.324 | 0.070 | 0.22 |
| **MEAN** | **0.36** | **0.103** | **0.285** |

![exp0059 grounding headline (4-seed swap_follow + swap_follow_s1)](../../assets/exp-0059/grounding-headline-0-1.png)

## Key findings

Floor 0.5 lifts step-1 (0.28→0.36) and conditional step-2 (0.17→0.285) on ALL 4 seeds — real and
consistent. The last-5 **exact swap_follow is ~0.10** (0.36·0.285 ≈ 0.10).

**Trend caveat (do NOT read this as a ceiling):** swap_follow is **still gently rising at round 30**,
not plateaued — the 4-seed per-round mean drifts from ~0.065 (rounds 20–23) to ~0.105 (rounds 26–29),
with r29 ≈ 0.13. So "~0.10" is a **30-round snapshot mid-climb, a lower bound** — whether more rounds
break past 0.10 is **untested**. (Contrast [[0051-rtfm-flat-vr-optimization]]'s budget probe, which had
NO path reward and was genuinely flat ~0.11 over 40 rounds; the path reward here appears to drive a
continued climb — so a **longer floor-0.5 run is a live lever**, not ruled out by exp0051.) The earlier
flat framing ("recovers but doesn't break the ceiling") was premature — corrected here.

The single-seed [[0057-rtfm-path-reward]] value (exact 0.14 / cond 0.38) was an **optimistic draw**
(honest correction, cf. [[0048-rtfm-validated-reading]] phase-1 inflation). The 4-seed mean is the
authoritative result.

Breaking exact past ~0.10 requires pushing the conditional higher. The coverage lever
([[0060-rtfm-coverage-floor-combo]]) was tested next; back-loading on top of the 0.5 base was tested
in [[0061-rtfm-backloading-confirm]].

## Lesson

Floor-0.5 is a robust, confirmed reward lever: it reliably lifts both step-1 and conditional step-2
across seeds. Two corrections to earlier framing: (1) the single-seed exp0057 exact (0.14) was a high
draw — the 4-seed mean is ~0.10; (2) but ~0.10 is NOT a demonstrated ceiling — swap_follow is still
rising at round 30, so a longer run is the obvious untested lever. The next frontier for exact is the
conditional — which exp0061 shows is not reward-magnitude-limited but exposure/data-bound, AND which
the still-rising trend suggests is also training-length-limited at the floor-0.5 reward.

## Links

[[0057-rtfm-path-reward]] · [[0058-rtfm-escalating-path-reward]] · [[0056-rtfm-coverage-sweep]] · [[0048-rtfm-validated-reading]]
