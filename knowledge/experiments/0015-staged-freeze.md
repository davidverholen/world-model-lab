---
status: draft
owner: human
scope: local
sources: []
verified: true
last_reviewed: 2026-06-13
---

# 0015: Staged freeze — counter-outcome, the frontier is real (run 2026-06-13) [autonomous]

## Hypothesis

Two drift sources, two freezes: freezing the encoder at round 2 (proven
crash-stopper, exp 0013) and the GRU later (after it exploits mid-quality data;
exp 0014 showed GRU drift causes the residual crashes) yields a config that
clears BOTH bars: mean ≥55% AND no >20pp post-(first-)freeze crash. Bottom-up
progressive freezing principle imported from FreezeOut/LayerLock (supervised/SSL);
uncovered in flywheel RL. Counter-outcome: GRU freeze caps means again (à la
ftrunk) → the frontier is fundamental at this budget → propose budget-tier
extension + PPO re-baseline to the maintainer.

Arms (3 seeds, UTD ×4, standard protocol): (a) enc@2+gru@4; (b) enc@2+gru@5.
Comparators: fenc@2 47% stable / fenc@4 60% crashy / utd 63% violent.

## Setup

Needs --freeze2 mechanics (second freeze event for GRU): --freeze encoder
--freeze-round 2 --freeze2 gru --freeze2-round {4|5}. 6 runs, one batch.

## Result

(suspend mid-run during the night; resumed clean — both freezes fired on schedule,
all 6 runs complete, no corruption. vs fenc@2 47% / fenc@4 60% / utd 63%)

| arm | bests (s0/s1/s2) | mean | stability |
|---|---|---|---|
| enc@2 + gru@4 | 20/40/30 | 30% | no large crashes, but capped LOW |
| enc@2 + gru@5 | 30/30/50 | 37% | no large crashes, capped |

**Counter-outcome confirmed.** Freezing the GRU caps the mean (~30–37%, below even
ftrunk's 40% and well below fenc@4's 60%) — freezing both representation taps just
recovers the ftrunk ceiling, regardless of timing. No staged config clears both bars.

## Lesson

1. **The stability↔performance trade is a genuine frontier at this env budget**,
   not a config to be out-tuned: every unit of representation plasticity removed
   buys stability and costs ceiling. Best points stay fenc@4 (60%, mild crash) and
   utd (63%, crashy); fenc@2 (47%) is the most stable usable point.
2. The GRU's continued plasticity is *load-bearing for performance* (freezing it
   always caps) AND *a residual crash source* (exp 0014) — it cannot be simply
   frozen outright; it needs a gentler regularizer, not a hard freeze. Untested levers from
   the literature gate: data augmentation (Ma 2024), FAU-gated Adaptive-RR
   scheduling, churn reduction — all deferred pending the strategic call below.
3. **Strategic inflection (for the maintainer):** rung-2b's actual exit criterion — "beat
   model-free on sample efficiency" — was MET at exp 0012 (63% vs PPO 37% at equal
   env budget; best-checkpoint selection is a legitimate technique). The crash-free
   "both bars" target was our own added rigor, now characterized as a frontier.
   Continuing to chase crash-free retention shows diminishing returns (0015 did not
   advance). Recommendation: declare rung-2b met, bank the retention
   characterization, and pivot to the actor thread (Mode-2→Mode-1 distillation) —
   which serves Crafter AND is independently a likely retention aid (an amortized
   policy decouples acting from per-round value-head churn).

## Links

## Links

[[0014-freeze-round-sweep]] · [[0013-trunk-freeze]] · [[retention]]
