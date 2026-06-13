---
status: draft
owner: human
scope: local
sources: []
verified: true
last_reviewed: 2026-06-13
---

# 0015: Staged freeze — encoder@2, GRU@4/5 (planned) [autonomous]

## Hypothesis

Two drift sources, two pensions: freezing the encoder at round 2 (proven
crash-stopper, exp 0013) and the GRU later (after it exploits mid-quality data;
exp 0014 showed GRU drift causes the residual crashes) yields a config that
clears BOTH bars: mean ≥55% AND no >20pp post-(first-)freeze crash. Bottom-up
progressive freezing principle imported from FreezeOut/LayerLock (supervised/SSL);
uncovered in flywheel RL. Counter-outcome: GRU freeze caps means again (à la
ftrunk) → the frontier is fundamental at this budget → propose budget-tier
extension + PPO re-baseline to Dave.

Arms (3 seeds, UTD ×4, standard protocol): (a) enc@2+gru@4; (b) enc@2+gru@5.
Comparators: fenc@2 47% stable / fenc@4 60% crashy / utd 63% violent.

## Setup

Needs --freeze2 mechanics (second freeze event for GRU): --freeze encoder
--freeze-round 2 --freeze2 gru --freeze2-round {4|5}. 6 runs, one batch.

## Result

_pending_

## Lesson

_pending_

## Links

[[0014-freeze-round-sweep]] · [[0013-trunk-freeze]] · [[retention]]
