---
status: draft
owner: human
scope: local
sources: []
verified: true
last_reviewed: 2026-06-12
---

# 0010: Retention mechanics — 2×2 factorial on DoorKey-6x6 (planned)

## Hypothesis

Exp 0009 isolated catastrophic interference: continued Adam training over shifting
round data destroys acquired competence (60% → 0%). Two mechanics attack it:
(a) **lr decay** ×0.3 after round 0 (gentler updates once competence exists);
(b) **EMA acting weights** (decay 0.995) — collection/eval/checkpoints use a slow
shadow so gradient noise never drives behavior directly. In a 2×2 factorial
(ctrl / lr03 / ema / lr03+ema, 3 seeds each, same budget+protocol as exp 0009):
at least one treatment arm shows BOTH (i) no 0009-style crash (no round where eval
drops >30pp below the running best) and (ii) mean best-eval ≥ 55% — beating PPO's
~37% matched-budget mean without relying on the checkpoint guard.

Pre-registered fallback: if EMA helps but plateaus low, the lag is hiding genuine
improvements → tune decay; if nothing helps, interference is in the encoder →
frozen-trunk arm next.

## Setup

12 runs in 2 batches of 6 parallel processes on the 5070 Ti (`remote.sh shell`,
OMP_NUM_THREADS=2). Per run: as exp 0009 (`--rounds 7 --round0-steps 20000
--mpc-steps 15000 --updates-per-round 1500 --success-frac 0.25
--ignition-events 5`) plus the arm's `--later-lr-scale {1.0|0.3}` /
`--ema-decay {0|0.995}`. Logs `exp0010_<arm>_s<seed>.log`; checkpoints
`runs/dk6_<arm>_s<seed>.pt`. Comparators: exp 0009 (ctrl arm replicates it
modulo new acting-path code), PPO exp 0008.

## Result

_pending_

## Lesson

_pending_

## Links

[[0009-ignition-mechanics]] · [[0008-doorkey6x6-vs-ppo]] · [[environment-ladder]]
