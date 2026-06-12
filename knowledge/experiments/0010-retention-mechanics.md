---
status: draft
owner: human
scope: local
sources: []
verified: true
last_reviewed: 2026-06-12
---

# 0010: Retention 2×2 — lr decay and EMA both fail; resets are next (run 2026-06-12)

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

**Hypothesis refuted across the whole factorial** — no arm achieved (i) no-crash or
(ii) mean ≥55%. Best-evals per seed (s0/s1/s2):

| arm | bests | mean | seed-1 trajectory after its round-0 peak |
|---|---|---|---|
| ctrl | 20/60/40 | 40% | 60→10→0→10→20→20→50 (replicates 0009 exactly) |
| lr03 | 20/60/40 | 40% | 60→20→10→20→30→10→10 — crash anyway, recovery LOST |
| ema 0.995 | 20/40/30 | 30% | 40→20→0→0→10→0→10 — crash anyway, peak DAMPED |
| lr03+ema | 20/40/50 | 37% | 40→10→20→0→0→10→10 |

Observations: (a) the post-round-0 crash occurs in **all four arms** for seed 1 —
deterministic interference from data ordering, not optimization noise; (b) EMA's lag
suppresses fresh competence (round-0 peaks 60→40) without protecting it afterwards;
(c) lr decay trades late-round recovery for nothing. Ops note: 6-wide parallel
caused heavy host-RAM paging (replay buffers ~10 GB/process; three processes paged
out for ~an hour) — uint8 obs storage queued as the fix; logs are block-buffered, so
empty-mid-run is normal.

## Lesson

1. **Weight-trajectory smoothing (lr/EMA) cannot fix interference here** — the damage
   is in the *direction* of the updates (new-round gradients overwrite round-0
   competence), not their size or jitter. A clean 12-run negative.
2. This is textbook **primacy bias** (QUEUE: arxiv:2205.07802): network overfits
   early data, then can't exploit better data. The literature's working fix is
   periodic **resets of late layers while keeping the replay buffer** — our setup is
   unusually reset-friendly (full replay retained, success oversampling already
   ensures the good windows are re-seen at high rate after a reset).
3. Exp 0011 design: at each round start, reinit heads (+ optionally the GRU,
   factorial again) and retrain from the accumulated replay; compare against ctrl.
   Secondary: ingest the Nikishin paper properly first (/ingest-source).
4. uint8 replay storage (4× RAM cut) before any further 6-wide batches.

## Links

## Links

[[0009-ignition-mechanics]] · [[0008-doorkey6x6-vs-ppo]] · [[environment-ladder]]
