---
status: draft
owner: human
scope: local
sources: []
verified: true
last_reviewed: 2026-06-12
---

# 0009: Ignition mechanics — oversampling + adaptive round 0 on DoorKey-6x6 (planned)

## Hypothesis

Exp 0008's failure decomposed into ignition (1 reward event → no value signal) and
training instability (success signal drowned in replay: 13 successes / 110k
transitions). Two mechanics fix both: (a) **adaptive round 0** — keep collecting
(up to 2×20k) until ≥5 reward events exist; (b) **success-episode oversampling** —
25% of every training batch is windows ending in a reward transition. With these,
≥2 of 3 seeds reach a final greedy success rate **above PPO's matched-budget mean
(~37% at 110k; compare at actual consumed budget per seed)** on DoorKey-6x6, and
no seed shows the round-6-style regression (high collection success → 0% eval)
while the best-checkpoint guard is in place.

Pre-registered alternative: if ignition succeeds but evals stay low, the
instability is not a sampling problem → lr schedule / EMA weights next.

## Setup

`uv run python scripts/sweep.py --remote --grid seed=0,1,2 --name exp0009 --
python -m world_model.train_recurrent --env-id MiniGrid-DoorKey-6x6-v0 --rounds 7
--round0-steps 20000 --mpc-steps 15000 --updates-per-round 1500
--success-frac 0.25 --ignition-events 5 --save runs/doorkey6x6_ign.pt`
(3 seeds, sequential on the 5070 Ti; checkpoints seed-suffixed; total_env_steps
reported per seed — budget varies 110k–130k with ignition, compared honestly
against PPO's curve at the matching checkpoint.)
PPO comparator: exp 0008 sweeps (`runs/sweeps/exp0008-ppo/`), curves up to 110k.

## Result

_pending_

## Lesson

_pending_

## Links

[[0008-doorkey6x6-vs-ppo]] · [[0006-value-head-doorkey]] · [[environment-ladder]]
