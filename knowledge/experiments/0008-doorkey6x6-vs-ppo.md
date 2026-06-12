---
status: draft
owner: human
scope: local
sources: []
verified: true
last_reviewed: 2026-06-12
---

# 0008: DoorKey-6x6 at equal budget — fast flywheel vs PPO (planned)

## Hypothesis

On DoorKey-6x6 (random baseline measured: 18/300 = 6.0%, avg episode 353 steps) at a
shared budget of 110k env steps, the world-model agent with a fast-spinning flywheel
(20k random ignition + 6 × 15k ε=0.3-MPC rounds — exp 0007 lesson: stop burning
budget on random collection) reaches a final greedy success rate that **exceeds
PPO's by ≥20 percentage points**, because PPO's exploration cost grows steeply with
room size while model-guided collection does not.

Pre-registered alternatives: (a) PPO also solves 6x6 within 110k → escalate to
DoorKey-8x8 (rung 2b moves again); (b) the flywheel fails to ignite from ~3 reward
events in 20k random steps → ignition mechanics (adaptive round 0, intrinsic
signal) become the next experiment instead.

## Setup

World model (remote 5070 Ti): `scripts/remote.sh run python -m
world_model.train_recurrent --env-id MiniGrid-DoorKey-6x6-v0 --rounds 7
--round0-steps 20000 --mpc-steps 15000 --updates-per-round 1500
--save runs/doorkey6x6_v.pt --seed 0` (single seed — runtime; noted as limitation).
PPO (local, 3 seeds): `uv run python scripts/sweep.py --grid seed=0,1,2
--name exp0008-ppo -- --project baselines/ppo python baselines/ppo/train_ppo.py
--env-id MiniGrid-DoorKey-6x6-v0 --total-steps 110000`.
Both: eval protocol as exp 0006/0007 (periodic 10 eps seeds 10000+; final 20 eps
seeds 0+; world-model side post-hoc local eval with matched obs guarded by
obs_shape).

## Result

_pending_

## Lesson

_pending_

## Links

[[0007-ppo-baseline]] · [[0006-value-head-doorkey]] · [[environment-ladder]]
