---
status: draft
owner: world-model
scope: local
sources: []
verified: true
last_reviewed: 2026-06-12
---

# 0008: DoorKey-6x6 at equal budget — flywheel fails to ignite, PPO wins (run 2026-06-12)

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

World model (remote desktop GPU): `scripts/remote.sh run python -m
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

**Hypothesis refuted via pre-registered alternative (b): ignition failure.**

PPO (3 seeds, 110k): flat 0% through 80k, late breakthrough 100k–110k; final 20-ep
evals **40% / 60% / 10%** (mean ~37%). Curves in `runs/sweeps/exp0008-ppo/`.

World model (single seed, desktop, commit 548d042): round-0 random collection drew
**1 reward event in 20k steps** (expectation ~3.3 at the measured 6.0% baseline —
unlucky draw on top of a thin signal). The flywheel never spun up:

| round | collection events | eval |
|---|---|---|
| 0 random | 1/55 eps | 0% |
| 1–2 | 0, 1 | 0% |
| 3 | 8 | 0% |
| 4–5 | 0, 2 | 10%, 10% |
| 6 | **13/47 = 28%** | **0%** |

Best checkpoint: round 5, **10%**. PPO wins at every checkpoint.

Round 6 is the second sighting of the exp-0005 instability: collection success 28%
(the model clearly improved mid-round) but training on that round's data *degraded*
greedy eval to 0%. The best-checkpoint guard contained the damage but the
underlying training dynamics are unsolved.

## Lesson

1. **Ignition is the binding constraint for sparse envs**: the value head needs
   some success examples; one is not enough, and hoping for lucky random draws
   doesn't scale up the ladder. Candidates for exp 0009 (in increasing ambition):
   adaptive round 0 (collect until ≥K reward events, count honestly in budget);
   success-episode oversampling in replay (13 successes drowned in 110k
   transitions plausibly explains round 6); intrinsic exploration signal.
2. **Round-over-round training instability is now a confirmed pattern** (exp 0005,
   here round 6): retraining on shifted data can wash out what the agent just
   learned. Mitigations to test: success-balanced sampling, lower lr after round 0,
   EMA weights for the eval/collection agent.
3. PPO's profile is consistent: long flat exploration phase, then fast convergence
   once *any* success is found. Its breakthrough moved 60–80k (5x5) → 100k+ (6x6);
   at DoorKey-8x8 it likely needs ≫110k — but we can't exploit that until ignition
   is solved on our side. **Rung 2b stays open**; scoreboard PPO 2 : WM 1.
4. Single-seed world-model runs are now clearly insufficient given how much
   round-0 luck matters (1 vs expected 3.3 events) — multi-seed via /sweep
   --remote once runtimes allow, or at least 2–3 seeds on ignition experiments.

## Links

## Links

[[0007-ppo-baseline]] · [[0006-value-head-doorkey]] · [[environment-ladder]]
