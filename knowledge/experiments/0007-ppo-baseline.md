---
status: draft
owner: human
scope: local
sources: []
verified: true
last_reviewed: 2026-06-12
---

# 0007: PPO sample-efficiency baseline on DoorKey-5x5 — PPO wins (run 2026-06-12)

## Hypothesis

At the exact env-step budget our world-model agent consumed in exp 0006 (100k steps,
all collection rounds counted), model-free PPO from the same egocentric RGB pixels
ends below 20% greedy success on DoorKey-5x5 across 3 seeds — i.e. the world-model
agent (~80%) beats PPO by ≥4× at equal budget, closing rung 2's exit criterion.
Rationale: sparse reward + partial observability + pixel obs is PPO's worst case;
no intrinsic motivation, no recurrence in the policy.

Falsifiable counter-outcome we take seriously: PPO solves DoorKey-5x5 cheaply
(it is a small env) — in that case rung 2 is NOT closed and we must scale the
comparison (DoorKey-6x6/8x8) before claiming a sample-efficiency win.

## Setup

`uv run python scripts/sweep.py --grid seed=0,1,2 --name exp0007 --
--project baselines/ppo python baselines/ppo/train_ppo.py --total-steps 100000`
sb3 PPO, CnnPolicy (NatureCNN), 8 DummyVec envs, defaults otherwise; eval every 20k
(10 eps, seeds 10000+) + final 20 eps (seeds 0+) — protocol mirrors exp 0006.
Isolated dep universe (sb3 caps gymnasium <1.3): baselines/ppo/pyproject.toml;
versions via `uv tree --project baselines/ppo` (gymnasium 1.2.x, minigrid, sb3 ≥2.7).
World-model comparator: exp 0006 (100k steps → 90%/75% greedy).

## Result

**Hypothesis refuted — the pre-registered counter-outcome occurred.** PPO reaches
**100% on all 3 seeds** (final 20-episode eval, seeds 0+), breaking through between
60k–80k steps. Eval curves (10 eps @ seeds 10000+):

| steps | PPO s0 | PPO s1 | PPO s2 | world model (exp 0006) |
|---|---|---|---|---|
| 20k | 0% | 0% | 0% | — |
| 40k | 0% | 10% | 0% | — |
| 60k | 30% | 70% | 40% | 0% (after round 0) |
| 80k | 100% | 100% | 100% | 50% (after round 1) |
| 100k | 100% | 100% | 100% | 90% (round 2) / 75–80% verified |

PPO is ahead at every comparable budget. Sweep artifacts: `runs/sweeps/exp0007/`.

## Lesson

1. **Rung 2 is NOT closed.** DoorKey-5x5 at 100k steps is inside vanilla PPO's
   comfort zone (8 parallel envs + NatureCNN, no tricks) — the env is too easy to
   discriminate model-based from model-free. Claiming a sample-efficiency win here
   would have been wrong; the baseline run was worth it.
2. **Where our agent actually loses the budget**: 60k steps of *random* collection
   before the flywheel starts spinning. PPO explores and learns simultaneously from
   step 0. Exp 0008 directions: smaller round 0 + more, shorter rounds (spin the
   flywheel early), and/or harder env (DoorKey-6x6/8x8) where PPO's
   exploration cost explodes but ε-MPC collection may not.
3. **Deployment asymmetry worth recording**: PPO amortizes into a microsecond
   policy; our agent runs 1024×3 imagined rollouts per action. Different artifacts —
   policy distillation from the planner (Dreamer-style actor) would close that gap
   and is on the path anyway.
4. Comparisons need full curves, not endpoints — both protocols already emit
   `eval_success=` per checkpoint, and /sweep aggregated them cleanly (first
   dogfood: worked, table above pasted from RESULTS.md + logs).
5. Proposal for [[environment-ladder]] (owner: human, needs Dave's sign-off):
   rung-2 exit splits into "solve DoorKey under partial obs" (✅ exp 0006) and
   "beat model-free on sample efficiency" — the latter moves to the smallest env
   where PPO struggles at budget (find it in exp 0008).

## Links

[[0006-value-head-doorkey]] · [[environment-ladder]] · [[imagination-training]]
