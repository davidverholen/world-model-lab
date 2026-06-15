---
status: draft
owner: world-model
scope: local
sources: []
verified: true
last_reviewed: 2026-06-12
---

# 0009: Ignition fixed, retention broken — DoorKey-6x6, 3 seeds (run 2026-06-12)

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
(3 seeds, sequential on the desktop GPU; checkpoints seed-suffixed; total_env_steps
reported per seed — budget varies 110k–130k with ignition, compared honestly
against PPO's curve at the matching checkpoint.)
PPO comparator: exp 0008 sweeps (`runs/sweeps/exp0008-ppo/`), curves up to 110k.

## Result

3 seeds in parallel on one GPU (~82% util, 49 °C — first use of `remote.sh shell`;
**wall 35 min** by log timestamps, 12:04→12:39, seeds within 80 s of each other —
parallel contention ≈ nil). Checkpoints: `runs/remote/doorkey6x6_ign_s{0,1,2}.pt`.

| seed | ignition (events @ steps) | best eval (round) | final round | total steps |
|---|---|---|---|---|
| 0 | 3 @ 40k (cap) | 20% (r4) | 10% | 130k |
| 1 | 6 @ 25k | **60% (r0)** | 50% | 115k |
| 2 | 3 @ 40k (cap) | 40% (r6) | 40% | 130k |

PPO comparator (exp 0008, 110k): 40/60/10, mean ~37%. WM best-checkpoints
20/60/40, mean 40% at 115–130k — **parity at best, not the hypothesized ≥20pp win**,
and bought with extra ignition budget.

**The decisive observation is seed 1**: 6 success examples + oversampling → 60%
greedy directly after round 0 (the approach works, fast!) — then five rounds of
further training *destroyed and only partially rebuilt* it (60→10→0→10→20→20→50%).
Hypothesis's no-regression criterion failed; the pre-registered alternative fires:
**instability is not a sampling problem.**

## Lesson

1. **Ignition: solved.** Adaptive round 0 + 25% success-window oversampling → every
   seed evaluates >0% after round 0 (exp 0008: 0%); 6 examples sufficed for 60%.
   Both mechanics stay on by default.
2. **Retention is the bottleneck now, cleanly isolated.** Continuing Adam training
   over a distribution that shifts every round catastrophically interferes with
   exactly the competence just acquired. Exp 0010 candidates (pre-registered):
   EMA/snapshot weights for the acting agent, lower lr after round 0, value-head
   target networks, or freezing encoder/dynamics once good and training only heads.
3. Seed variance is large (ignition luck: 25k vs 40k just to start); 3 seeds is the
   floor for any claim on this rung.
4. Parallel-seeds-on-one-GPU works exactly as the [[compute-strategy]] corollary
   predicted (3× wall-time, 49 °C); `remote.sh shell/kill` are the new primitives.
5. Rung 2b verdict: **still open** — we match PPO's mean via best-checkpoint
   selection, but matching with a guard is not beating. Fix retention first;
   the 60%-after-25k-steps data point says the win is there once the agent can
   keep what it learns (PPO at 25k: 0%).

## Links

## Links

[[0008-doorkey6x6-vs-ppo]] · [[0006-value-head-doorkey]] · [[environment-ladder]]
