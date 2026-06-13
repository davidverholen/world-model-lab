---
status: draft
owner: human
scope: local
sources: [arxiv:2301.04104]
verified: true
last_reviewed: 2026-06-13
---

# 0017: Imagination actor-critic — the on-policy fast actor (planned)

## Hypothesis

The on-policy fix exp 0016 mandated. A Dreamer-style actor-critic trained purely on
the world model's imagined rollouts (REINFORCE actor + value baseline + entropy,
lambda-returns, EMA target critic; H=15, lam=0.95, ent 3e-3) — with collection
switched from the slow CEM planner to the fast actor (eps-greedy) — learns a usable
reactive policy on DoorKey-6x6: **eval (20 eps, fixed seeds) reaches >= the planner's
properly-measured success on the same env (~25-35%)**, at ~1000x lower inference
cost. This delivers the Mode-1 policy (Crafter-scale-collection prerequisite).

Watch (the known risk): **model exploitation** — if `imagined_return` climbs while
`eval_success` stays flat, the actor is exploiting world-model/reward-head errors
rather than learning real competence. Joint WM training (collect-with-actor each
round) is the guard; if the gap persists, the diagnosis is reward-head/world-model
inaccuracy off the planner's distribution.

Don't re-ablate what Dreamer settled — implement the recipe, deviate only for our
discrete-action / GRU-belief setup (REINFORCE, no dynamics backprop).

## Setup

`train_dreamer.py` (reuses settled WM recipe: wm_train + collect_round + encoder
freeze @round 2). 3 seeds, DoorKey-6x6, 7 rounds (20k random ignition + 6x15k
actor-collected), 4000 WM + 4000 AC updates/round. Dispatch to desktop (laptop
heat; ~longer GPU-bound run). Eval 20 eps seeds 10000+ (per the 0016 eval-rigor
fix). Baseline: planner on same checkpoint/seeds; BC actor (exp 0016: ~0%).

## Result

_pending_

## Lesson

_pending_

## Links

[[0016-actor-distillation]] · [[dreamerv3-2023]] · [[imagination-training]] ·
[[hierarchy-and-credit]] · [[capability-map]]
