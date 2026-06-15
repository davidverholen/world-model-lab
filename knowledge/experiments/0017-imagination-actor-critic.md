---
status: draft
owner: world-model
scope: local
sources: [arxiv:2301.04104]
verified: true
last_reviewed: 2026-06-13
---

# 0017: Imagination actor-critic — model exploitation (missing continue predictor) (run 2026-06-13)

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

3 seeds, desktop (fast — actor-collection is ~free vs the old planner). **Model
exploitation confirmed, all seeds:**

| seed | imagined_return | eval_success |
|---|---|---|
| 0 | 2.057 | 0.00 |
| 1 | 3.013 | 0.00 |
| 2 | 0.483 | 0.00 |

imagined_return of 2–3 is IMPOSSIBLE in a real DoorKey episode (max ~1.0, paid once
on termination). The actor drives the world model into hallucinated high-reward
latent regions and farms reward; eval 0% everywhere.

## Lesson

**We omitted Dreamer's continue predictor.** Real DoorKey TERMINATES on goal; our
imagination rolls H=15 steps with NO termination, so once the actor imagines a
goal-like state it keeps collecting reward-head output for the remaining steps —
imagined_return = the ~1.0 goal reward counted multiple times. Literature gate
(DreamerV2/V3): a continue/discount predictor c(s)∈[0,1], trained on real done
flags, weights imagined steps by the cumulative product of predicted continue
probs — zeroing post-termination reward. This is the single clearest omission; the
WM machinery, on-policy AC, and fast actor-collection all work.

Exp 0018: add the continue predictor (one head, trained on dones, used to discount
imagined λ-returns). Clean single-variable fix. (s2's <1 return suggests a possible
secondary reward-head off-distribution issue — diagnose only if 0018 doesn't
resolve it; one fix at a time.)

## Links

## Links

[[0016-actor-distillation]] · [[dreamerv3-2023]] · [[imagination-training]] ·
[[hierarchy-and-credit]] · [[capability-map]]
