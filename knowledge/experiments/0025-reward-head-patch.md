---
status: draft
owner: human
scope: local
sources: [arxiv:2301.04104]
verified: true
last_reviewed: 2026-06-13
---

# 0025: Bounded reward head + patch tokens — stabilize the navigation win (pre-registered)

## Hypothesis

Two findings converge: (0023 diagnostic) the agent exploits an over-predicting MSE reward
head (spams a useless action, never moves); (0024) DINO **patch tokens** unlock navigation
and a real table — but only on the lucky seed, because the reward exploitation is unfixed
(imagined_return 8–22) and tips the other seed into the stationary collapse. So combine the
two levers: **patch tokens** (perception → navigation) + the **two-hot distributional reward
head** (bound per-step reward → remove the spurious value of useless actions → stop the
exploitation → stabilize). Predict: (i) imagined_return falls toward the real scale (~1–3),
(ii) `behavior_report` shows movement on **all** seeds (not 1/2), (iii) deeper achievements
(place_table, collect_wood, maybe collect_stone) more consistently.

## Setup

`train_crafter --pool cls+patch` with the **TwoHotRewardHead** (now the train_crafter default,
models/twohot.py) + two-hot critic + critic-on-replay. 3 seeds (the 0024 variance needs >2),
10 rounds, else identical to 0023/0024. behavior_report on every checkpoint (movement is the
primary readout now, not just eval_reward). Null/partial result → isolate (reward-fix with CLS
vs patch) or escalate to exploration bonus / entropy.

Command per seed: `python -m world_model.train_crafter --pool cls+patch --rounds 10
--round0-steps 10000 --actor-steps 10000 --updates-per-round 2000 --ac-updates-per-round 2000
--seq-batch 16 --window 24 --burn-in 8 --horizon 15 --eval-episodes 8 --eval-length 1000
--ep-length 2000 --repval 0.3 --save runs/crafter_rew.pt --seed {0,1,2}`

## Result

**Success — the combined fix resolves the exploitation/stationary/seed-variance failure
mode.** 3 seeds, 10 rounds, `--pool cls+patch` + two-hot reward head + two-hot critic +
critic-on-replay. best_eval_reward s0 2.35, s1 2.47, **s2 3.47**. Achievement trajectories
all climbed (s1 sustained 3.38 late; s2 3.12; s0 2.25), and — the decisive readout —
**`behavior_report` PASSES on ALL 3 seeds** (vs 0023 all-stationary, 0024 seed-dependent):

| seed | moved_frac | bbox span | action entropy | achievements (unique) | notable |
|---|---|---|---|---|---|
| s0 | 0.11 | 18.7 | 1.88/2.83 | 3 | make_stone_pickaxe; residual make_iron_sword 0.33 |
| s1 | 0.18 | 21.7 | 1.52 | 4 | move_down 0.39 (movement-led); place_furnace |
| s2 | 0.17 | 18.3 | 2.04 | **6** | make_wood_pickaxe; reward 3.77 — deepest agent yet |

All three **move** (vs 0023's moved_frac 0.00, do:0.5–0.96), action entropy is high
(1.5–2.0, no collapse), and they do **real tech-tree work** — wood/stone pickaxes, furnace.
`imagined_return` ended 1.3–1.7 (vs 0023/0024's 8–22) and *trended down* over training: the
model got more honest as it learned. Residual: s0 still over-uses make_iron_sword (0.33, a
no-op without iron) — minor lingering exploitation, but it still moves + passes.

## Lesson

**Both levers were necessary; together they fix it.** Patch tokens (exp 0024) give the
agent the spatial info to *navigate*; the two-hot reward head bounds per-step reward so the
imagination AC can't *chase a hallucinated reward* (the mk_iron_sword/drink-at-cap/stationary
exploitation). 0024 (patch only) was seed-dependent because the reward was still inflated;
0025 (patch + bounded reward) makes **all** seeds move, explore, and craft, with calibrated
value and no collapse. This is the headline rung-3 result: a **consistent, non-degenerate
Crafter agent that genuinely plays.** The exploitation arc (0017→0025) is closed.

The remaining ceiling (~3–6 achievements, shallow/mid tree) is now the **compute-scaling**
question, not an exploitation bug — we are ~10× under-trained vs DreamerV3's Crafter budget.
→ next: the scaling-slope experiment (does 2×/4× compute climb the tree? exp 0026).

**Process win:** `behavior_report` (built from the maintainer's "it doesn't move"
observation) was the decisive metric. By eval_reward alone, 0025 s0 (2.35) ≈ 0024 s0 (2.47)
— *flat*. The real win is invisible in the score and obvious in the behavior: all seeds now
*move and craft* instead of farming saplings in place. Watching + the gate, not the number.

## Links

[[0024-dino-patch-tokens]] · [[0023-twohot-value]] · [[dreamerv3-2023]] · [[crafter]]
