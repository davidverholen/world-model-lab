---
status: draft
owner: human
scope: local
sources: [arxiv:2301.04104]
verified: true
last_reviewed: 2026-06-14
---

# 0027: Step/data scaling — is the crafting DEPTH data-bound? (pre-registered)

## Hypothesis

Exp 0026 showed more *training* (2× replay ratio) buys breadth/reliability but NOT crafting
depth, and over-trains toward in-place farming. So depth is not under-training-on-this-data.
The other compute axis is **data/exploration**: more env steps = more *chances* to discover and
reinforce the multi-step wood→table→pickaxe→stone sequence. Test it cleanly: 0025 recipe at **1×
replay** (keep exploration healthy — avoid 0026's over-training) but **2× the rounds** (20 vs
10 = 2× data, ~200k steps). Predict: if depth is data-bound, the eval-achievement *union* gains
**crafting tech-tree** achievements (`make_wood_pickaxe`, `place_table`, `collect_stone`, …) and
movement stays healthy. Null (still only shallow/survival achievements) → depth is NOT
compute-bound but **exploration/hierarchy-bound** (the agent doesn't *explore toward* the deep
sequence) → next is an exploration bonus / hierarchy, not more scale.

## Setup

0025 recipe (`--pool cls+patch`, two-hot reward + critic, repval 0.3, `--updates-per-round 2000
--ac-updates-per-round 2000` = 1× replay) with **`--rounds 20`** (2× data). 2 seeds. One variable
(rounds) vs 0025. Read the eval `unlocked=[...]` names (depth, not just count) + behavior_report
(movement must stay healthy). Crafting is rare/high-variance — 2 seeds is a weak depth estimate;
a clear gain or clear absence is the signal, a marginal one is inconclusive.

Command per seed: `python -m world_model.train_crafter --pool cls+patch --rounds 20
--round0-steps 10000 --actor-steps 10000 --updates-per-round 2000 --ac-updates-per-round 2000
--seq-batch 16 --window 24 --burn-in 8 --horizon 15 --eval-episodes 8 --eval-length 1000
--ep-length 2000 --repval 0.3 --save runs/crafter_steps.pt --seed {0,1}`

## Result

_pending (dispatched)_

## Lesson

_pending_

## Links

[[0026-replay-ratio-scaling]] · [[0025-reward-head-patch]] · [[crafter]] · [[hierarchy-and-credit]] · [[compute-strategy]]
