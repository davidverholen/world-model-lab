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

_pending (dispatched)_

## Lesson

_pending_

## Links

[[0024-dino-patch-tokens]] · [[0023-twohot-value]] · [[dreamerv3-2023]] · [[crafter]]
