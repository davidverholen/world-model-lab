---
status: draft
owner: world-model
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

**Null for the depth hypothesis — 2× data buys breadth/reliability, NOT crafting depth.**
2 seeds, 20 rounds (~200k steps), 1× replay (else identical to 0025). best_eval_reward
**s0 3.47** (round 16), **s1 3.72** (round 18) — both *above* 0025's 2.35/2.47/3.47, and
eval_achievement count rose (peak 4.38 / 4.62 vs 0025's ~3.25). behavior_report **PASS**
both, and — the win over 0026 — **movement held** (moved_frac 0.13/0.14, span 13.8/16.3,
entropy 1.87/2.04, no farming-in-place collapse): 1× replay avoided 0026's over-training,
as predicted.

BUT the **depth did not exceed 0025**. Deepest achievement per seed: s0 stalls at
`place_table` (never a pickaxe); s1 reaches `make_wood_pickaxe` *once* (round 15, transient,
also in its behavior histogram at 0.06). The eval union is the same shallow/mid set as
0026 plus place_table + one transient wood pickaxe — **no `collect_stone`, no
`make_stone_pickaxe`, no `place_furnace`** (all of which *0025* reached on individual
seeds). So 2× exploration data lifted the shallow-achievement *count/reward* but did not
climb the tech-tree past where 1× data already reached. imagined_return stayed calibrated
(~1.0–3.3, no inflation).

Caveat: crafting is high-variance and this is 2 seeds vs 0025's 3 — the *absence* of stone/
furnace is partly seed luck. But the signal that matters is directional: more data did not
push the **frontier** deeper, only made the shallow tier more reliable. Same shape as 0026.

## Lesson

**Both compute axes are now ruled out as the depth lever — depth is exploration/hierarchy-
bound, not compute-bound.** 0026 (replay-ratio, more gradient steps/datum) → breadth not
depth + over-training. 0027 (step/data, more env steps) → breadth/reliability not depth,
movement healthy. Two orthogonal scaling knobs, same verdict: the crafting tech-tree
(table → wood pickaxe → stone → furnace → iron) does **not** open by throwing compute at the
current objective. The agent rarely *discovers and reinforces* the multi-step sequence —
this is the DoorKey-class long-horizon credit-assignment problem returning at the deep tree,
exactly as 0026 predicted. Per the research-cycle **stall rule**, the depth plateau has now
survived 2 redesigns (0026, 0027) with the same failure mode → **switch the thread from
scaling to structured exploration / hierarchy** (the pre-staged queue anchors:
Achievement Distillation 2307.03486, Curious Replay 2306.15934, structured-exploration
2305.00508). One *positive* carry-forward: 1× replay + 2× data is the healthiest recipe yet
(highest reward + count with movement intact) — use it as the 0028 baseline, not 0025.

→ next (exp 0028): the cheapest exploration lever first — **Curious Replay** (novelty/
surprise-prioritized replay sampling), which reuses our existing buffer + two-hot losses as
the surprise signal. Ingest it at method depth before implementing (lit-gate).

## Links

[[0026-replay-ratio-scaling]] · [[0025-reward-head-patch]] · [[crafter]] · [[hierarchy-and-credit]] · [[compute-strategy]]
