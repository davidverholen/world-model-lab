---
status: draft
owner: human
scope: local
sources: [arxiv:2301.04104]
verified: true
last_reviewed: 2026-06-14
---

# 0026: Replay-ratio scaling — are we under-trained? (pre-registered)

## Hypothesis

Exp 0025 fixed the exploitation; agents now move/explore/craft and reach ~3–6 achievements
(shallow/mid tree: pickaxes, furnace). The remaining ceiling is plausibly **compute-bound**,
not an algorithm wall (we are ~10× under-trained vs DreamerV3's Crafter budget). Test the
**cheapest** compute axis first: the **replay ratio**. We currently do 2000 WM + 2000 AC
updates per 10k collected env steps = **0.4 updates/env-step** — very low (DreamerV3 uses up
to 16×). Training is cached (frozen DINO embeddings in replay), so *more gradient steps per
datum is cheap*. Predict: if we're under-trained, **2× the updates/round lifts achievements
past the 0025 ~3–6** (deeper tree). Null (no lift) → not under-trained on this data → the
ceiling is data-bound (next: step-scaling = more rounds) or algorithm-bound (hierarchy).
WATCH: higher replay ratio on a fixed per-round dataset risks retention/overfit
([[retention]]) — two-hot + critic-on-replay should help; flag if eval destabilizes.

## Setup

0025 recipe (`--pool cls+patch`, two-hot reward + two-hot critic, repval 0.3) with
**`--updates-per-round 4000 --ac-updates-per-round 4000`** (2× vs 0025's 2000). Same 10
rounds, same collection (10k/round), 2 seeds. One variable (replay ratio) → clean
attribution. Baseline = exp 0025 (same recipe at 1× ratio). behavior_report on the
checkpoints (movement must hold; watch for collapse from over-training).

Command per seed: `python -m world_model.train_crafter --pool cls+patch --rounds 10
--round0-steps 10000 --actor-steps 10000 --updates-per-round 4000 --ac-updates-per-round 4000
--seq-batch 16 --window 24 --burn-in 8 --horizon 15 --eval-episodes 8 --eval-length 1000
--ep-length 2000 --repval 0.3 --save runs/crafter_rr.pt --seed {0,1}`

## Result

_pending (dispatched)_

## Lesson

_pending_

## Links

[[0025-reward-head-patch]] · [[dreamerv3-2023]] · [[crafter]] · [[retention]] · [[compute-strategy]]
