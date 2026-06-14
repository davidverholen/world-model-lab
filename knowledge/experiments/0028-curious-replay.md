---
status: draft
owner: human
scope: local
sources: [arxiv:2306.15934, arxiv:2301.04104]
verified: true
last_reviewed: 2026-06-14
---

# 0028: Curious Replay — prioritize WM-learning toward surprising transitions (pre-registered)

## Hypothesis

0026 + 0027 ruled out both compute axes as the depth lever: more gradient-steps (replay
ratio) and more env-steps (data) both buy *breadth/reliability*, not the crafting tech-tree.
Verdict: depth is **exploration/hierarchy-bound** — the agent rarely discovers AND the world
model rarely *learns* the rare multi-step dynamics (table→wood-pickaxe→stone→furnace→iron),
because uniform replay drowns those rare transitions in a sea of walk/collect_sapling steps.

**Curious Replay** ([[curious-replay-2023]], Kauvar et al. ICML 2023) is the published fix and
— per the ingest — its Crafter gains are reported **specifically on the deep nodes** we are
stuck on (Collect Iron, Make Stone Pickaxe; Make Iron Sword 2/10 seeds vs 0/10 baseline),
not a uniform score lift. Mechanism: sample replay items with priority
`p_i = c·β^(v_i) + (|L_i| + ε)^α` — a count term (`v_i` = times sampled; down-weights
over-seen transitions) plus a model-loss term (up-weights high-prediction-error =
*surprising/novel* transitions). Rare deep-tree transitions have high WM loss → get replayed
more → the WM learns their dynamics → imagination through them becomes accurate → the actor
can finally plan the deep sequence.

Predict: WM-prioritized replay lifts the **frontier** achievement (the eval `unlocked=[...]`
union gains `collect_stone` / `make_stone_pickaxe` / `place_furnace`, absent in 0026/0027),
not just the shallow count. Movement stays healthy (behavior_report PASS). Null (frontier
unchanged) → curiosity-on-WM is not the lever either → escalate to structured
exploration/hierarchy (achievement-graph methods [[2305.00508]] / Achievement Distillation
[[2307.03486]]) — i.e. the problem is the ACTOR not discovering the path, not the WM not
learning it.

## Setup

Baseline = **exp 0027** recipe (the healthiest yet: 1× replay + 2× data — 20 rounds, 2000
WM + 2000 AC updates/round, `--pool cls+patch`, two-hot reward+critic, repval 0.3). One
variable: **`--curious`** — WM-training replay sampled by the Curious-Replay priority instead
of uniform. 2 seeds. Compare frontier achievement + behavior_report vs 0027.

**Scope (isolation):** Curious Replay applied to **WM training only**, not the imagination-AC
replay — this tests the cleanest mechanism (does teaching the WM the rare dynamics help?) and
keeps one variable. If partial, extending CR to the AC burn-in sampling is the obvious 0029.

**Priority signal `L_i`:** per-window **recon + balanced-KL** (the RSSM dynamics-prediction
error — the principled "adversarial curiosity" term; auxiliary reward/value/continue heads
excluded). All available in our frozen-encoder/cached-embedding stack. CR hyperparameters =
paper defaults (`α=0.7, β=0.7, c=1e4, ε=0.01, p_max=1e5`, tuned once by the authors and held
fixed including on Crafter). **Caveat (from ingest):** DINO embedding space is semantically
compressed, so embedding-recon error may be lower-variance than the paper's pixel loss —
whether it stays discriminative enough as a curiosity signal is the open empirical question
this experiment also answers.

Command per seed: `python -m world_model.train_crafter --pool cls+patch --rounds 20
--round0-steps 10000 --actor-steps 10000 --updates-per-round 2000 --ac-updates-per-round 2000
--seq-batch 16 --window 24 --burn-in 8 --horizon 15 --eval-episodes 8 --eval-length 1000
--ep-length 2000 --repval 0.3 --curious --save runs/crafter_cr.pt --seed {0,1}`

## Result

_pending (dispatched)_

## Lesson

_pending_

## Links

[[0027-step-scaling]] · [[0026-replay-ratio-scaling]] · [[0025-reward-head-patch]] · [[curious-replay-2023]] · [[dreamerv3-2023]] · [[crafter]] · [[hierarchy-and-credit]]
