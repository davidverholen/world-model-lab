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

**NULL for depth — Curious Replay lifts exploration/movement but does NOT move the crafting
frontier.** 2 seeds, 20 rounds, `--curious` (else identical to 0027). best_eval_reward s0 3.22
(round 18), s1 3.73 (round 12) — ≈ 0027 (3.47/3.72), s0 slightly below. Eval-achievement
*frontier* unchanged: full union s0 = {collect_wood, defeat_skeleton, defeat_zombie, eat_cow,
place_plant} (no place_table!); s1 = {collect_wood, defeat_zombie, eat_cow, **make_wood_sword**,
place_plant, place_table} — one transient wood-tier table craft (sword), same depth as 0027
s1's wood pickaxe. **No collect_stone, no stone_pickaxe, no furnace on either seed.**

The nuance: behavior_report **PASS** both, and Curious Replay measurably **improved
exploration** vs 0027 — moved_frac **0.23 / 0.19** (vs 0027's 0.13/0.14), bbox span up to 25
tiles, and s1's behavior_report reward **4.10** is the highest we've recorded. Faint frontier
flicker: s1's behavior histogram shows `place_stone:0.08` (implies it occasionally collects +
places stone in some episodes), but it never stabilizes into the eval union. The drink/do-spam
persists (top action `do` 0.33–0.35) and s0 shows the no-op `make_iron_sword:0.08` residual —
both consistent with the [[0029-stat-perception-probe]] actor-side finding. imagined_return
stayed calibrated (~1.4–1.9).

## Lesson

**Curious Replay does what it says — prioritizing surprising transitions increases exploration
(more movement, wider bbox, faint stone-tier touches) — but it is a WM-side lever and does NOT
crack the depth wall.** This is the predicted result: 0029 localized the bottleneck to the
actor/credit side (the agent *perceives* its state fine; it lacks the incentive/credit path to
value the long deep sequence), and CR improves the *world model's* coverage of rare dynamics,
not the *actor's* propensity to pursue them. So the depth plateau has now survived THREE levers
— replay-ratio (0026), data (0027), curiosity-prioritized replay (0028) — plus a perception
probe (0029) that cleared the encoder/WM. Every thread converges on **actor-side
exploration/credit-assignment** as the real frontier.

CAVEAT being checked: the maintainer flagged the Crafter leaderboard — Curious Replay is the
published *champion*, yet here it is NULL for depth. Resolving that tension (does the champion
itself actually reach the deep tree, or does ALL of Crafter SOTA plateau at the mid tier?) is a
scout task in flight — if even SOTA leaves stone_pickaxe→iron→diamond near-0%, our "stuck" is
substantially the universal Crafter ceiling, not only our bug, and the honest next move is
hierarchy/structured-exploration (Achievement Distillation 2307.03486 / 2305.00508) tempered by
realistic expectations about how far the deep tree is reachable at our compute at all.

→ next: gated on the leaderboard scout. If SOTA also plateaus → reset depth expectations + try
ONE actor-side lever (structured exploration / hierarchy) as the capstone, not endless WM work.

## Links

[[0027-step-scaling]] · [[0026-replay-ratio-scaling]] · [[0029-stat-perception-probe]] · [[curious-replay-2023]] · [[crafter]] · [[hierarchy-and-credit]]

## Links

[[0027-step-scaling]] · [[0026-replay-ratio-scaling]] · [[0025-reward-head-patch]] · [[curious-replay-2023]] · [[dreamerv3-2023]] · [[crafter]] · [[hierarchy-and-credit]]
