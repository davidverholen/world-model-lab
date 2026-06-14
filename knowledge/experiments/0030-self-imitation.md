---
status: draft
owner: human
scope: local
sources: [arxiv:2307.03486, arxiv:1806.05635]
verified: true
last_reviewed: 2026-06-14
---

# 0030: Self-imitation on achievement trajectories — the actor-side depth capstone (pre-registered)

## Hypothesis

The depth wall is actor/credit-side, established three ways: 0026/0027 ruled out compute,
0028 (Curious Replay, WM-side) lifted exploration but NOT depth, and 0029 cleared
perception (drink decodes at 0.94 in the belief — the agent *sees* its state). The Crafter
leaderboard ([[crafter]] scout 2026-06-14) confirms the locus: the best from-scratch agent,
**Achievement Distillation** ([[achievement-distillation-2023]]), is an *actor-side* method,
not a WM trick. And our newly-measured **Crafter Score = 2.61%** (crafter_steps_s1, 30 eps)
vs DreamerV3 14.5% reveals the real gap is **BREADTH**: we touch only 8/22 achievements; the
geometric-mean score is crushed by the ~14 we never unlock (incl. easy-ish `make_wood_pickaxe`,
`place_stone`, `eat_plant`, `defeat_skeleton`).

Lever: **self-imitation** (SIL, Oh et al. 2018) — reinforce the REAL actions whose realized
return beat the grounded critic, `(R − V)_+`-weighted log-prob, oversampling
achievement-unlock windows. Rationale: rare achievement trajectories are drowned by the
imagined policy-gradient; SIL up-weights them so the policy *consolidates* the successes
exploration occasionally finds (0028's behavior_report showed transient `place_stone` /
`make_wood_sword` that never stabilized into the eval union). SIL is the most direct attack on
the diagnosed credit bottleneck and near-free on our stack (reuses the real-belief roll +
real MC returns already computed for the replay-critic).

Predict: SIL lifts the **number of distinct achievements** at nonzero rate (breadth) and thus
the **Crafter Score** above 0028/0027's ~2.6%, by stabilizing the wood-tier crafts and
occasional stone touches. Target: move materially toward the open-source band (DreamerV3
14.5%) — reaching even ~6–10% would be a real climb; matching DreamerV3 is the stretch.
behavior_report must stay PASS (SIL can over-exploit a narrow success → watch action collapse).
Null (score flat) → SIL insufficient → 0031 = the faithful AD `L_pred` contrastive auxiliary
(achievement-aware representation; ingested, ~3.4pp of AD's gain).

## Setup

Base = **exp 0028** (`--curious`: CR provides the discovery SIL needs to consolidate) +
**`--self-imitation`** (sil_weight 0.5, sil_success_frac 0.5). One variable vs 0028. Applied to
the imagination AC only (policy-side; critic stays grounded by repval — keeps SIL a clean actor
lever). 2 seeds, 20 rounds. **`--eval-episodes 16`** for a less-noisy Crafter Score (now the
headline metric, reported every round). behavior_report on the checkpoints.

**Scope/fidelity note:** this is SIL (reinforce own successes), NOT Achievement Distillation
(contrastive achievement-aware representation). SIL attacks *credit* (our diagnosis); AD's
L_pred attacks *representation* (0029 says ours is fine) — hence SIL first, AD `L_pred` as the
0031 fallback. The loss is `actor_loss −= sil_weight · mean[(R−V)_+ · log π(a_real|belief)]`,
real beliefs detached (grad to actor only), baseline = EMA target_critic.

Command per seed: `python -m world_model.train_crafter --pool cls+patch --rounds 20
--round0-steps 10000 --actor-steps 10000 --updates-per-round 2000 --ac-updates-per-round 2000
--seq-batch 16 --window 24 --burn-in 8 --horizon 15 --eval-episodes 16 --eval-length 1000
--ep-length 2000 --repval 0.3 --curious --self-imitation --save runs/crafter_sil.pt --seed {0,1}`

## Result

_pending (dispatched)_

## Lesson

_pending_

## Links

[[0028-curious-replay]] · [[0029-stat-perception-probe]] · [[0027-step-scaling]] · [[achievement-distillation-2023]] · [[curious-replay-2023]] · [[crafter]] · [[hierarchy-and-credit]]
