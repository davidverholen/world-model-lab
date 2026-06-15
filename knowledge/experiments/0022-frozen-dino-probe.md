---
status: draft
owner: world-model
scope: local
sources: [arxiv:2411.04983, arxiv:2109.06780]
verified: true
last_reviewed: 2026-06-13
---

# 0022: Frozen DINOv2 features encode Crafter state — encoder bet validated

## Hypothesis

The [[frozen-encoder-lean]] plan for rung 3 is frozen DINOv2 + small learned dynamics
(DINO-WM recipe, arxiv:2411.04983). Risk: DINO is trained on natural photos; Crafter is
pixel-art/tile graphics, out of that distribution — its features may be useless here.
Test before building the pipeline (measure-don't-assume): if frozen DINO features encode
Crafter game state, a *linear* probe should predict scene content well above baseline.

## Setup

`scripts/probe_dino_crafter.py`. 600 frames from random play (8 seeds). Frozen
**DINOv2-S/14** (22.1M params, torch.hub), obs resized 64→98 (=14×7), ImageNet-normalized,
**CLS token** (384-d) extracted. Labels = multi-hot "which materials are in the agent's
9×9 view" (free, from `info['semantic']` cropped around `player_pos`; 13 informative
materials, base rate 5–95%). Linear probe (384→13, BCE, 80/20 split) + shuffled-label
control. CPU/GPU, ~30 s.

## Result

**Strong pass.** Mean test accuracy **0.981** vs majority baseline 0.804 (**+0.177 lift**);
per-material 0.93–1.00 across all 13 materials. Shuffled-label control collapses to 0.748
(≈ baseline) — the signal is real, not probe overfitting. Frozen DINOv2 CLS features
*linearly* encode Crafter scene content despite the out-of-distribution pixel-art domain.

## Lesson

**Frozen DINOv2-S validated as the Crafter encoder — no fine-tuning needed.** The single
biggest risk in the frozen-encoder approach (does natural-image SSL transfer to game
pixel-art?) is retired for one cheap probe. This is a *lower bound* on usefulness: CLS is
global; the 49 spatial patch tokens carry more (position of the tree/stone), which the
world model can use. Greenlights the DINO-WM-style build: frozen DINOv2 → small learned
RSSM dynamics. Architectural consequences (frozen ⇒): drop SIGReg (a frozen encoder can't
collapse), cache embeddings in replay (encode once, skip the encoder in every WM update),
immune to primacy/drift by construction ([[retention]]). DINOv3-S is an upgrade path.

## Links

[[frozen-encoder-lean]] · [[crafter]] · [[0006-crafter-vs-craftax]] · [[retention]] ·
[[0019-stochastic-latents]]
