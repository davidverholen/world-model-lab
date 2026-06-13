---
status: draft
owner: agent
scope: local
sources: [arxiv:2301.08243, arxiv:2404.08471, arxiv:2506.09985, arxiv:2511.08544, openreview:lecun-path]
verified: false
last_reviewed: 2026-06-12
---

# JEPA — Joint-Embedding Predictive Architecture

## What it is

JEPA learns by predicting the *embedding* of a missing/future part of the input from
the embedding of a visible part — never reconstructing pixels. The claim
(openreview:lecun-path): pixel-space prediction wastes capacity on unpredictable detail
(leaf textures, noise); latent-space prediction lets the model keep only what is
predictable and semantically relevant.

Lineage: [[ijepa-2023]] (images, masked-block prediction, arxiv:2301.08243) → V-JEPA (video,
arxiv:2404.08471) → V-JEPA 2 (1M hours of video; action-conditioned variant V-JEPA 2-AC
does zero-shot robot pick-and-place, arxiv:2506.09985) → LeJEPA (theory-grounded
training objective, ~50 lines, arxiv:2511.08544).

## Why it matters here

Our first experiment (`world_model.collect`) is a deliberately naive JEPA: encoder +
action-conditioned latent predictor trained jointly on MSE. **This collapses** — the
encoder can map everything to a constant and get zero loss. The known fixes:

- EMA target encoder + stop-gradient (I-JEPA's approach)
- variance/covariance regularization (VICReg family)
- LeJEPA's SIGReg: regularize embeddings toward an isotropic Gaussian — provably
  optimal per arxiv:2511.08544, and the most attractive for us (single hyperparameter,
  simple to implement, code at github.com/rbalestr-lab/lejepa)

"When Does LeJEPA Learn a World Model?" (arxiv:2605.26379, May 2026) gives conditions
under which the recovered latents reflect the environment's true hidden structure:
roughly Gaussian latents and *broad exploration of the state space* — which directly
constrains our data-collection policy (random-only exploration may not suffice in
sparse environments).

## Open questions

- Does a LeJEPA-regularized latent predictor on MiniGrid produce latents that linear-probe
  for agent position / key possession? (Good first representation-quality metric.)
- JEPA gives representations and prediction, but no policy. Pair with planning (MPC in
  latent space, like V-JEPA 2-AC) or with an actor-critic ([[imagination-training]])?

## Links

[[world-models]] · [[latent-collapse]] · [[imagination-training]] · [[ami-labs]] ·
[[vjepa2-2025]] · [[lejepa-2025]]
