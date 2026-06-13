---
status: draft
owner: human
scope: local
sources: [arxiv:2301.04104, arxiv:1811.04551]
verified: true
last_reviewed: 2026-06-13
---

# 0019: Stochastic latents (RSSM) — the exploitation fix (in progress) [Dave: invest now]

## Hypothesis

Exp 0017/0018 showed a DETERMINISTIC world model is maximally exploitable: the
imagination actor-critic drives the model to a single fake-reward state (imagined
return 2-5 vs real max 1; eval 0). Making the latent STOCHASTIC (RSSM, PlaNet/Dreamer
lineage) fixes this: imagination samples z from a prior p(z|h) trained (via KL) to
match the posterior q(z|h,obs) over real data, so imagined rollouts stay near the
real distribution and there is no single exploitable trajectory. Expect:
imagined_return ≤ ~1.0 AND eval rises off 0, approaching the planner (~25-35%).
Dave chose this over the DAgger shortcut ("invest now") — RSSM is needed for Crafter
regardless. Build incrementally (module → train loop → agents → smoke → dispatch),
not big-bang, given it's the largest architecture change in the project.

## Setup

Gaussian RSSM (models/rssm.py): state=(h deterministic GRU, z stochastic);
belief=concat(h,z) for the heads. WM loss = recon(belief→embed) + KL(post||prior)
with free bits + KL balancing (DreamerV3 β_dyn 0.5 / β_rep 0.1); encoder+SIGReg kept
for embed anti-collapse. Imagination samples z~prior. Reuses continue predictor
(0018), actor-critic (0017). DoorKey-6x6, 3 seeds, desktop.

## Result

_pending (building)_

## Lesson

_pending_

## Links

[[0017-imagination-actor-critic]] · [[0018-continue-predictor]] · [[dreamerv3-2023]] ·
[[imagination-training]] · [[generative-vs-predictive]]
