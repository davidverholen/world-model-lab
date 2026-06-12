---
status: stub
owner: agent
scope: local
sources: [arxiv:2511.08544, arxiv:2605.26379]
verified: true
last_reviewed: 2026-06-12
---

# LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics (Balestriero & LeCun, 2025)

**Lab:** Brown / NYU–AMI orbit · **Code:** https://github.com/rbalestr-lab/lejepa · **Read state:** abstract-only

## One-paragraph summary

A theory of JEPAs: the isotropic Gaussian is the optimal embedding distribution for
minimizing downstream prediction risk; introduces SIGReg (Sketched Isotropic Gaussian
Regularization) to push embeddings there. JEPA loss + SIGReg = LeJEPA: one trade-off
hyperparameter, linear time/memory, no stop-gradient/EMA heuristics, ~50 lines of code.
Validated on 60+ architectures; 79% ImageNet-1k linear probe with ViT-H/14.

Companion (May 2026, arxiv:2605.26379): "When Does LeJEPA Learn a World Model?" —
formal conditions (Gaussian latents, broad state-space exploration) under which LeJEPA
recovers the environment's true hidden structure. First rigorous public output of the
[[ami-labs]] research program; benchmark in the same paper finds current world models
brittle.

## Relevance to our experiments

SIGReg is our planned fix for the latent collapse that experiment 0001 deliberately
reproduces — simplest anti-collapse method to implement and theoretically grounded.
The companion paper's exploration condition directly informs our data-collection policy.

## Links

[[jepa]] · [[ami-labs]] · [[world-models]]
