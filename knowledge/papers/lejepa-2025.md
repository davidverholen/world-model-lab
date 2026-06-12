---
status: draft
owner: agent
scope: local
sources: [arxiv:2511.08544, arxiv:2605.26379]
verified: true
last_reviewed: 2026-06-12
---

# LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics (Balestriero & LeCun, 2025)

**Lab:** Brown / NYU–AMI orbit · **Code:** https://github.com/rbalestr-lab/lejepa · **Read state:** skimmed (method sections via HTML + reference repo, 2026-06-12)

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

## Key ideas (method details)

- **Theory**: isotropic Gaussian N(0, I) is the embedding distribution minimizing
  downstream prediction risk → regularize toward it instead of heuristics.
- **SIGReg** (Def. 2): mean over random unit directions `a` of a univariate test
  statistic on the projected embeddings. Test = **Epps-Pulley**: weighted L2 distance
  between empirical characteristic function and the N(0,1) CF,
  `EP = N ∫ |φ̂(t) − e^{−t²/2}|² w(t) dt`, `w(t)=e^{−t²}`; trapezoid with 17 points on
  [−5, 5]. Directions resampled each step (1024 recommended, 512 competitive).
- **Total loss** (Eq. 9): `(1−λ)·prediction + λ·SIGReg`, λ=0.05 recommended;
  batch ≥ 128.
- **Stability**: EP gradients/curvature are bounded (Thm 4) — no schedulers,
  no stop-gradient, no EMA teacher. O(N) time/memory, ~50 LOC.

## Relevance to our experiments

SIGReg is our planned fix for the latent collapse that experiment 0001 deliberately
reproduces — simplest anti-collapse method to implement and theoretically grounded.
The companion paper's exploration condition directly informs our data-collection policy.

## Links

[[jepa]] · [[ami-labs]] · [[world-models]]
