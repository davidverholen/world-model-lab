---
status: draft
owner: world-model
scope: local
sources: [arxiv:2511.08544, arxiv:2301.08243]
verified: true
last_reviewed: 2026-06-12
---

# Latent Collapse

## What it is

The failure mode of self-supervised latent prediction: if both the encoder and the
predictor are trained to minimize "predicted latent ≈ actual next latent", the
*globally optimal* solution is degenerate — the encoder maps every observation to the
same constant vector, prediction becomes trivially perfect, and the representation
contains nothing. Loss → 0, information → 0. It is not an instability; it is the
objective working as written.

Verified locally: experiment [[0001-naive-latent-regression]] — latent_std 0.0003
after 200 updates on MiniGrid, pred_loss 0.00000, probe R² ≈ 0.

## Why it matters here

Every JEPA-style world model we build has this failure mode by default; the choice of
anti-collapse mechanism is a core architecture decision. Collapse metrics
(per-dimension latent std, held-out probe R²) must be logged in *every* latent-model
experiment — a low prediction loss means nothing without them.

## Known fixes

| Mechanism | Used by | Idea |
|---|---|---|
| EMA target encoder + stop-gradient | BYOL, I-JEPA, V-JEPA | targets come from a slow copy; encoder can't chase itself |
| Variance/covariance regularization | VICReg | keep per-dim variance ≥ threshold, decorrelate dims |
| **SIGReg: distribution matching to N(0, I)** | LeJEPA | test embeddings against isotropic Gaussian (Epps-Pulley on random 1D slices); constant embedding = maximal test statistic |
| Reconstruction loss | Dreamer/RSSM | decoder grounds latents in pixels (pays for it by modeling detail) |
| Value/reward grounding | MuZero, TD-MPC2 | latents must predict reward/value, which varies |

Our pick: SIGReg ([[0002-sigreg-anti-collapse]]) — heuristic-free, one hyperparameter,
provably optimal target distribution, ~30 lines (`world_model/models/sigreg.py`).

## Measurement pitfall (learned in exp 0002)

In small fixed-layout envs, "collapse" reduces latent *amplitude*, not *information*:
the encoder is deterministic, so its ~dozens of distinct inputs still map to distinct
(microscopically separated) outputs. Consequences for metrics:

- never standardize latents before probing — it rescues collapsed encoders
  (verified: probe R² = 1.00 on a fully collapsed encoder after standardization);
- probe at raw scale with a fixed unit ridge, so amplitude matters like it would for
  any real downstream consumer;
- in envs with few distinct observations, held-out probe splits share observations
  with training — treat probe numbers as sanity checks, not quality measures, until
  the env is big/procedural enough;
- exp 0003 (Empty-8x8, CV'd α, 2k samples, 3 seeds): linear-probe R² is still wildly
  seed-dependent → demoted to diagnostic-only. **Primary metric is now the
  dynamics-vs-copy ratio**: variance-normalized held-out MSE of f(z,a) vs the copy
  baseline (z′=z). Consistent across seeds and measures what a world model is for.

## Open questions

- Does λ=0.05 transfer from LeJEPA's multi-view SSL setting to our action-conditioned
  dynamics setting, or does the prediction/regularization balance need retuning?
- SIGReg needs batch ≥ ~128 for a stable test statistic — constraint to remember when
  memory gets tight on bigger models.

## Links

[[jepa]] · [[lejepa-2025]] · [[0001-naive-latent-regression]] · [[0002-sigreg-anti-collapse]]
