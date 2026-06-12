---
status: draft
owner: human
scope: local
sources: [arxiv:2511.08544]
verified: true
last_reviewed: 2026-06-12
---

# 0001: Naive latent regression collapses (planned)

## Hypothesis

Jointly training encoder + action-conditioned latent predictor with plain MSE on
MiniGrid-Empty-5x5 collapses: mean per-dimension latent std → ~0 within a few hundred
updates, because constant embeddings minimize the loss. (Textbook failure; we reproduce
it deliberately to establish the baseline and the collapse metric before adding SIGReg.)

## Setup

`uv run python -m world_model.collect` — 2000 random steps, ConvEncoder(128) +
LatentDynamicsPredictor, Adam 3e-4, 200 updates, batch 64. Collapse metric:
`latent_std` printed every 50 updates. Record commit hash + output here when run.

## Result

Formal run 2026-06-12, commit 6692aa2, default settings (`uv run python -m
world_model.collect`, seed 0, CUDA/RTX 4070): pred_loss=0.00000 from update 50 on;
latent_std 0.0007 → 0.0003 (updates 50→200). Full collapse, as hypothesized.
(Earlier same-day smoke run at reduced settings matched: latent_std 0.0005 @ 100.)

## Lesson

Confirmed: joint latent regression without anti-collapse machinery is degenerate by
construction — pred_loss is meaningless as a standalone metric. Every latent-model
experiment must log a collapse indicator (latent_std) and a representation metric
(probe R²). Follow-up: [[0002-sigreg-anti-collapse]].

## Lesson

_pending — expected: motivates SIGReg/EMA-target follow-up (experiment 0002)_

## Links

[[jepa]] · [[lejepa-2025]] · [[latent-collapse]] (create when 0002 lands)
