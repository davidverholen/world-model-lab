---
status: draft
owner: world-model
scope: local
sources: [arxiv:2301.08243]
verified: true
last_reviewed: 2026-06-12
---

# I-JEPA: Self-Supervised Learning from Images with a JEPA (Assran et al., 2023)

**Lab:** Meta AI · **Code:** github.com/facebookresearch/ijepa · **Read state:**
skimmed (method sections via full text, 2026-06-12)

## One-paragraph summary

First concrete instantiation of LeCun's JEPA: predict the *representations* of
several target blocks of an image from one context block — no pixel reconstruction,
no augmentations, no contrastive negatives. Semantic features emerge because targets
are large blocks (forcing object-level abstraction) and prediction happens in latent
space (unpredictable pixel detail costs nothing).

## Key ideas (implementation-grade, fetched 2026-06-12)

- **Three networks**: context encoder (ViT), target encoder (EMA copy), predictor
  (narrow ViT, width 384, depth 6–16 by encoder size).
- **Multi-block masking**: 4 target blocks, scale (0.15, 0.2), aspect (0.75, 1.5);
  1 context block, scale (0.85, 1.0), targets cut out of it. Targets are masked at
  the *output* of the target encoder (they get full-image context) — subtle but key.
- **Anti-collapse = asymmetry**: EMA target encoder (momentum 0.996 → 1.0 linearly)
  + predictor asymmetry. No variance/covariance regularizer (contrast: [[lejepa-2025]]
  replaces exactly this heuristic with SIGReg).
- **Loss**: average L2 in representation space on target patches.
- Efficiency: ViT-H/14 79.3% linear @ 300 epochs vs MAE 77.2% @ 1600 — ~5× fewer
  iterations; latent prediction is cheap.

## Relevance to our experiments

The masking-based pretext is image-internal — our temporal analogue replaces
"predict masked blocks" with "predict next-step latents given action", but the
collapse-prevention design space is identical: I-JEPA's EMA-teacher vs our SIGReg
(exp 0002 chose SIGReg; EMA-teacher remains the untested alternative if SIGReg
limits us). Target-masked-at-output is a trick worth remembering when we add
spatial latents (patch-level world models, rung 3+).

## Links

[[jepa]] · [[lejepa-2025]] · [[vjepa2-2025]] · [[latent-collapse]] · [[lecun-2022-path]]
