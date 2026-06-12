---
status: draft
owner: human
scope: local
sources: [arxiv:2511.08544]
verified: true
last_reviewed: 2026-06-12
---

# 0002: SIGReg prevents latent collapse (run 2026-06-12)

## Hypothesis

Adding SIGReg (LeJEPA's Epps-Pulley sliced test against N(0,I), λ=0.05) to the joint
latent-regression objective of experiment [[0001-naive-latent-regression]] prevents
collapse on MiniGrid-Empty-5x5: latent_std stays O(1) instead of →0, AND the latents
become linearly decodable — held-out ridge-probe R² (latent → agent x,y) ≥ 0.8,
versus ≈0 for the collapsed baseline.

## Setup

`uv run python -m world_model.collect --sigreg-weight 0.05` (SIGReg arm) vs
`--sigreg-weight 0` (control with probe). 2000 random steps, ConvEncoder(128),
batch 128, Adam 3e-4, 200 updates, 512 slices, 17 integration points, seed 0.
SIGReg applied to concatenated [z, z_next] batch embeddings.
Probe: ridge (α=1e-3) from frozen latents to agent (x,y), 500 fresh steps, 80/20 split.

## Result

Run 2026-06-12, code as of milestone commit following 6692aa2, CUDA/RTX 4070, seed 0.

| arm | updates | latent_std (end) | probe R² (final protocol) |
|---|---|---|---|
| control λ=0 | 200 | 0.0003 | **−0.49** |
| SIGReg λ=0.05 | 200 | 0.75 | 0.28† |
| SIGReg λ=0.05 | 1000 | 0.74 (stable 750–1000) | **0.25** |

† earlier probe protocol, not comparable across rows; kept for the record.

Probe protocol churn (the unplanned finding): three probe variants gave wildly
different answers on the *same* latents — α=1e-3 unstandardized: 0.63 @ 200 but
−0.91 @ 1000 (ill-conditioned, 129 features / 400 samples); standardized + α=1.0:
**R²=1.00 for the fully collapsed control** (standardization rescales microscopic
residual variation to full amplitude; Empty-5x5 has only ~36 distinct observations,
so train/test share them and ridge memorizes); final protocol: raw scale + α=1.0 —
collapse reads negative, SIGReg reads positive, directionally sane.

## Lesson

1. **SIGReg works as anti-collapse**: latent_std stabilizes ~0.75 with zero heuristic
   machinery; λ=0.05 transfers from LeJEPA's SSL setting to action-conditioned
   dynamics without tuning. Hypothesis part 1 confirmed.
2. **Probe target (R² ≥ 0.8) not met** (0.25). Unclear how much is representation vs
   metric: strong ridge biases R² down, 500 samples is few, and Empty-5x5's tiny
   observation set makes linear-probe "generalization" near-meaningless.
3. **Collapse in toy envs is amplitude loss, not information loss** — a deterministic
   encoder keeps distinguishable (microscopic) outputs for its ~36 distinct inputs.
   Any probe that normalizes scale will lie. Recorded in [[latent-collapse]].
4. Follow-up (exp 0003): proper probe protocol — cross-validated ridge α, ≥2k probe
   samples, multiple seeds, and a larger env (Empty-8x8 / DoorKey) before trusting
   any representation-quality number.

## Links

## Links

[[jepa]] · [[lejepa-2025]] · [[latent-collapse]] · [[0001-naive-latent-regression]]
