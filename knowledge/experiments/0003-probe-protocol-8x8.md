---
status: draft
owner: human
scope: local
sources: [arxiv:2511.08544]
verified: true
last_reviewed: 2026-06-12
---

# 0003: Trustworthy probes on Empty-8x8 + dynamics-vs-copy baseline (run 2026-06-12)

## Hypothesis

With a hardened protocol (Empty-8x8 = more distinct states; 2000 probe samples;
60/20/20 split with ridge α cross-validated on val, R² on test; no standardization;
3 seeds), SIGReg latents (λ=0.05, 1000 updates) show (a) consistently positive probe
R² across seeds, separated from the collapsed control, and (b) a held-out latent
dynamics error below the copy baseline (variance-normalized MSE ratio < 1), i.e. the
predictor learned action-conditioned dynamics — the rung-1 exit criterion.

## Setup

`uv run python -m world_model.collect --env-id MiniGrid-Empty-8x8-v0
--sigreg-weight {0.05|0} --updates 1000 --seed {0,1,2}` — 2000 random steps,
batch 128, ConvEncoder(128), 512 slices. Dynamics eval: 2000 fresh held-out
transitions (seed 2), MSE(f(z,a), z′)/Var vs MSE(z, z′)/Var.

## Result

Run 2026-06-12, code at milestone commit following ec6188f, CUDA/laptop GPU.

| seed | arm | latent_std | dynamics ratio (model/copy, <1 = learned) | probe R² |
|---|---|---|---|---|
| 0 | SIGReg | 0.83 | **0.203** (0.129 / 0.636) | 0.615 |
| 1 | SIGReg | 0.84 | **0.242** (0.153 / 0.631) | −1.269 |
| 2 | SIGReg | 0.82 | **0.296** (0.155 / 0.524) | 0.039 |
| 0 | control | 0.0001 | 1.629 | −0.007 |
| 1 | control | 0.0001 | 1.536 | −1.778 |
| 2 | control | 0.0001 | 1.741 | −1.227 |

## Lesson

1. **Hypothesis (b) confirmed, all seeds**: the SIGReg world model predicts held-out
   latent transitions 4–5× better than the copy baseline; collapsed controls are
   *worse* than copy once variance-normalized. Dynamics-vs-copy ratio is now our
   primary model-quality metric — it is consistent across seeds and directly measures
   what a world model is for.
2. **Hypothesis (a) refuted**: linear probe R² remains wildly seed-dependent
   (0.62 / −1.27 / 0.04) even with CV'd α and 2k samples. Linear position-decodability
   is not a reliable signal at this scale — demoted to diagnostic-only
   (see [[latent-collapse]] measurement section).
3. **Rung-1 exit criterion met** (no collapse + predictor beats trivial baseline)
   → recorded on [[environment-ladder]]; rung 2 (DoorKey, partial obs, memory) is open.

## Links

[[latent-collapse]] · [[0002-sigreg-anti-collapse]] · [[environment-ladder]] · [[lejepa-2025]]
