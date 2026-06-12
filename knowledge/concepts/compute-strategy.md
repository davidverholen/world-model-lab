---
status: draft
owner: human
scope: local
sources: []
verified: false
last_reviewed: 2026-06-12
---

# Compute Strategy

## What it is

Predicted (not yet measured) cost/benefit of our three compute options per
environment-ladder rung. Prices checked 2026-06-12 (vast.ai: 4090 ~$0.31–0.44/hr,
5090 ~$0.53/hr, H100 from ~$0.90/hr).

| Option | VRAM / bandwidth / FP32 | Best for |
|---|---|---|
| RTX 4070 Laptop (local) | 8 GB / 256 GB/s / ~15–20 TFLOPS | rungs 1–2: iteration speed, env-loop-bound work |
| RTX 5070 Ti (desktop, idle) | 16 GB / ~900 GB/s / ~44 TFLOPS | rung 3: multi-hour GPU-bound training (~2.5–3× laptop) |
| vast.ai burst (4090/5090) | 24–32 GB, ~$0.35–0.55/hr | rung 3–4: parallel sweeps (N seeds × arms), baselines |
| H100+ class | 80 GB, $1–2+/hr | rung 4+: ≥100M-param transformer world models only |

## Key predictions (validate when first dispatching remotely)

- Rungs 1–2 workloads are env-loop/latency-bound: bigger GPUs ≈ 1.0–1.3× — don't switch.
- DreamerV3-small on Crafter 1M steps: ~20–30 h (4070L) → ~8–12 h (5070 Ti) → ~4–6 h (5090).
- Rental fits us unusually well: envs generate data → nothing to upload but code,
  nothing to download but checkpoints/logs.
- Trigger for wiring up the 5070 Ti (and writing the dispatch ADR): first run
  projected > ~4 h on the laptop.

## Links

[[environment-ladder]] · CLAUDE.md hardware section · future ADR: remote dispatch
