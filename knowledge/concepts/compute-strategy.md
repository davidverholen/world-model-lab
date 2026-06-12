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

## Measured: laptop thermals & power cap (2026-06-12)

90 s synthetic training load (`matmul` loop, 100% util): temp 62→78 °C and still
climbing; **power capped at ~45 W** (4070 Laptop TGP range is 35–115 W — ours is
near the floor); clocks ~1.15–1.2 GHz vs 3.1 GHz max boost; driver counters show
**SW Thermal Slowdown was already active ~396 s cumulative** across today's training.
Effective throughput is therefore ~35–40% of nominal 4070L → revised speedup
estimate for the 5070 Ti on GPU-bound training: **~4–6×** (was 2.5–3×).
Comfort/longevity: sustained low-80s °C chassis heat + full fans on a laptop.

## Measured: laptop vs desktop benchmark (2026-06-12, scripts/gpu_bench.py)

| | 4070 Laptop (45 W) | 5070 Ti desktop | ratio |
|---|---|---|---|
| matmul fp32 sustained | 7.0 TFLOPS | 31.4 TFLOPS | **4.5×** ✅ (predicted 4–6×) |
| recurrent train-step (b32, w24) | 30.1 upd/s | 27.9 upd/s | **0.93×** ✅ (predicted ≈1× for latency-bound) |

Both predictions confirmed in one run: the desktop is 4.5× on throughput-bound
work, and **exactly nothing** on our current small-model sequential-GRU training
step (kernel-launch latency dominates; the GPU idles either way). Consequences:
- dispatch decision = is the workload throughput-bound (big batches, conv-heavy,
  rung 3+) OR long-running (>~1 h ⇒ remote regardless, for thermal reasons)?
- to make the desktop pay off for recurrent training: bigger batch×window,
  torch.compile / CUDA graphs to fuse the GRU loop — try when rung-2 training grows.
- **latency-bound corollary (2026-06-12)**: since the GPU idles, N seeds can run as
  N concurrent processes on ONE GPU (bottleneck: CPU cores for env loops) — a 3-seed
  rung-2 sweep is ~1.5–2 h / ~$0.60 on a rented 4090, and renting an H100 for it
  would cost ~10× for zero (possibly negative) speedup. "Best available GPU" is the
  wrong axis below rung 3; sweep.py `--parallel N` is the planned enhancement.

## Key predictions (validate when first dispatching remotely)

- Rungs 1–2 workloads are env-loop/latency-bound: bigger GPUs ≈ 1.0–1.3× — don't switch.
- DreamerV3-small on Crafter 1M steps: ~30–45 h (4070L, throttled) → ~8–12 h
  (5070 Ti) → ~4–6 h (5090).
- Rental fits us unusually well: envs generate data → nothing to upload but code,
  nothing to download but checkpoints/logs.
- Trigger for wiring up the 5070 Ti (and writing the dispatch ADR): first run
  projected > **~1 h** on the laptop (lowered from 4 h after the thermal
  measurement — long runs throttle AND cook the chassis).

## Links

[[environment-ladder]] · CLAUDE.md hardware section · future ADR: remote dispatch
