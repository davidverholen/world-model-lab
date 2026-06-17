---
status: verified
owner: world-model
scope: local
verified: true
last_reviewed: 2026-06-17
---

# rtfm rollout performance — the collection loop is bound by crafter WORLD-GEN, not the GPU

Profiled the rung-4 RTFM collection on the desktop 16 GB GPU, 2026-06-17. **Conclusion: training
throughput is bottlenecked on CPU world generation at every episode `reset()`, NOT on the GPU and NOT on
the DINO encode.** A single seed leaves the GPU ~90% idle because the CPU is busy generating crafter
worlds (pure-Python simplex noise); we currently hide that idle GPU only by running several seeds at once
(the [[compute-strategy]] "concurrent seeds fill the idle GPU" trick).

## Evidence

System snapshot (4 seeds): GPU 52% util / 76 W of 300 W / ~698 MiB/seed; each seed pegs ~1 CPU core,
state `R`; host 55% idle. Rate ~225 s/round → ~5 h/seed.

`cProfile` of one real collection (30 episodes × 16 steps, real DINO, cuda), top by internal time:

| function | tottime | share |
|---|---|---|
| `opensimplex._extrapolate3` | 12.0 s | 63% |
| `opensimplex._noise3` | 2.2 s | 11% |
| `crafter/worldgen._set_material` (cum) | 15.6 s | (world-gen total **~79%**) |
| `torch._C._nn.linear` (DINO) | 0.40 s | 2% |
| all DINO/RSSM torch ops combined | < 1.5 s | < 8% |

**~79% of collection time is crafter world generation at `env.reset()`** (every episode builds a fresh
procedural world via simplex noise). The encode and the world-model forward are a rounding error.

## What was tried and REFUTED

**Mitigation A — vectorize the rollout (batch the encode across parallel envs).** Implemented behind a
default-off `--vec-collect` flag and A/B-timed on the real GPU at the real batch (60 envs), length-2,
training minimized so wall ≈ collection: **scalar 564.8 s vs vectorized 554.8 s — a ~1.8% difference,
i.e. no win.** Correct prediction from the profile: batching the encode cannot help a workload whose
time is in CPU world-gen, not GPU encode. The vectorized code was **reverted** (220 lines of
parity-risk for ~0 gain). Lesson: profile before optimizing — the GPU being idle meant the CPU was the
floor, not that batch-1 GPU latency was the fixable cost.

## Mitigations that actually fit the bottleneck

- **D — process-level parallelism (the immediate, no-code win).** Each run is ~1 CPU core + ~0.7–1 GB
  GPU + ~2.5 GB RAM, with the GPU otherwise idle. The box has **16 cores / ~21 GB free RAM / 16 GB GPU**,
  so ~10–12 runs fit concurrently (RAM-bound near ~8–10), GPU is never the limit. → run coefficient
  sweeps (many configs × seeds) **at once** instead of serially. This is the real throughput lever given
  the bottleneck, and it directly serves "test several coeffs in parallel." Tooling exists:
  `scripts/dispatch_rtfm.sh` (N seeds) + `scripts/sweep.py` / `/sweep`.
- **World-gen caching by seed (the real per-RUN latency lever — likely a crafter-rtfm handoff).** Seeds
  repeat: n_train_seeds=400 over 80 rounds × 60 episodes regenerates each seed's world ~12×. Memoizing
  the generated world per seed (snapshot once, restore on reuse) would cut world-gens ~12× and collapse
  the dominant 79% → a potential multi-× single-run speedup. But world generation lives in the
  benchmark-env domain (`crafter_rtfm` / crafter), which is **read-only to us** per the coordinator
  charter → this is a `../commons` handoff (a requirement: faster/cacheable world-gen for repeated
  seeds, or a JIT/numba simplex), not a local edit.
- **C — TF32 + transfer dedup (shipped, PR1 cc643e6).** Harmless and consistent with sibling scripts;
  immaterial to this bottleneck but kept.

## Validation protocol (for any change that touches collection semantics)

Run the optimized code at **exp0061f1's exact config** (n400, uniform-0.5, 30 rounds) and confirm
swap_follow / swap_follow_s1 / conditional land within seed noise of the recorded baseline
(exact ~0.115, conditional ~0.33, step-1 ~0.345) AND measure the speedup. (Byte-parity is impossible if
the change reorders RNG draws.)

## Links

[[0063-rtfm-long-run-ceiling]] · [[rung4-manual-conditioned-agent]] · [[compute-strategy]] · [[0061-rtfm-backloading-confirm]]
