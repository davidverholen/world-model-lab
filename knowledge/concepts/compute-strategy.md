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
- **latency-bound corollary (2026-06-12, measured)**: since the GPU idles, N seeds
  run as N concurrent processes on ONE GPU (bottleneck: CPU cores for env loops).
  Measured: 3-seed DoorKey-6x6 sweep (7 rounds each) = **35 min wall** on the
  5070 Ti, seeds within 80 s of each other (exp 0009) — contention ≈ nil at N=3.
  Renting an H100 for this class would cost ~10× for zero (possibly negative)
  speedup. "Best available GPU" is the wrong axis below rung 3; sweep.py
  `--parallel N` is the planned enhancement.

## Key predictions (validate when first dispatching remotely)

- Rungs 1–2 workloads are env-loop/latency-bound: bigger GPUs ≈ 1.0–1.3× — don't switch.
- DreamerV3-small on Crafter 1M steps: ~30–45 h (4070L, throttled) → ~8–12 h
  (5070 Ti) → ~4–6 h (5090).
- Rental fits us unusually well: envs generate data → nothing to upload but code,
  nothing to download but checkpoints/logs.
- Trigger for wiring up the 5070 Ti (and writing the dispatch ADR): first run
  projected > **~1 h** on the laptop (lowered from 4 h after the thermal
  measurement — long runs throttle AND cook the chassis).

## Compute efficiency as the lab's strategy (Dave, 2026-06-13)

Stated principle, generalized from the architecture-testing tier ladder
([[hierarchy-and-credit]]) and a DeepMind talk's claim that for AGI compute
efficiency may matter more than architecture.

**The claim, refined**: architecture and compute-efficiency are not orthogonal —
the architectures that matter are the ones with a better capability-per-FLOP
*slope* (transformers vs LSTMs; world models' ~10–100× sample efficiency vs
model-free). Our world-model bet IS a compute-efficiency bet. So "efficiency >
architecture" really means "judge architectures by their efficiency curve."

**For a home lab this is the entire moat.** We cannot out-scale big labs. The only
available research positions are (a) compute-efficiency contributions (more with
less — exactly what the field says matters) and (b) questions big labs aren't
asking (our niche frontier). Both are efficiency plays. Efficiency governs how we
work AND what we research.

**Operating rules:**
- *Minimum SUFFICIENT scale* (not minimum scale): identify the smallest scale at
  which the tested effect CAN appear, pay exactly that. Too cheap = never see it
  (cf. tier-2 can't show skill reuse). This is the compute analog of the
  whitepaper's "minimum needed context".
- *Measure slopes, not endpoints*: prefer "does A have a better efficiency slope
  across small→medium envs?" over "does A win at full scale?" — measurable cheaply
  and extrapolable (scaling-laws style). The tier ladder is a scaling curve.
- Already-embodied tactics that ARE efficiency: literature-first (a $0 search
  replaces a $50 sweep), pre-registration with counter-outcomes (never run an
  experiment that can't change your mind), cheapest-env-that-reveals-the-effect,
  parallel seeds on idle GPU, uint8 replay, best-checkpoint guard.

## Crafter-phase rental mapping (asked by Dave 2026-06-12)

- Actor *development* (MiniGrid-scale iterations): rental buys ~nothing
  (measured latency-bound regime; bottleneck is the redesign loop).
- Crafter *hyperparameter sweeps*: the killer rental use — ~10 parallel 4090s
  turn a week of sequential desktop tuning into overnight for $30–80. Trigger:
  first designed Crafter sweep; prerequisite: one session of vast provisioning
  (docker/setup + sweep.py backend). Cloud spend stays human-triggered
  (autonomous-mode guardrail).
- Single long runs: 5090 ≈ 2–2.5× the 5070 Ti (~$3/run) — nice, not strategic.
- If the Python env loop becomes the wall: Craftax (JAX, env-on-GPU, ~100×)
  is the radical option — would reopen ADR 0001 (Dave's call). Decision shape
  pre-agreed (2026-06-12): NO framework-abstraction layer (JAX's value — fused
  jit/vmap/scan incl. the env — is exactly what abstractions can't express;
  meta-framework maintenance would displace research). Instead: hybrid dlpack
  spike first; if decisive, a one-way `craftax/` sub-project port of the rung-3
  agent (baselines/ppo pattern), parity via golden tests against the PyTorch
  reference, the KB agent-architecture page as the framework-neutral spec.
  JEPA-side research stays PyTorch.
  Nuances (2026-06-12): no PyTorch port of Craftax known (verify at trigger);
  its gym wrapper + dlpack hybrid gives PyTorch most of the env speedup without
  a port — hence spike-first. JAX's other real advantages for us: scan/jit would
  fuse our measured kernel-launch-bound GRU loop (28 upd/s problem), vmap = native
  parallel seeds. PyTorch's partial answer (torch.compile/CUDA graphs) is UNTESTED
  on our stack — cheap experiment, do BEFORE any framework decision. JAX-native
  MiniGrid-likes exist too (XLand-MiniGrid, gymnax).
  Step-2→3 gain estimate (2026-06-12, to be replaced by spike measurements):
  ~2–10× for env-dominated workloads (tiny policy, huge batches) but ~1.5–3× for
  OUR world-model-heavy training (model matmuls dominate; GRU-loop overhead is
  torch.compile's target). Strategic: fusion is Craftax-specific — Atari (C++)
  and Minecraft (Java) can never fuse, so the hybrid IS the ladder's lasting
  pattern; a full port pays only under massive Craftax-native experiment volume.

## Upgrade path (<€5k home lab, decided 2026-06-12: not yet)

Buy trigger: Crafter-scale runs keep the 5070 Ti >90% utilized for multi-hour
stretches AND experiments queue behind it (gpu_bench + run timestamps make this
measurable, not vibes). Then, in order of €-efficiency:
1. used RTX 3090 24 GB (~€700) as second GPU in the desktop — VRAM headroom +
   a parallel lane;
2. dedicated tower: RTX 5090 32 GB + 16-core CPU + 96 GB RAM (~€3.5–4k) —
   ~2–2.5× throughput, rung-4 headroom;
3. vast.ai bursts stay the answer for rare big sweeps regardless (at ~€0.35/kWh,
   owning only beats renting for daily sustained use).
Rejected: enterprise rack gear (>€5k for non-ancient silicon, noise, idle power),
Mac unified memory (MPS second-class for the PyTorch/CUDA RL ecosystem).

## Links

[[environment-ladder]] · CLAUDE.md hardware section · future ADR: remote dispatch
