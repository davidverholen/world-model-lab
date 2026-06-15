---
status: draft
owner: world-model
scope: local
sources: []
verified: false
last_reviewed: 2026-06-12
---

# Compute Strategy

> **Status note (2026-06-15): the desktop-remote path is retired.** The Windows
> desktop GPU is no longer available and its dispatch machinery was removed (ADR 0004
> superseded). Compute is now **local laptop GPU** + **on-demand rented cloud GPU**
> (vast.ai, maintainer-triggered). The `desktop GPU` rows below are kept as *measured
> reference data* (real benchmarks that still inform the latency-bound vs throughput-
> bound reasoning and rental sizing) — read them as "a ~16 GB / ~44 TFLOPS card," which
> now means a rented box, not a standing machine.

## What it is

Predicted (not yet measured) cost/benefit of our compute options per
environment-ladder rung. Prices checked 2026-06-12 (vast.ai: 4090 ~$0.31–0.44/hr,
5090 ~$0.53/hr, H100 from ~$0.90/hr).

| Option | VRAM / bandwidth / FP32 | Best for |
|---|---|---|
| laptop GPU (local) | 8 GB / 256 GB/s / ~15–20 TFLOPS | rungs 1–2: iteration speed, env-loop-bound work; short (<~1 h) runs |
| ~16 GB card (rented, e.g. 4090) | 16–24 GB / ~900 GB/s / ~44 TFLOPS | rung 3: multi-hour GPU-bound training (~2.5–3× laptop; measured on the retired desktop) |
| vast.ai burst (4090/5090) | 24–32 GB, ~$0.35–0.55/hr | rung 3–4: parallel sweeps (N seeds × arms), baselines |
| H100+ class | 80 GB, $1–2+/hr | rung 4+: ≥100M-param transformer world models only |

## Measured: laptop thermals & power cap (2026-06-12)

90 s synthetic training load (`matmul` loop, 100% util): temp 62→78 °C and still
climbing; **power capped at ~45 W** (laptop GPU TGP range is 35–115 W — ours is
near the floor); clocks ~1.15–1.2 GHz vs 3.1 GHz max boost; driver counters show
**SW Thermal Slowdown was already active ~396 s cumulative** across today's training.
Effective throughput is therefore ~35–40% of nominal laptop GPU → revised speedup
estimate for the desktop GPU on GPU-bound training: **~4–6×** (was 2.5–3×).
Comfort/longevity: sustained low-80s °C chassis heat + full fans on a laptop.

## Measured: laptop vs desktop benchmark (2026-06-12, scripts/gpu_bench.py)

| | laptop GPU (45 W) | desktop GPU | ratio |
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
  desktop GPU, seeds within 80 s of each other (exp 0009) — contention ≈ nil at N=3.
  Renting an H100 for this class would cost ~10× for zero (possibly negative)
  speedup. "Best available GPU" is the wrong axis below rung 3; sweep.py
  `--parallel N` is the planned enhancement.

## Key predictions (validate when first dispatching remotely)

- Rungs 1–2 workloads are env-loop/latency-bound: bigger GPUs ≈ 1.0–1.3× — don't switch.
- DreamerV3-small on Crafter 1M steps: ~30–45 h (laptop GPU, throttled) → ~8–12 h
  (desktop GPU) → ~4–6 h (5090).
- Rental fits us unusually well: envs generate data → nothing to upload but code,
  nothing to download but checkpoints/logs.
- Trigger for wiring up the desktop GPU (and writing the dispatch ADR): first run
  projected > **~1 h** on the laptop (lowered from 4 h after the thermal
  measurement — long runs throttle AND cook the chassis).

## Compute efficiency as the lab's strategy (2026-06-13)

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

## Run profile — where the time actually goes (measured 2026-06-13, scripts/profile_run.py)

A standard UTD run (~30 min) breaks down as:

| % | component | note |
|---|---|---|
| **79%** | training (42k GRU-loop updates) | **the only worthwhile target** — launch-bound (TF32 gives 1.02× ⇒ not matmul-bound) |
| 16% | MPC collection planning | GPU model-evals per env step |
| 4% | eval (full planner) | |
| **0.4%** | env stepping | **a compiled/JAX MiniGrid would save ~nothing — question closed** |

Acceleration verdict ("reasonable effort only"): no big easy wins; we're
already fleet-efficient via 6-wide parallelism (launch-bound runs overlap, GPU ~85%).
Measured levers on the training step: TF32 1.02× (enabled anyway, free), bf16 1.23×,
torch.compile reduce-overhead 1.33× **but CUDA-graph capture conflicts with our
freeze/reset optimizer rebuilds** → not worth the fragility now. **Shipped:** TF32
on by default + `--amp` (bf16) opt-in flag (OFF by default to keep fp32 retention
results comparable; use for long actor/Crafter runs). **Deferred:** torch.compile
until the actor work brings bigger models and drops the reset/freeze pattern.

## Crafter-phase rental mapping (2026-06-12)

- Actor *development* (MiniGrid-scale iterations): rental buys ~nothing
  (measured latency-bound regime; bottleneck is the redesign loop).
- Crafter *hyperparameter sweeps*: the killer rental use — ~10 parallel 4090s
  turn a week of sequential desktop tuning into overnight for $30–80. Trigger:
  first designed Crafter sweep; prerequisite: one session of vast provisioning
  (docker/setup + sweep.py backend). Cloud spend stays human-triggered
  (autonomous-mode guardrail).
- Single long runs: 5090 ≈ 2–2.5× the desktop GPU (~$3/run) — nice, not strategic.
- If the Python env loop becomes the wall: Craftax (JAX, env-on-GPU, ~100×)
  is the radical option — would reopen ADR 0001 (decision pending). Decision shape
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

## JAX rewrite — consolidated revisit triggers (2026-06-13; extends [[0001-pytorch-over-jax]])

Standing answer to "should we ever rewrite to JAX?": **not now, and the advantage
SHRINKS as we climb.** JAX's one decisive lever is end-to-end on-GPU parallel-env
throughput (vmap over 1000s of envs) — which only pays when env-stepping is the wall.
We are model-based / sample-efficient (~1e6 steps, GPU-bound training is the wall, env
stepping measured at **0.4%** above), so we don't need it; and the ladder moves AWAY
from JAX's sweet spot — **real envs can't be JAX-fused: Atari is C++, Minecraft is Java,
the real world isn't code.** JAX-reimplemented envs (Craftax/Brax/Gymnax) are the ONLY
place the speedup exists, i.e. just the Crafter rung. After Crafter the question is moot.

Revisit triggers (any one, none current):
1. **TPU access** appears — changes the whole calculus.
2. **Sample-hungry pivot** — model-based efficiency stalls and we fall back to scale, or
   we adopt population-based / massive-exploration methods needing 1e8–1e9 steps.
3. **Env-stepping profiles as ≥~30% wall-clock** at our scale (the [[0006-crafter-vs-craftax]]
   escape trigger).

Even then: **surgical, not wholesale** — spin up a JAX env (or a JAX agent) for that one
experiment via the dlpack hybrid / a `craftax/` sub-project (see Crafter-phase rental
mapping above), keeping the JEPA/world-model research stack in PyTorch. Framework choice
is bifurcated by workload, not fashion: hyperscale parallel model-free RL → JAX; our
representation-learning + pretrained-encoder + model-based profile → PyTorch (also where
DINOv3 / V-JEPA, our [[frozen-encoder-lean]] dependency, live). JAX is ascendant in its
niche, not a dying bet — we're just not in that niche.

## Upgrade path (<€5k home lab, decided 2026-06-12: not yet)

Buy trigger: Crafter-scale runs keep the desktop GPU >90% utilized for multi-hour
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

## Measured: vectorized env collection (2026-06-14)

Crafter collection is CPU-bound (single-threaded Python `env.step`), so the GPU idles
during the collect/eval phases (~75% util at 2 parallel seeds — the 25% idle is collection).
`train_crafter --n-envs N` now steps N Crafter workers in parallel (gymnasium AsyncVectorEnv,
NEXT_STEP autoreset) while batching the frozen encoder + RSSM policy on the GPU
(`collect_embed_vec`; opt-in, `--n-envs 1` = unchanged serial path). Measured **1.6× faster
collection at n=6 on the laptop** (random collection, 122→193 steps/s) — modest here because
the laptop CPU is weak and the per-step GPU-encode/IPC is serialized; expected to scale
better on a many-core box (desktop / rental), which is the point. Implication for renting
(see GPU-utilization note): to keep a rented card ~100% utilized, run **enough parallel seeds
to fill the collect-phase idle** AND size **vCPUs ≥ seeds** (one collection thread each);
`--n-envs` lowers the per-seed CPU idle so fewer seeds/cheaper instances saturate the GPU.
The bigger lever (deferred) is async collect-while-train (overlap collection with grad steps).
Equivalence to serial: unit-tested data invariants (contiguous episodes, no cross-boundary
windows) + integration smoke; not bit-identical (parallel streams).

## Links

[[environment-ladder]] · CLAUDE.md hardware section · [[0004-remote-dispatch]] (superseded — desktop-remote retired)
