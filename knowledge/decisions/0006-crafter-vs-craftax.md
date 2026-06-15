---
status: current
owner: world-model
scope: local
sources: [arxiv:2402.16801, arxiv:2109.06780]
verified: true
last_reviewed: 2026-06-13
---

# 0006: Crafter (original) over Craftax for rung 3

**Status:** accepted (2026-06-13)

## Context

Rung 3 of the [[0003-environment-ladder]] is Crafter — a 2D open-world survival game
(64×64 RGB, 17 actions, 22-achievement tech tree: wood→table→pickaxe→stone→…→diamond;
score = geometric mean of achievement rates). Two implementations exist:

- **Crafter** (original, Hafner 2021): Python/numpy, gym-compatible, pixel obs. Single
  env, CPU — slow. Integrates with our PyTorch stack as trivially as MiniGrid did.
- **Craftax-Classic** (Oxford 2024): a ground-up **JAX** rewrite, ~**257× faster** than
  Crafter — but that figure is for **end-to-end JAX** (PureJaxRL: env *and* agent on-GPU,
  no host transfer). Full **Craftax** adds NetHack-style depth/dungeons (169×).

Three facts decide this:

1. **The 257× is a model-FREE, sample-HUNGRY figure.** Craftax's headline ("1B env
   steps in <1 h on one GPU, 90% optimal") is *PPO*. Env speed dominates only when you
   need ~1e9 steps. We are **model-BASED / sample-efficient** (DreamerV3-class, ~1e6
   env steps): our bottleneck is GPU-bound world-model training, not env stepping. The
   speedup is largest in precisely the regime we are not in.
2. **Stack mismatch vs [[0001-pytorch-over-jax]].** Craftax is JAX with no native
   PyTorch path; using it with our world model needs custom JAX↔torch interop (per-step
   host/device friction that erodes the speedup) or rewriting our entire training stack
   (RSSM, heads, flywheel) in JAX. ADR 0001 chose PyTorch and pre-registered the *only*
   revisit trigger as "heavy parallel-env training where JAX's JIT is decisive" — not
   met here.
3. **We want pixels, not symbols.** Craftax's fastest mode is *symbolic* obs (~10× over
   its own pixel mode), but learning from pixels via a (frozen) vision encoder is our
   actual research thesis ([[frozen-encoder-lean]] in memory). Symbolic obs would be an
   easier, different problem that doesn't serve the goal — so we'd use pixel obs anyway,
   shrinking Craftax's edge further.

## Decision

**Use Crafter (original) for rung-3 first contact and the imagination-AC port.** Keep the
PyTorch stack; integrate Crafter behind the same env-wrapper interface as MiniGrid; use
pixel observations.

**Craftax is a documented escape hatch, not the default.** Switch only if a measured
trigger fires: (a) env-stepping becomes a measured ≥~30% wall-clock bottleneck at our
sample budget, or (b) we later move to a sample-hungry / massively-parallel regime. At
that point the cost (JAX interop or partial rewrite) is justified; until then it is not.
Decision is **measure-don't-assume**: profile env vs train share on the first Crafter run
before reopening this.

## Consequences

- Trivial, low-risk integration (Crafter is gym-compatible like MiniGrid); fast path to
  a "hello Crafter" baseline and the real science (does the calibrated imagination loop
  transfer up a rung).
- We forgo the 257× — acceptable because at ~1e6 model-based steps env time is not our
  wall. If profiling says otherwise, the escape hatch is pre-defined.
- Lighter dependency tree (no second CUDA framework alongside torch-CUDA).
- Apples-to-apples with the published Crafter score and DreamerV3 numbers (same env).
- Note vs SOTA line: the current Craftax SOTA papers ([[QUEUE]]: 2502.01591, 2605.16457)
  run on Craftax; if we later benchmark head-to-head against them we may need Craftax for
  comparability — flagged, not blocking first contact.

## Links

[[0001-pytorch-over-jax]] · [[0003-environment-ladder]] · [[0005-minecraft-milestone]] ·
[[compute-strategy]] · [[frozen-encoder-lean]]
