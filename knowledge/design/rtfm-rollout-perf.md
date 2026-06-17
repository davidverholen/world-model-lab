---
status: verified
owner: world-model
scope: local
verified: true
last_reviewed: 2026-06-17
---

# rtfm rollout performance — the collection loop is latency-bound, not GPU-bound

Live profiling of an exp0063 dispatch (4 seeds, n400, 80 rounds) on the desktop 16 GB GPU,
2026-06-17. **Conclusion: the rung-4 RTFM training is bottlenecked on the per-step collection loop, not
on the GPU.** A single seed leaves the GPU ~90% idle; we currently hide that only by running 4 seeds at
once (they overlap to ~52% util — the [[compute-strategy]] "concurrent seeds fill the idle GPU" trick).

## Evidence

| Signal | Reading | Meaning |
|---|---|---|
| GPU util | 52% across **4** seeds | one seed ≈ 13% |
| GPU power | 76 W / 300 W | barely working |
| GPU mem | 698 MiB/seed (3.8/16 GB) | batch-1 forwards |
| CPU/seed | ~109% (≈1 core), state `R` | single-core CPU-bound rollout |
| Host | 55% idle, 17% iowait | cores to spare |
| Rate | ~225 s/round early → ~5 h/seed | the four seeds overlap on the GPU |

## Root causes (all in [[rung4-manual-conditioned-agent]]'s collection loop, `train_rtfm.py:306-360`)

1. **Batch-1 DINO encode every env step** (`embed_of`, l.283) — the frozen ViT runs on one frame at a
   time (`unsqueeze(0)`). Batch-1 ViT is pure launch-latency: the GPU does microseconds of work then
   waits. Dominant GPU-side cost, spent inefficiently.
2. **Per-step GPU→CPU sync** (l.353) — `emb[0].cpu().numpy()` AND `nemb[0].cpu().numpy()` every step
   force two device syncs that serialize CPU↔GPU. `emb` is just last step's `nemb` re-transferred (a
   redundant copy).
3. **Pure-CPU env stepping, sequential** — 60 episodes/round stepped one at a time on one core; the
   crafter-rtfm env + per-step path-reward pointer is the CPU cost pegging the core. A fresh env is even
   rebuilt per episode (l.287).
4. **No TF32 / AMP / `torch.compile`** — `set_float32_matmul_precision` + autocast exist in
   `train_recurrent`/`dreamer` but are absent from `train_rtfm`; the training forward runs full fp32.
   (The encoder is correctly `@torch.no_grad()`, so no autograd leak.)

## Mitigations, by ROI

- **A — vectorize the rollout (the real win):** step B envs in parallel, ONE batched encode of B frames,
  bulk-transfer once per round. Attacks #1+#2+#3 together. Touches the pre-registered collection path
  (incl. the stateful path-reward pointer) → **reviewer-gate + a seed-parity test** (same seed ⇒
  equivalent replay/metrics) is mandatory so the exp0057→0063 ladder stays comparable. Implement behind
  a flag; keep the scalar loop for parity.
- **B — kill the per-step sync** (subset of A, parity-preserving on its own): keep a rolling CPU copy of
  the previous embed and transfer only `nemb` once. Halves per-step syncs with byte-identical buffers.
- **C — TF32 + opt-in bf16 autocast** on the train forward (`--amp`, ~1.2× proven in `train_recurrent`).
  Non-rollout, low-risk; modest here because the bottleneck is collection, not the update.
- **D — more concurrent seeds/experiments:** GPU has ~13 GB + ~50% util free → raises *throughput*, not
  per-seed latency. Already the standing strategy.
- **F — xFormers (DINO falls back to slow attention per the load warnings) / reuse env across episodes:**
  marginal while batch-1 (encode is latency- not compute-bound); revisit after A makes encode GPU-bound.

## Validation protocol (before any optimized pipeline becomes the new baseline)

Run the optimized code at **exp0061f1's exact config** (n400, uniform-0.5, 30 rounds) and confirm
swap_follow / swap_follow_s1 / conditional land within seed noise of the recorded baseline
(exact ~0.115, conditional ~0.33, step-1 ~0.345) AND measure the speedup. Only then re-dispatch the
deferred [[0063-rtfm-long-run-ceiling]] on the fast pipeline.

## Links

[[0063-rtfm-long-run-ceiling]] · [[rung4-manual-conditioned-agent]] · [[compute-strategy]] · [[0061-rtfm-backloading-confirm]]
