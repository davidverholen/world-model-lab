---
status: draft
owner: world-model
scope: local
verified: false
last_reviewed: 2026-06-17
---

# 0065: per-step swap-follow plateau over a LARGE budget — does 1-step and 2-step composition plateau or keep climbing?

**Status: PRE-REGISTERED — DISPATCHING (2 length arms in PARALLEL, large budget).** Operationalizes the
[[0063-rtfm-long-run-ceiling]] still-rising-vs-plateau question, sharpened on four axes: (a) **per-step**
tracking (step-1 = `swap_follow_s1`, full 2-step = `swap_follow`), (b) a **length-1 reference** arm (the
achievable single-step ceiling), (c) a **larger budget** — 120 rounds, ~18×/recipe at n400 (well into the
converged regime, so a flat tail is a *real* plateau), (d) **more seeds** per the exp0064 noise lesson.
Runs as a [[rtfm-rollout-perf]] exploit: both arms concurrent, GPU idle.

## Why

The reward axis is exhausted ([[0064-rtfm-path-reward-coef-sweep]]: magnitude is a null; back-loading
refuted in [[0061-rtfm-backloading-confirm]]). The live, untested question is whether the ~0.10–0.15
full-2-step `swap_follow` is a **real ceiling** or a **training-length artifact** — the 2026-06-17 audit
found it still gently rising at round 30, never run long. n400 at 120 rounds settles that cleanly: a flat
tail there is a genuine plateau, not under-training. The length-1 arm anchors the contrast — it tells us
the ceiling the single-step skill saturates at, so the length-2 gaps (Arm-A ceiling → Arm-B step-1 →
Arm-B full) localize exactly where competence stops.

## Hypothesis / fork

Two arms, proven config (n400, uniform-0.5 path reward, manual-aux 1.0, VR 0.3), 120 rounds. Read the
**whole trajectory** (not last-5) per arm — plateau vs continued slow climb is the question.

- **Arm A — length 1:** does pure single-step swap-follow plateau, and at what level? (the achievable
  1-step ceiling). Expect it to saturate; *where* is the reference.
- **Arm B — length 2:** track `swap_follow_s1` (step-1 within the 2-step task) and `swap_follow` (full
  chain). Does step-1 climb toward Arm-A's ceiling? Does the full chain plateau (~0.10–0.15) or keep
  climbing past ~0.15?

Forks:
- **Full 2-step keeps climbing past ~0.15** → the ceiling was a training-length artifact; composition
  cracks with enough budget → push budget / a real rung-4 break.
- **Full plateaus, step-1 high & near Arm-A's ceiling, `swapped ≈ 0`** → an *acceptable* competent
  ceiling (the [[0063-rtfm-long-run-ceiling]] framing): the agent reads + does step-1 reliably, the
  residual is genuine 2-step composition difficulty → next lever is the composition mechanism
  (backward curriculum / architecture), not budget or reward.
- **Full plateaus low AND step-1 also low / well below Arm-A** → step-1 itself doesn't transfer into the
  2-step task → an upstream representation/curriculum problem, not just composition.

## Setup

Two parallel dispatches; read full per-round trajectories of `swap_follow` + `swap_follow_s1`:

```
# Arm A — length 1 reference (3 seeds): pure 1-step, no curriculum, swap_follow == step-1
scripts/dispatch_rtfm.sh exp0065a 3 --rounds 120 --curriculum-rounds 0 --length 1 \
  --manual-aux-coef 1.0 --validated-reading-coef 0.3 --n-train-seeds 400 \
  --path-reward-coef 0.5 --path-reward-factor 1 --path-reward-decay 1.0

# Arm B — length 2 (5 seeds): 20 curriculum + 100 length-2; track s1 (step-1) + swap_follow (full)
scripts/dispatch_rtfm.sh exp0065b 5 --rounds 120 --curriculum-rounds 20 --length 2 \
  --manual-aux-coef 1.0 --validated-reading-coef 0.3 --n-train-seeds 400 \
  --path-reward-coef 0.5 --path-reward-factor 1 --path-reward-decay 1.0
```

8 procs concurrent (3 + 5); ~overnight wall (~120 rounds). Larger seed counts than exp0064 (noise
lesson). Plot the full trajectories at the end (`plot_experiment --compare`).

## Standing bar — tag: `LADDER-EXIT`

Whether full-2-step `swap_follow` plateaus or keeps climbing under a converged-regime budget — and, if it
plateaus, whether step-1 sits high enough (near the length-1 ceiling, `swapped ≈ 0`) to call it an
*acceptable competent* ceiling vs an upstream failure. Either resolves the rung-4 next-lever cleanly
(budget/break vs composition-mechanism vs representation).

## Links

[[0063-rtfm-long-run-ceiling]] · [[0064-rtfm-path-reward-coef-sweep]] · [[0061-rtfm-backloading-confirm]] · [[0057-rtfm-path-reward]] · [[rtfm-rollout-perf]] · [[mentored-learning-loop]]
