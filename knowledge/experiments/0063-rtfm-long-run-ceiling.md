---
status: stale
owner: world-model
scope: local
verified: false
last_reviewed: 2026-06-17
---

# 0063: n400 long run — where does swap_follow settle, and is it an *acceptable* (competent-agent) ceiling?

**Status: PRE-REGISTERED — DEFERRED pending a rollout-perf rebuild (stale, NOT a result).** Dispatched
2026-06-17 and stopped after ~4 rounds: live profiling showed the run is **rollout-latency-bound**, not
GPU-bound (GPU 52% util / 76 W of 300 W across 4 seeds, ~698 MiB/seed; each seed pegs ~1 CPU core on the
batch-1, sync-per-step collection loop). Burning ~5 h/seed on a pipeline that leaves the GPU 90% idle is
waste, so the run is deferred until the collection path is optimized + parity-validated. **The
hypothesis, config, and acceptance bar below are UNCHANGED** — only the implementation underneath gets
faster. Perf diagnosis + plan: [[rtfm-rollout-perf]] — **note: the vectorized-rollout rebuild was abandoned**
(world-gen is the bottleneck, not the encode; the long run just runs at scalar speed, with idle-GPU
parallelism for breadth). **SUPERSEDED by [[0065-rtfm-per-step-plateau-large-budget]]**, which runs this
n400 ceiling question at a larger budget (120 rounds), with per-step tracking, a length-1 reference arm,
and more seeds (exp0064 noise lesson). This page is kept for the acceptable-ceiling framing it originated.

Reframed (2026-06-17): this is **not** a pure "break the ceiling" bet anymore. It runs the proven-best
**n400** config much longer and asks a two-part question — (1) *does* swap_follow plateau, and (2) if it
does, is that plateau an **acceptable** ceiling, i.e. one where the agent already **acts confidently in
the given env** (reliably reads + executes the manual, doesn't act habitually, consistent across seeds),
so the residual gap is genuine 2-step *composition* difficulty rather than a training/representation
failure. The n1600 **generalization** long-run is a deliberately *separate* future experiment, not folded
in here (keep one variable moving per run).

## Why (the audit finding)

The 2026-06-17 claim audit showed the uniform-0.5 / floor-0.5 path-reward runs do NOT plateau by round
30 — they keep gently climbing:
- [[0059-rtfm-floor-confirm]] (floor 0.5, 4 seeds): per-round swap_follow ~0.065 (r20–23) → ~0.105
  (r26–29), **r29 ≈ 0.13**.
- [[0061-rtfm-backloading-confirm]] f1 (uniform 0.5, 2 seeds): **r29 ≈ 0.15**.

Crucial contrast: [[0051-rtfm-flat-vr-optimization]]'s budget probe WAS flat ~0.11 over 40 rounds — but
it had **no path reward**. So the path reward appears to drive a continued climb → **a longer run is a
live, untested lever** (this corrects the earlier "longer training won't help", which was based on the
no-path-reward exp0051). exp0063 runs the best-and-still-rising config much longer **at n400**, where
80 rounds gives ~9–12× practice/recipe (deep into the converged regime) — so wherever it settles is a
*real* settling point, not an under-training artifact. That cleanliness is exactly why this is the n400
config and not n1600 (which at 80 rounds reaches only ~2–3×/recipe, so a plateau there would stay
ambiguous — that ambiguity is the reason n1600-long is split into its own experiment).

## Hypothesis / fork

Take the **confirmed best, still-rising config** — uniform-0.5 path reward
(`--path-reward-coef 0.5 --path-reward-factor 1 --path-reward-decay 1.0`), **n400** (clean: change ONLY
the training length vs [[0061-rtfm-backloading-confirm]] f1) — and run **20 curriculum + 60 length-2 =
80 rounds** (vs the usual 10+20=30). exp0061f1 (n400, 30 rounds) is the direct 30-round reference, so
this isolates **training length** at fixed coverage. Read the full length-2 trajectory (rounds 20–79),
not just last-5, to judge plateau vs continued climb, and watch swap_follow / swap_follow_s1 / grounding
/ conditional together.

- **swap_follow keeps climbing past ~0.15** → no real ceiling at n400/this-reward; the path reward
  genuinely cracks the 2-step wall given enough rounds. (Watch swap_follow specifically, NOT `correct`
  — exp0051 showed `correct` can climb via experience-based completion that swap_follow rightly
  discounts.) → a clean rung-4 break.
- **swap_follow plateaus AND the agent acts confidently** (the *acceptable*-ceiling case, criteria
  below) → the ~0.10–0.15 ceiling is a genuine **composition-difficulty floor**, not a training failure:
  the agent reliably reads and executes the manual and just can't yet chain two steps. This is an
  acceptable rung-4 stopping point — accept the competent-but-shallow agent and move the lever to the
  *composition* mechanism (backward curriculum / progress-state actor) on its own merits, OR judge the
  rung passed for now.
- **swap_follow plateaus AND the agent does NOT act confidently** (the *unacceptable*-ceiling case) →
  it's a real training/representation failure (low step-1, habitual / no grounding, or high variance) →
  the fix is upstream (reader/representation or curriculum), and a longer run won't help.
- **Divergence/collapse over the long run** → instability at 80 rounds (watch imagined_return, recon).

## What counts as "acts confidently in the given env" (the acceptable-ceiling bar)

A plateau is **acceptable** only if, at the settled level (mean over rounds 60–79, multi-seed), ALL of:

1. **Reads + executes step-1 reliably** — `swap_follow_s1` ≳ 0.35 on held-out swapped manuals (it
   confidently performs the *manual-specified* first action, not a habitual default).
2. **Genuinely uses the manual, not habit** — `grounding` (CORRECT − NONE) clearly > 0, and `swapped`
   reward ≪ `correct` (it adapts its recipe to the manual rather than running a fixed habitual one).
3. **Follows through at a real rate** — conditional `swap_follow / swap_follow_s1` ≳ 0.30 (when it nails
   step-1 it completes the chain a meaningful fraction of the time, i.e. it's *trying* to compose).
4. **Consistent + stable** — low cross-seed variance and no downward drift / collapse over rounds 60–79
   (confidence = reproducible competence, not a lucky seed or a transient peak).

Miss any of 1–3, or fail 4, and the plateau is the *unacceptable* kind → upstream fix. (These thresholds
are a proposal grounded in the exp0057/0059/0061 levels — adjust before dispatch if the maintainer reads
"confident" stricter/looser.)

## Setup

`scripts/dispatch_rtfm.sh exp0063 4 --rounds 80 --curriculum-rounds 20 --length 2 --manual-aux-coef
1.0 --validated-reading-coef 0.3 --n-train-seeds 400 --path-reward-coef 0.5 --path-reward-factor 1
--path-reward-decay 1.0`. 4 seeds. (Path reward is constant — env reading_shaping disabled — so there
is no anneal schedule confound over the longer run.) Long run (~80 rounds); local desktop GPU handles
multi-hour. Read the full length-2 trajectory (rounds 20–79), not just last-5, to judge plateau vs
continued climb, and evaluate the §"acts confidently" criteria at the settled level.

## Standing bar — tag: `LADDER-EXIT`

Either outcome is a rung-4 decision point: (a) swap_follow climbs **clearly past ~0.15** (multi-seed,
`swapped ≈ 0`) — the first genuine break of the 2-step wall; or (b) swap_follow plateaus but the agent
meets the §acts-confidently bar — an **acceptable** competent-agent ceiling (residual = genuine
composition difficulty), which legitimately closes the reward/length axis and redirects the next lever to
the composition mechanism. A plateau that FAILS the confidence bar is an unacceptable ceiling → upstream
reader/curriculum fix.

## Follow-ups / variants (separate experiments)

- **n1600 generalization long-run** — same config at `--n-train-seeds 1600`, run long enough for matched
  per-recipe exposure (~120 rounds), to test whether broad coverage cracks the conditional via
  rule-learning. Kept separate because at 80 rounds n1600 is under-budget (~2–3×/recipe) and would
  confound this clean n400 ceiling read. (Direct 30-round reference: [[0062-rtfm-coverage-confirm]].)
- If (b) above (acceptable ceiling) lands: a **backward curriculum** (practice step-2 from
  step-1-done states) or a progress-state actor for the composition step — needs code → reviewer-gate.

## Links

[[0059-rtfm-floor-confirm]] · [[0061-rtfm-backloading-confirm]] · [[0051-rtfm-flat-vr-optimization]] · [[0057-rtfm-path-reward]] · [[0062-rtfm-coverage-confirm]] · [[mentored-learning-loop]]
