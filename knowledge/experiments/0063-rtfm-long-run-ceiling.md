---
status: draft
owner: world-model
scope: local
verified: false
last_reviewed: 2026-06-17
---

# 0063: does a LONGER run break the ~0.10 ceiling? (the trend was still rising at round 30)

**Status: PRE-REGISTERED (documented before dispatch).** Directly tests the claim-audit finding: the
confirmed reward lever's `swap_follow` was **still rising at round 30**, not plateaued — so the
"~0.10 ceiling" may be a *training-length artifact* at this reward, not a real wall.

## Why (the audit finding)

The 2026-06-17 claim audit showed the uniform-0.5 / floor-0.5 path-reward runs do NOT plateau by round
30 — they keep gently climbing:
- [[0059-rtfm-floor-confirm]] (floor 0.5, 4 seeds): per-round swap_follow ~0.065 (r20–23) → ~0.105
  (r26–29), **r29 ≈ 0.13**.
- [[0061-rtfm-backloading-confirm]] f1 (uniform 0.5, 2 seeds): **r29 ≈ 0.15**.

Crucial contrast: [[0051-rtfm-flat-vr-optimization]]'s budget probe WAS flat ~0.11 over 40 rounds — but
it had **no path reward**. So the path reward appears to drive a continued climb → **a longer run is a
live, untested lever** (this corrects the earlier "longer training won't help", which was based on the
no-path-reward exp0051). exp0063 just runs the best-and-still-rising config much longer.

## Hypothesis / fork

Take the **confirmed best, still-rising config** — uniform-0.5 path reward
(`--path-reward-coef 0.5 --path-reward-factor 1 --path-reward-decay 1.0`), n400 (clean: change ONLY
the training length vs exp0061f1) — and run **20 curriculum + 60 length-2 = 80 rounds** (vs the usual
10+20=30). Watch swap_follow / swap_follow_s1 / conditional over the 60 length-2 rounds.

- **swap_follow keeps climbing past ~0.15** → the ~0.10 "ceiling" was a training-length artifact; the
  path reward genuinely cracks the 2-step wall given enough rounds. (Watch swap_follow specifically,
  NOT `correct` — exp0051 showed `correct` can climb via experience-based completion that swap_follow
  rightly discounts.)
- **swap_follow plateaus (~0.10–0.13)** → the ceiling is real at n400/this-reward → the lever is
  elsewhere: more coverage (n1600 × long — the follow-up if this plateaus) or a backward curriculum /
  architecture for the conditional.
- **Divergence/collapse over the long run** → instability at 80 rounds (watch imagined_return, recon).

## Setup

`scripts/dispatch_rtfm.sh exp0063 4 --rounds 80 --curriculum-rounds 20 --length 2 --manual-aux-coef
1.0 --validated-reading-coef 0.3 --n-train-seeds 400 --path-reward-coef 0.5 --path-reward-factor 1
--path-reward-decay 1.0`. 4 seeds. (Path reward is constant — env reading_shaping disabled — so there
is no anneal schedule confound over the longer run.) Long run (~80 rounds); local desktop GPU handles
multi-hour. Read the full length-2 trajectory (rounds 20–79), not just last-5, to judge plateau vs
continued climb.

## Standing bar (success) — tag: `LADDER-EXIT`

swap_follow on held-out length-2 climbs **clearly past the ~0.10 ceiling** (≳0.15, multi-seed) with
`swapped ≈ 0` — the first genuine break of the rung-4 ceiling, showing the path reward + enough
training cracks the 2-step composition wall. A plateau cleanly redirects to coverage/exposure.

## Links

[[0059-rtfm-floor-confirm]] · [[0061-rtfm-backloading-confirm]] · [[0051-rtfm-flat-vr-optimization]] · [[0057-rtfm-path-reward]] · [[0062-rtfm-coverage-confirm]] · [[mentored-learning-loop]]
