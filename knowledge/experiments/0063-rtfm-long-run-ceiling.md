---
status: draft
owner: world-model
scope: local
verified: false
last_reviewed: 2026-06-17
---

# 0063: does a LONGER run break the ~0.10 ceiling? (the trend was still rising at round 30)

**Status: PRE-REGISTERED — NOT YET RUN (queued for the next session).** Was dispatched once in error
then stopped immediately (round 0, no data) for a clean cut after exp0062; the maintainer is handing
over to a fresh session to run this. Directly tests the claim-audit finding: the confirmed reward
lever's `swap_follow` was **still rising at round 30**, not plateaued — so the "~0.10 ceiling" may be
a *training-length artifact* at this reward, not a real wall. **Next session: dispatch the command in
§Setup.**

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
(`--path-reward-coef 0.5 --path-reward-factor 1 --path-reward-decay 1.0`) — at **n-train-seeds 1600**
(high coverage = GENERALIZATION focus: the agent must learn the chaining *rule* across many distinct
recipes rather than over-fit a small set; eval is held-out).

**Why n1600 + a LONG run (not n400):** n1600 is a *harder* task than n400 — and [[0062-rtfm-coverage-confirm]]
showed exactly that, n1600 < n400 at 30 rounds, because at fixed budget n1600 gets only ~1.1×/recipe
vs n400's ~4.5×. That low score is **the harder task being under-budgeted, not coverage failing.** This
long run is the fix: 80 rounds → n1600 ~3×/recipe, giving the harder generalization task enough practice
to see if it learns the rule and breaks the ceiling. **Honest caveat:** even 80 rounds (~3×/recipe) is
still *below* n400@30r's 4.5×/recipe, so a plateau here would NOT cleanly mean "coverage can't help" —
it could still be under-budget (→ even more rounds). [[0062-rtfm-coverage-confirm]] (n1600, 30 rounds)
is the direct 30-round reference. Watch swap_follow / swap_follow_s1 / conditional over the 60 length-2
rounds.

- **swap_follow keeps climbing past ~0.15** → the ~0.10 "ceiling" was a training-length artifact; the
  path reward + coverage genuinely crack the 2-step wall given enough rounds. (Watch swap_follow
  specifically, NOT `correct` — exp0051 showed `correct` can climb via experience-based completion that
  swap_follow rightly discounts.)
- **swap_follow plateaus (~0.10–0.13)** → ambiguous (per the caveat above): either the ceiling is real,
  OR n1600 is still under-budget at ~3×/recipe → the disambiguating follow-up is a clean equal-exposure
  coverage test (n400 vs n1600 at matched per-recipe practice, ~120 rounds for n1600) and/or a backward
  curriculum / architecture for the conditional. (A clean ceiling-break test that avoids this ambiguity
  would instead extend the proven-best **n400** config long — kept as the alternative if the n1600
  generalization run is inconclusive.)
- **Divergence/collapse over the long run** → instability at 80 rounds (watch imagined_return, recon).

## Setup

`scripts/dispatch_rtfm.sh exp0063 4 --rounds 80 --curriculum-rounds 20 --length 2 --manual-aux-coef
1.0 --validated-reading-coef 0.3 --n-train-seeds 1600 --path-reward-coef 0.5 --path-reward-factor 1
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
