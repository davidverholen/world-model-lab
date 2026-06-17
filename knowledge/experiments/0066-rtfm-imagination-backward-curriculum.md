---
status: draft
owner: world-model
scope: local
verified: false
last_reviewed: 2026-06-17
---

# 0066: imagination backward curriculum — does seeding the actor from prefix-done states crack the step-2 conditional (and shrink seed variance)?

**Status: PRE-REGISTERED — code committed (32a2fe0), reviewer-gated; awaiting the exp0065 GPU to
dispatch.** First test of the §3a lever. Default-off `--backward-curriculum`; an opus reviewer caught
and fixed a load-bearing off-by-one (imagination-start offset is `burn_in-2`, not `burn_in-1`) before any
run, so the curriculum seeds the true step-1-done frontier.

## Why

[[0065-rtfm-per-step-plateau-large-budget]] localized the 2-step wall to the **conditional
P(step-2|step-1)** — a high-variance **exploration/credit lottery** for the later step (deep states
visited ~p^(k-1); some seeds stumble into step-2, most don't). The reward axis is exhausted
([[0064-rtfm-path-reward-coef-sweep]] null; back-loading refuted). The lever
([[hierarchical-imagination-agent]] §3a, lit-grounded by [[romi-2021]]/[[florensa-2017]]/[[lexa-2021]]):
**manufacture the step-2-completion experience** by seeding the actor-critic's imagination START
distribution from REAL buffered step-1-done latents (oversample `bc-frac` of starts), so every seed gets
the dense tail-credit that only the lucky seeds (s3) found by exploration.

## Hypothesis / fork

ON vs OFF at normal length (length-2, 30 rounds = 10 curriculum + 20 length-2), multi-seed, proven config.
Read `swap_follow` (full 2-step), `swap_follow_s1` (step-1), the conditional, AND `bc_realized_frac`
(buffer-supply telemetry — if ≪ `bc-frac`, the prefix-done pool is starved, the §3a supply risk).

- **ON lifts the conditional above OFF (multi-seed) AND shrinks seed variance** (the laggard seeds catch
  up — they get the experience s3 lucked into) → the backward curriculum cracks the composition wall; the
  first mechanism beyond reward to move step-2 → scale toward length-3 (the bootstrap).
- **ON ≈ OFF** → no help at length-2. First check `bc_realized_frac`: if starved, it's a buffer-supply
  problem (need more/active seeding), not a refutation of the mechanism. If supply is healthy and still
  flat → the imagination seeding isn't sufficient at depth-2 → reconsider (frontier definition,
  conditioning, or the WM-fidelity-at-frontier caveat).
- **ON < OFF / unstable** → imagination seeding HURTS — likely the §3a co-evolution caveat biting (WM
  infidelity one step past the frontier → training the actor on poorly-modeled states). Watch
  imagined_return / recon for divergence; would argue for a shorter horizon or a stricter frontier gate.

**Step-1 should be ~unchanged** (the curriculum targets step-2 specifically); a clean win is conditional
up, step-1 flat.

## Setup

Dispatch after exp0065 frees the GPU/RAM. Proven config throughout: `--manual-aux-coef 1.0
--validated-reading-coef 0.3 --n-train-seeds 400 --path-reward-coef 0.5 --path-reward-factor 1
--path-reward-decay 1.0`, `--rounds 30 --curriculum-rounds 10 --length 2`.

```
scripts/dispatch_rtfm.sh exp0066off 4 <proven> --no-backward-curriculum                  # OFF baseline
scripts/dispatch_rtfm.sh exp0066on  4 <proven> --backward-curriculum --bc-frac 0.5        # ON (curriculum)
```

8 procs (4+4), in parallel ([[rtfm-rollout-perf]]; RAM-gated — verify headroom first). **Split / exploration**
(the maintainer's "find something by accident", if capacity allows): extra ON arms at `--bc-frac 0.25`
and `0.75` for a dose-response. Read full trajectories + last-5; compare ON vs OFF on conditional and
seed-variance; watch `BC prefix_done_start_frac` each round.

## Standing bar — tag: `LADDER-EXIT`

ON's held-out conditional clearly exceeds OFF (multi-seed, step-1 unchanged, seed-variance reduced) —
the first non-reward mechanism to move the 2-step composition, and the substrate for length-3+. A clean
null (with healthy supply) or a HURT redirects to the frontier/conditioning/WM-fidelity sub-questions.

## Links

[[0065-rtfm-per-step-plateau-large-budget]] · [[hierarchical-imagination-agent]] · [[romi-2021]] · [[florensa-2017]] · [[lexa-2021]] · [[0064-rtfm-path-reward-coef-sweep]] · [[rtfm-rollout-perf]]
