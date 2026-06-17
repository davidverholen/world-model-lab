---
status: draft
owner: world-model
scope: local
verified: false
last_reviewed: 2026-06-17
---

# 0062: coverage confirm — is the step-2 conditional data/exposure-bound?

**Status: IN FLIGHT** (`runs/exp0062`, 4 seeds, dispatched — pre-registration written after dispatch;
process slip, flagged). Decides the lever for the step-2 *conditional*, the now-localized nut of the
2-step composition wall.

## Why this run (what the night established)

The reward-shape axis is exhausted and points elsewhere:
- [[0057-rtfm-path-reward]] / [[0059-rtfm-floor-confirm]]: a **uniform path reward (~0.5)** robustly lifts step-1
  (0.28→0.36) and the **conditional P(step-2|step-1)** (0.17→~0.30) across seeds — the confirmed reward
  lever — but only recovers exact swap_follow ~0.10 (0.36·0.30 ≈ 0.10).
- [[0058-rtfm-escalating-path-reward]] / [[0061-rtfm-backloading-confirm]]: **back-loading the reward REFUTED** — making step-2
  worth more *lowers* the conditional (0.33→0.20); the agent attempts step-1 more but follows through
  worse. So the conditional is **NOT step-2-reward-magnitude-limited.**
- [[0056-rtfm-coverage-sweep]] (1 seed): more distinct recipes (n1600) lifted the conditional to ~0.40.

Converging read: the conditional is **exposure / data-bound** — the model can't reliably *execute*
step-2 given step-1, and it improves with broader recipe exposure, not with more step-2 reward. This
run tests that robustly.

## Hypothesis / fork

Run n-train-seeds **1600** (4× recipes) at the **confirmed working reward** (uniform-0.5 path reward:
`--path-reward-coef 0.5 --path-reward-factor 1 --path-reward-decay 1.0`), 4 seeds. Compare against
[[0061-rtfm-backloading-confirm]] factor-1 (n400, same reward) — the only difference is coverage.

- **Conditional rises with coverage (n1600 > n400)** → the wall is **data/exposure-bound** (the
  maintainer's combinatorial hypothesis, now at multi-seed) → levers: more coverage + a **backward
  curriculum** (practice step-2 from step-1-done states; needs code → reviewer-gate).
- **Conditional flat** → not data-bound either → a deeper representational/binding limit at depth-2
  → revisit the architecture (e.g. progress-state in the actor, or a structured/compositional reader).

## Setup

`scripts/dispatch_rtfm.sh exp0062 4 --rounds 30 --curriculum-rounds 10 --length 2 --manual-aux-coef
1.0 --validated-reading-coef 0.3 --n-train-seeds 1600 --path-reward-coef 0.5 --path-reward-factor 1
--path-reward-decay 1.0`. Read last-5 length-2 step-1 / exact / conditional vs the n400 reference.

## Standing bar — diagnostic (tag: `EXTRA-RIGOR`)

Multi-seed: does coverage robustly lift the conditional above the n400 ~0.30? Either result picks the
next lever unambiguously (more-data/backward-curriculum vs architectural).

## Links

[[0056-rtfm-coverage-sweep]] · [[0057-rtfm-path-reward]] · [[0058-rtfm-escalating-path-reward]] · [[0061-rtfm-backloading-confirm]] · [[0055-rtfm-ensemble-pessimism]] · [[mentored-learning-loop]]
