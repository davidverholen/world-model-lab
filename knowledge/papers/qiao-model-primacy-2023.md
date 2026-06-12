---
status: draft
owner: agent
scope: local
sources: [arxiv:2310.15017]
verified: true
last_reviewed: 2026-06-12
---

# Mind the Model, Not the Agent: The Primacy Bias in Model-Based RL (Qiao, Lyu & Li, 2023)

**Lab:** Tsinghua SIGS · **Found:** 2026-06-12 via the hitting-a-wall protocol
(S2 citation walk of Nikishin) — *after* we had wrongly claimed frontier status.
**Read state:** skimmed (method+results via full text).

## One-paragraph summary

Splits MBRL primacy bias into **agent** primacy bias (Q/policy overfit early data)
and **model** primacy bias (world model overfits the early policy's state
distribution). Evidence: scaling agent-UTD *helps* MBPO (until ~80); scaling
model-UTD ×10 collapses it; Nikishin-style agent resets that help SAC *harm* MBPO.
Remedy: **world model resetting** — MBPO: reset last 2 hidden layers of all
ensemble members every 2e4 steps; DreamerV2: soft-reset ONLY the transition
predictor via φ←(1−α)φ+α·φ_random, α=0.8. Agent never reset, buffers kept.
18–36% gains — **but only at high model-UTD; at default/low UTD resets degrade.**

## Relevance to our experiments (reconciliation with exps 0009–0011)

- Their "agent resets harm MBPO" = our exp 0011 result, independently confirmed.
- Their UTD-dependence inverts our framing: at our model-UTD (~0.1, far below
  their default) their theory predicts resets fail (they did) AND suggests we may
  be **under-training, not over-fitting** — their Fig-3-style UTD scaling is the
  diagnostic we haven't run.
- Their DreamerV2 recipe (shrink-perturb α=0.8 of the transition predictor only)
  is the faithful "world-model reset" for our architecture — opposite target to
  our surgical-heads sketch; exp 0012 should carry BOTH as arms.
- **What they don't cover** (the surviving, narrowed frontier): sparse rewards
  (beyond Atari), belief-state/POMDP filtering, online MPC planning, fixed-budget
  flywheel collection — our exact regime.

## Limitations / critiques

Transformer resets flagged as open; HalfCheetah shows task sensitivity;
non-monotonic ensemble effects unexplained; no sparse-reward analysis.

## Links

[[nikishin-primacy-2022]] · [[retention]] · [[0011-nikishin-resets]] ·
[[dreamerv3-2023]] · [[agent-architecture]]
