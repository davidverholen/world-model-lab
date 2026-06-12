---
status: draft
owner: human
scope: local
sources: [arxiv:2310.15017, arxiv:2205.07802]
verified: true
last_reviewed: 2026-06-12
---

# 0012: UTD scaling + targeted resets — literature-grounded retention arms (planned)

## Hypothesis

Informed by [[qiao-model-primacy-2023]] (ingested first — literature-first rule,
hardened): at least one of three arms clears the standing bar (no eval >20pp below
running best after recovery; mean best-eval ≥55% over 3 seeds) on DoorKey-6x6 at
the standard protocol:

- **(a) qiao**: shrink-perturb α=0.8 of the *next-latent predictor only*, every
  round — their faithful DreamerV2 world-model-reset recipe. Their theory predicts
  weak/negative effect at our low model-UTD; running it tests that prediction
  in-setting.
- **(b) utd**: no resets, updates-per-round 1500→6000 — the under-training
  hypothesis their Fig-3 raises (agent-UTD scaling *helped* MBPO; our exp-0010/0011
  "instability" may partly be an under-trained value/model at each round).
  Watch for the opposite failure (primacy amplification, cf. 0011 hr s1 60→20).
- **(c) surgical**: hard reinit of reward+value heads ONLY (dynamics untouched —
  the exp-0011 mis-mapping corrected), once at round 3, post-reset budget 6000.
  Literature-uncovered for our setting.

## Setup

Standard 0009–0011 protocol (DoorKey-6x6, 7 rounds, 20k ignition round 0 ≥5 events,
6×15k ε=0.3-MPC rounds, success-frac 0.25, 3 seeds/arm). Batch 1 (6-wide):
`--reset qiao` ×3, `--reset surgical --reset-round 3 --post-reset-updates 6000` ×3.
Batch 2 (3-wide): `--updates-per-round 6000` ×3. Comparators: ctrl 20/60/40
(exps 0009/0010), PPO ~37% @ 110k (exp 0008).

## Result

_pending_

## Lesson

_pending_

## Links

[[qiao-model-primacy-2023]] · [[0011-nikishin-resets]] · [[retention]] ·
[[nikishin-primacy-2022]] · [[agent-architecture]]
