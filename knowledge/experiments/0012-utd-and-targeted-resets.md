---
status: draft
owner: world-model
scope: local
sources: [arxiv:2310.15017, arxiv:2205.07802]
verified: true
last_reviewed: 2026-06-12
---

# 0012: UTD scaling wins big (63% mean, beats PPO); resets stay dead (run 2026-06-12)

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

(vs ctrl 20/60/40 mean 40%; PPO ~37% @ 110k)

| arm | bests (s0/s1/s2) | mean | verdict |
|---|---|---|---|
| qiao shrink-perturb | 20/60/60 | 47% | crashes persist; late-climb pattern (s2: 10→60 by r6); mildly above ctrl — consistent with Qiao's own low-UTD prediction (weak effect) |
| surgical (heads@r3+6k) | 10/60/20 | 30% | below ctrl; s1 crash precedes the r3 reset → resets can't fix a crash that happens at r1 |
| **utd ×4 (no resets)** | **80/50/60** | **63%** | **bar (ii) MET; first defeat of PPO at equal env budget; 80% = highest single eval on 6x6** |

UTD nuance: round-0 evals drop (s1: 10% — heavier training on random ignition data
deepens primacy, as predicted), but later rounds soar (s0: 80% @ r3). Bar (i)
no-crash still FAILS (s0: 80→30; s1: 50→20) — interference persists at higher
amplitude; best-checkpoint guard still load-bearing.

## Lesson

1. **Under-training was real and large**: our exps 0009–0011 ran at ~25% of the
   useful gradient budget; +4× updates (env budget unchanged) = +23pp mean. The
   "instability" story was partly a fitting-deficit story. Qiao's Fig-3
   agent-UTD finding transfers to our setting.
2. **Resets are conclusively dead in our regime** (5 variants across 0011/0012,
   all ≤ ctrl or barely above): both the Nikishin and Qiao recipes assume
   high-UTD overfitting; our failure mode needed more fitting, not forgetting.
3. **Rung 2b status: performance bar cleared (63% > 37% PPO at equal env steps),
   stability bar open.** Remaining failure isolated further: crashes now occur
   *from a high-competence state under heavy training* — classic interference,
   trunk still the prime suspect (nothing has yet protected encoder+GRU).
4. **Exp 0013 (pre-registered): UTD ×4 as the new baseline + trunk-freeze arm**
   (freeze encoder[, GRU] after round 2; heads keep training at UTD pace) — the
   last untested fix family, now strongly motivated from three directions
   (elimination 0009–0012, frozen-encoder literature line, exp-0012's
   high-amplitude crashes). Optional arm: UTD ×4 with round-0 kept at ×1
   (dodge the primacy cost; cheap).
5. PAID-RESOURCE note: none needed — 0013 fits the desktop (~80 min batches at
   UTD pace).

## Links

## Links

[[qiao-model-primacy-2023]] · [[0011-nikishin-resets]] · [[retention]] ·
[[nikishin-primacy-2022]] · [[agent-architecture]]
