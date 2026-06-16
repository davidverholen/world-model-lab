---
status: draft
owner: world-model
scope: local
verified: true
last_reviewed: 2026-06-16
---

# 0055: ensemble-uncertainty (MOPO) pessimism — fix the reward over-optimism WITHOUT over-suppressing

**Status: CONCLUDED — counter-outcome, but PIVOTAL.** Pessimism did NOT break the exact-swap_follow
ceiling (λ∈{0.5,1,2,5} → 0.054/0.074/0.076/0.084, all ≈ or below the ~0.10 baseline; reward head
still over-optimistic, c50 mean reward-gap +0.58 vs exp0054 +0.67). BUT the new per-step metric
([[0054-rtfm-reward-gap]]→swap_follow_s1) revealed **why the whole reward thread (exp0045–0055) never
moved the needle: the wall is NOT reading or reward — it is 2-STEP COMPOSITION.** Reframe below.

## RESULT — the reframe: reading works (~0.4), chaining fails (~0.08)

Backfilled per-step swap-following on the swapped manuals (step-1 match vs full-chain exact), 60 eval
seeds, single seed/arm:

| λ | exact (full 2-step) | **step-1** | ratio |
|---|---|---|---|
| 0.5 | 0.050 | **0.450** | 9.0× |
| 1.0 | 0.067 | **0.417** | 6.2× |
| 2.0 | 0.083 | **0.350** | 4.2× |
| 5.0 | 0.100 | **0.350** | 3.5× |

The agent follows the **first** displayed instruction **35–45%** of the time (≈7× the ~0.06
single-action chance) — *genuine, working reading-to-action on held-out swapped manuals.* It completes
the **whole 2-step chain only 5–10%.** The wall is **step-2-given-step-1**: a depth-2 composition /
sequencing failure (the maintainer's hypothesis), NOT a grounding or reward-readout failure.

**This relabels the entire rung-4 "~0.10 grounding ceiling."** It was an artifact of an all-or-nothing
metric: real grounding is ~0.4 and working; the ceiling is **2-step composition** (the
[[0043-rtfm-execution-wall]] multi-step credit-assignment wall, seen through a grounding lens). The
reward machinery (conservatism [[0047-rtfm-conservative-reward]], validated-reading
[[0048-rtfm-validated-reading]], pessimism here) was sanding a surface that was never the bottleneck.

## Lesson

(1) **Measure the right thing.** Five experiments chased reward over-optimism because the exact-match
swap_follow hid that reading already works at depth-1; one per-step metric reframed the problem.
(2) **The ensemble-pessimism build is sound** (reviewer SHIP; epistemic gap real) and stays available,
but reward honesty is not the binding constraint at length-2. (3) The grounding result is actually
ENCOURAGING — read→act is ~0.4 on held-out swapped manuals. → pivot to **2-step composition** levers
([[0056-rtfm-coverage-sweep]] + curriculum/planning), watching swap_follow_s1 vs exact.

## Original pre-registration (retained)

Direct response to [[0054-rtfm-reward-gap]]: the task reward head over-predicts (+0.67/+0.79 over 8 imagined steps) atop a
faithful WM, so the actor chases phantom reward. [[0047-rtfm-conservative-reward]] already tried the
blunt fix (blanket CROP/CQL push-down) and it monotonically over-suppressed (crushed the true gesture
too). This is the *targeted* version.

## Hypothesis

A **deep ensemble of K reward heads** (bootstrapped → they agree in-distribution, disagree OOD) gives a
per-state epistemic-uncertainty signal `std_k(rew)`. At imagination time the actor optimises
`mean_k(rew) − λ·std_k(rew)` (MOPO/MOReL pessimism). **Predict:** this cancels the phantom reward
exactly where it lives (OOD-uncertain regions, high std) while sparing the true gesture (seen in data,
std≈0) — so length-2 `swap_follow` lifts past the ~0.10 ceiling, unlike exp0047's blanket push-down.

Why it should NOT repeat exp0047: (1) **targeted** — penalises *disagreement*, not all OOD actions;
(2) **imagination-time on the actor's objective** — leaves the heads' learned reward values honest
(no value crushing). The reviewer confirmed the epistemic gap is real (in-dist std ≈0.001–0.007 vs OOD
≈0.12–0.20, ~30–100×) so the signal the method rides on exists.

## Counter-outcomes (named)

- **Null** — ensemble agrees everywhere (std≈0), λ does nothing → strengthen bootstrap diversity / more
  members.
- **Over-pessimism** — high λ subtracts so much it kills real reward too (inverted-U, like the VR coef
  and exp0047) → pick the sweet spot from the sweep.
- **Gap shrinks but swap_follow stays ~0.10** — the reward readout was NOT the binding constraint after
  all → the residual is pure credit-assignment / 2-step execution ([[0043-rtfm-execution-wall]]); pivot
  to planning/horizon, now well-justified.

## Setup (BUILT)

- `EnsembleRewardHead(K)` in `models/twohot.py` — K bootstrapped two-hot heads; `forward`=mean,
  `mean_std`=(mean,std), `twohot_loss`=mean over members with per-member Bernoulli(0.5) bootstrap.
  Drop-in (K=1 / λ=0 = byte-for-byte the single head).
- `imagine_ac_rtfm`: per-step imagined reward `mean − λ·std` when the ensemble + `λ>0`.
- Flags `--reward-ensemble-size K`, `--reward-uncertainty-coef λ`. Checkpoint persists
  `reward_ensemble_size`; `imagine_report.py` reloads the ensemble for the reward-gap probe.
- Test `test_ensemble_reward_head_shapes_and_epistemic_disagreement` (OOD std > in-dist std).
- **Note:** ensemble pessimism and the exp0047 `--reward-conservative-coef` are independent levers and
  could be combined, but exp0055 runs pessimism ALONE (cons-coef 0) to isolate it.

## Pre-registered sweep

K=5, bootstrap on, on the working ~0.10 baseline (folded VR 0.3, new save format).
**λ ∈ {0.5, 1, 2, 5} × 1 seed** (broad coverage first, the maintainer's "different values then control"
methodology). Read BOTH length-2 `swap_follow` (clears 0.10?) AND the `imagine_report.py` reward-gap
(does pessimism shrink the +0.7 toward 0, and at which λ does it overshoot to negative = over-pessimism?).
Winner → 4-seed control.

`scripts/dispatch_rtfm.sh exp0055cXX 1 --rounds 30 --curriculum-rounds 10 --length 2
--manual-aux-coef 1.0 --validated-reading-coef 0.3 --reward-ensemble-size 5 --reward-uncertainty-coef λ`

## Standing bar (success) — tag: `LADDER-EXIT`

Length-2 `swap_follow` clearly above the ~0.10 ceiling (multi-seed) with `swapped ≈ 0`, AND the
reward-gap shrunk toward 0 at that λ (mechanism confirmed). That would show the over-optimism was the
binding downstream constraint and that targeted pessimism is the fix exp0047's blanket form couldn't be.

## Links

[[0054-rtfm-reward-gap]] · [[0047-rtfm-conservative-reward]] · [[0045-rtfm-oracle-probe]] · [[0053-rtfm-imagination-fidelity]] · [[0043-rtfm-execution-wall]] · [[0048-rtfm-validated-reading]]
