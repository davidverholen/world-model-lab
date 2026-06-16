---
status: draft
owner: world-model
scope: local
verified: true
last_reviewed: 2026-06-16
---

# 0052: decoupled validated-reading reward HEAD — grounding incentive directly on the actor

**Status: CONCLUDED — NEGATIVE.** The decoupled VR head (λ=1.0) did not beat the ceiling; it
**collapsed the policy** to a degenerate manual-blind fixed point (below the folded-VR baseline).
Implements the on-deck lever named in exp0051: apply the maintainer's incentive-structure insight
directly to the acting policy. Result + diagnosis below.

## Result — NEGATIVE: policy collapse (2 seeds, length-2)

`runs/exp0052` (2 seeds, `--vr-head-coef 1.0`, `--validated-reading-coef 0`, else the exp0051
baseline config). The length-2 eval **froze for 13 straight rounds** (10–22) at a degenerate point:

| length-2, last 6 rounds | exp0052 (decoupled VR head) | exp0051 baseline (folded VR) |
|---|---|---|
| swap_follow | **0.00–0.02 (frozen)** | ~0.10 (varies, peaks 0.15) |
| correct / none / swapped | **all 0.02 (manual-blind)** | differentiated |
| tutorial events / round | **0–1 (dead flywheel)** | 2–7 (turning) |
| imagined_return (len-2) | **~15–25 (inflated)** | ~8–12 |

`correct = none = swapped` exactly + frozen across rounds = a **manual-blind degenerate policy**: the
greedy agent does the same first-2 actions regardless of the displayed manual; swap_follow 0.02 is
just the chance that fixed pair matches. The flywheel is dead (≈0 events) — strictly worse than the
folded-VR baseline it was meant to beat.

## Diagnosis — a solved problem re-opened on a new channel

The imagined return inflated to ~15–25 (vs ~8–12 baseline): the actor learned to chase states where
the **VR head over-predicts**. The VR head is trained only on *real* collected transitions, so it gets
**exploited on out-of-distribution imagined states** — the exact failure of [[0045-rtfm-oracle-probe]]
(the reward head over-rated OOD action sequences), which [[0047-rtfm-conservative-reward]]'s CROP/CQL
push-down was built to fix. We put **no such conservative guard on the VR head**, so a dense,
directly-actor-optimized intrinsic head reward-hacks itself and the policy collapses. Decoupling is not
proven wrong — an actor-optimized dense intrinsic head just needs the same OOD guard the task reward
head has. [[0053-rtfm-imagination-fidelity]] corroborates: the WM *dynamics* imagine faithfully for
both baseline and collapsed models, so the collapse is **downstream of the world model** (policy /
reward-readout), consistent with this diagnosis.

## Lesson

(1) **Pushing the grounding incentive HARDER keeps failing** — the lever ladder (objective→folded VR
0.10, coef-saturate, 2× compute, hierarchy ~0, decoupled head collapse) says the bottleneck is no
longer incentive-strength but **downstream execution / reward-readout**. (2) A dense intrinsic reward
optimized directly by the actor in imagination needs an **OOD conservatism guard** or it reward-hacks
(same as the extrinsic reward head). (3) The robust best on the rung remains the **folded VR ~0.10**.
→ pivot to localizing the wall ([[0053-rtfm-imagination-fidelity]]) rather than another reward variant.

## Original pre-registration (for the record)

Implements the on-deck lever named in exp0051: apply the maintainer's incentive-structure insight
directly to the acting policy.

## Why (the maintainer's insight)

The hierarchy thread ([[0050-rtfm-hierarchy]]) failed partly because the WORKER — the level that
actually acts — had **no grounding incentive**; only the WM's blended reward head carried it. The
insight: *every acting level needs the grounding incentive directly*, not laundered through a shared
reward head. In the flat agent ([[0048-rtfm-validated-reading]]) VR was **folded into the collection
task reward** `r_read` (coef·VR added in), so the actor feels VR only through the WM's single,
blurred reward head — and the coef lever saturated at the ~0.10 ceiling (exp0048 0.1/0.3, exp0051
budget-2× and coef-0.5 all ≈0.10–0.115). The ceiling is **not compute-bound** (exp0051 budget probe)
→ the lever must be a *denser / less-diluted* grounding signal, not more training.

## Hypothesis

Give validated-reading its **own dedicated reward head** and put it **directly on the acting
policy**: store RAW (unscaled) VR in a separate buffer channel (task reward stays clean), train a
dedicated two-hot VR head as a **detached readout** (WM representation byte-identical to the validated
baseline), and have the imagination actor optimise `task_reward + λ·VR_head(belief, action)`
(λ = `--vr-head-coef`). **Predict:** an undiluted grounding incentive on the actor pushes length-2
`swap_follow` **above the ~0.10 ceiling** (target ≳0.15 multi-seed), with `swapped ≈ 0` (anti-baking
preserved — VR is reality-judged at collection, so it cannot be faked).

## Counter-outcomes (named)

- **Null / ≈0.10** (decoupling doesn't beat the folded path): the dilution was not the bottleneck —
  the ~0.10 ceiling is a deeper representational/credit-assignment limit on the 2-step gesture →
  relocate (back to the multi-step execution wall, not the incentive channel).
- **VR-head collapse** (`vr_loss` flat, head predicts ~0 everywhere): the two-hot binning can't
  resolve the small ≥0 VR magnitudes, or the detached readout under-trains → scale VR before storing,
  or widen the actor coef λ. (Reviewer judged collapse structurally unlikely — two-hot interpolates
  small positives between the bins straddling 0.)
- **Actor exploits VR_head** (imagined VR runs away, `imagined_return` blows up, task suffers): the
  λ·VR term over-weights → lower λ; VR head is detached so it can't corrupt the WM, but the actor can
  still over-chase an over-predicted VR (the dark-room failure, one level up). Watch `imagined_return`.

## Setup (BUILT)

- **Buffer (`training/replay.py`):** new `Transition.vr` field (default 0.0 → all non-VR callers
  byte-for-byte unchanged) + parallel `ReplayBuffer.vr` array + `vr` key in `_window_batch`. The `vr`
  channel never enters `compute_returns`/`success_starts` (task-reward-only) — verified by reviewer.
- **Collection (`collect_rtfm`):** compute RAW VR once; folded path (`--validated-reading-coef>0`)
  still adds coef·VR into `r_read`; decoupled path (`store_vr`, set when `--vr-head-coef>0`) keeps
  `r_read` clean and stores RAW VR in the channel.
- **WM train (`wm_train_rtfm`):** train `vr_head` (a `TwoHotRewardHead`) with
  `twohot_loss(belief.detach(), action, vr)` — pure readout, added UNWEIGHTED to the WM loss (detach
  means its CE magnitude can't perturb the representation); head lives in the WM optimizer. Returns a
  new `vr_loss` stat (5-tuple).
- **Actor (`imagine_ac_rtfm`):** imagined per-step reward = `rew(bel,a) + λ·vr_head(bel,a)`.
- **Flag:** `--vr-head-coef λ` (0 = off/unchanged). Built only when λ>0.
- **Test:** `test_vr_channel_roundtrip` (replay channel is separate from reward, defaults to 0).
- **Reviewer (opus):** SHIP — decoupling correct, detach preserves WM, two-hot represents small VR,
  backward-compatible, no train/eval leakage.

## Dispatch plan

2 seeds first (cheap probe), `--vr-head-coef 1.0` as the initial λ (RAW VR is small, so a larger λ
than the folded 0.1–0.3 is expected — the head predicts unscaled VR). Same flat rung-4 config as
exp0048/0051 (curriculum, one_shot, length-2, `--validated-reading-coef 0` so VR flows ONLY through
the decoupled head). If it clears ~0.10, run a 4-seed control + a small λ sweep.

## Standing bar (success) — tag: `LADDER-EXIT`

Length-2 `swap_follow` **above the ~0.10 ceiling** (target ≳0.15, multi-seed), `swapped ≈ 0`,
`correct > none`. That would show the grounding ceiling was a *dilution* limit (fixable by putting the
incentive directly on the actor) rather than a representational wall — the next brick of the
[[mentored-learning-loop]].

## Links

[[0048-rtfm-validated-reading]] · [[0049-rtfm-sustained-vr]] · [[0050-rtfm-hierarchy]] · [[0051-rtfm-flat-vr-optimization]] · [[validated-reading-reward]] · [[mentored-learning-loop]] · [[vime-2016]] · [[marino-hypothesis-2020]]
