---
status: draft
owner: world-model
scope: local
verified: false
last_reviewed: 2026-06-16
---

# 0052: decoupled validated-reading reward HEAD — grounding incentive directly on the actor

**Status: BUILT + reviewer-passed (SHIP), CPU-smoked; GPU dispatch pending (waiting for
[[0051-rtfm-flat-vr-optimization]] coef-0.5 to free the 8 GB GPU).** Implements the on-deck lever
named in exp0051: apply the maintainer's incentive-structure insight directly to the acting policy.

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
