---
status: draft
owner: world-model
scope: local
verified: true
last_reviewed: 2026-06-14
---

# 0032: Does the manual-conditioned WM learn to GROUND? (rung-4, pre-registered)

## Hypothesis

The first reading-to-learn-dynamics result. The rung-4 agent ([[rung4-manual-conditioned-agent]])
conditions the RSSM dynamics on the per-episode manual (frozen MiniLM token cross-attention folded
into the deterministic state h). Trained on crafter-rtfm **r1 randomized recipes** (held-out splits),
does it **learn to ground** — read the recipe each episode and perform it — rather than bake a fixed
policy? The unfakeable test is **swap-following on HELD-OUT manuals**: given a *swapped* recipe, a
grounded agent performs the *swapped* gesture; a baked one ignores it.

Predict: over rounds, on the held-out eval split, **`grounding = score(correct) − score(none)`
rises above 0** (reading helps) AND **`swap_follow` rises** (it follows the swapped manual = grounds,
not memorizes). Null (grounding ≈ 0 or swap_follow ≈ 0 while correct rises) → it's solving without
reading (baking / a no-text shortcut) → debug the conditioning or the env's reading-necessity.
Reference: crafter-rtfm's own thin policy-stub reached correct=1.00/none=0.00/swap_follow=1.00
([[HO-0004]]); ours is the harder WORLD-MODEL version (plan through the manual-conditioned dynamics).

## Setup

`world_model.train_rtfm` on the desktop (rung-4 deployed: crafter-rtfm at the sibling path, `rtfm`
extra). r1 recipes, gesture length 3, max_steps 48. Train on `splits.split_seeds("r1","train")`
(content-disjoint), eval on the held-out `"eval"` split each round via crafter-rtfm's harness
(run_episode in correct/none/swapped + swap_follow_rate). 2 seeds. Config: 20 rounds, 60 episodes/
round, 400 WM + 400 AC updates, seq-batch 16, window 12, burn-in 4, horizon 10.

Command per seed: `python -m world_model.train_rtfm --rounds 20 --episodes-per-round 60
--updates-per-round 400 --ac-updates-per-round 400 --seq-batch 16 --window 12 --burn-in 4
--horizon 10 --n-train-seeds 400 --n-eval-seeds 40 --max-steps 48 --length 3 --save runs/rtfm.pt
--seed {0,1}`

## Result

**`correct` never lifted off zero at length-3 — but the run was a productive debugging arc that
calibrated the rung-4 training, ending in the length-1 diagnostic ([[0033-rtfm-length1-grounding]]).**
The per-round diagnostics (added mid-run) exposed and fixed, in order, a sequence of classic
training failures — each a real insight, each a known rung-3 lesson rediscovered for the new env:

1. **Value inflation** (imagined_return 1.4→2.6→3.0, actor_loss→0, correct=0). The first version
   omitted critic-on-replay. → added **repval=0.3** (exp 0021): imagined_return bounded again.
2. **Sparse-reward actor starvation** (calibrated value but actor no traction — r1 reward ~2%,
   uniform sampling). → added **success-oversampling** (exp 0009) to the imagination AC.
3. **Reward-head hallucination** (success-oversampling the *WM* training biased the reward head:
   imagined_return 5–9 while correct=0). → **split it**: WM/reward head trains UNBIASED
   (success_frac=0), only the imagination AC oversamples. Reward head re-calibrated to ~1.1–1.8.
4. With the full calibrated recipe (bounded reward 0025 + patch tokens 0024 + repval 0021 +
   AC-only success-oversampling), the machinery is healthy (recon 0.17, value bounded) but
   **`correct` still 0 at round 6** — the 3-action exact-sequence credit assignment doesn't resolve.

To separate "can't ground" from "can't do a 3-step sequence," dropped to length-1 → exp 0033.

## Lesson

**Standing up a manual-conditioned WM training pipeline reproduced the entire rung-3 calibration
arc (value inflation, sparse-reward, reward-head bias) — all fixable with the known fixes — but the
calibrated agent still didn't solve length-3.** The diagnostics (per-round recon/kl/actor/critic/
imagined_return, added after the first blind rounds) were decisive: they turned "correct=0 mystery"
into specific, named failure modes in 2 rounds each. **Process lesson: print the internal losses
from the start** when standing up a new training loop — blind eval-only logging cost the first
rounds. The length-3 non-convergence is *not* the headline; the headline is what the length-1
diagnostic ([[0033-rtfm-length1-grounding]]) then revealed about grounding-vs-baking.

## Links

[[0033-rtfm-length1-grounding]] · [[rung4-manual-conditioned-agent]] · [[grounding-env-spec]] · [[dreamerv3-2023]]

## Links

[[rung4-manual-conditioned-agent]] · [[grounding-env-spec]] · [[0031-semantic-foundation-probe]] · [[no-hardcoded-env]]
