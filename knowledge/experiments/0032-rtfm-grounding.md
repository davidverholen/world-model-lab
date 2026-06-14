---
status: draft
owner: human
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

_pending (dispatched)_

## Lesson

_pending_

## Links

[[rung4-manual-conditioned-agent]] · [[grounding-env-spec]] · [[0031-semantic-foundation-probe]] · [[no-hardcoded-env]]
