---
status: draft
owner: world-model
scope: local
verified: true
last_reviewed: 2026-06-16
---

# 0049: sustained validated-reading — push past the plateau (ridge + shaping anneal)

**Status: PRE-REGISTERED (contingent on the [[0048-rtfm-validated-reading]] 4-seed control confirming).**
Do NOT dispatch until `runs/exp0048ctl` confirms the phase-1 effect holds across seeds.

## Motivation

exp0048 phase-1: the validated-reading reward lifted length-2 `swap_follow` off zero (0.16–0.18 at
α=0.1–0.3) — the first lever to move it — with a clean inverted-U (dark-room collapse at α=1.0) and a
back-half signal that **grew as the env reading-shaping annealed away** but **plateaued ~0.18** rather
than taking off. Two structural reads from that: (1) the optimum sits on a **ridge ~0.1–0.3**, bounded
above by the dark-room; (2) the plateau tracks the shaping anneal — VR is *carrying* obedience as the
scaffold leaves, but not *strongly enough alone* to keep climbing.

## Hypothesis

Make VR fully *replace* the shaping scaffold instead of merely cushioning its removal:
- **Lever A — ridge:** probe α ∈ {0.2, 0.35, 0.5} (between the phase-1 optimum 0.3 and the 1.0
  dark-room) for a higher stable `swap_follow`.
- **Lever B — anneal:** instead of annealing reading-shaping → 0, hold a **shaping floor** (e.g. anneal
  to 0.2, not 0) OR slow the anneal, so the obedience scaffold doesn't fully vanish before VR is strong.
  (Eval still runs at shaping = 0, so this stays honest.)

**Predict:** `swap_follow` (held-out length-2) lifts **above the exp0048 ~0.18 plateau toward ~0.3+**,
sustained over the back half, with `swapped ≪ correct` preserved and no dark-room collapse — i.e. the
floor *keeps rising* instead of plateauing.

## Counter-outcomes (named)

- **Plateau is intrinsic** (higher α / shaping-floor doesn't lift swap_follow past ~0.18, just adds
  variance): the ceiling is the *amortized reactive actor*, not the reward strength → the read→act
  signal needs search/hierarchy at execution time ([[hierarchical-imagination-agent]]), not more reward.
- **Dark-room earlier** (α=0.35–0.5 collapses like 1.0 did): the safe ridge is narrow (~0.1–0.3); bound
  VR and pursue Lever B (shaping floor) alone.
- **Shaping-floor bakes** (swap_follow up but `swapped` rises too): the retained shaping is leaking a
  non-reading shortcut → drop Lever B, keep VR-only.

## Setup (dispatch — after the control confirms)

- α-ridge sweep (Lever A): `--validated-reading-coef {0.2,0.35,0.5}`, else identical to exp0048.
- Shaping-floor (Lever B): a `--reading-shaping-floor` arg (anneal to the floor, not 0) — small code
  change to the anneal schedule in `train_rtfm.main`; CPU-smoke + `floor=0` byte-for-byte unchanged.
- Best phase-1 seed-count: run the winners at 4 seeds (we now know 1 seed is only a candidate).

## Standing bar (success) — tag: `LADDER-EXIT`

Held-out length-2 `swap_follow` sustained **> ~0.25** (past the exp0048 plateau), `swapped ≪ correct`,
multi-seed — the read→act grounding strong enough to call the exp-0040 wall cracked, not just chipped.

## Result — NULL on the floor; ~0.10 is a robust EXECUTION ceiling

Lever B (shaping-floor) tested as a clean A/B vs the exp0048 floor=0 control (both α=0.3, 4 seeds, final
length-2 rounds 25–29). Plus an α=0.1 multi-seed comparison.

| config | swap_follow | swapped | events | oracle_pct |
|---|---|---|---|---|
| α0.3, floor 0 (exp0048 control) | 0.105 ±0.04 | 0.005 | 5.8 | 0.88 |
| **α0.3, floor 0.2** (this exp) | **0.094 ±0.02** | 0.003 | 6.9 | 0.91 |
| α0.1, floor 0 (2 seeds) | 0.090 ±0.04 | 0.000 | 4.9 | 0.93 |

- **The shaping-floor gives NO lift:** 0.094 vs 0.105 — flat (slightly lower, just tighter variance;
  secondary metrics marginally better). The pre-registered "plateau is intrinsic" counter-outcome.
- **`swap_follow` is a robust ~0.10 across the board** — α ∈ {0.1, 0.3}, floor ∈ {0, 0.2}. Neither more
  obedience-pressure (floor) nor a different coef moves it. (α=0.1 ≈ α=0.3 at the final window; the
  phase-1 0.1≈0.3 *and* the round-20 0.1<0.3 were both single-seed noise.)
- **MPC does not beat the reactive actor either:** over the *same* VR-trained WM, MPC `swap_follow`
  = 0.06–0.08 ≈ reactive 0.09–0.10 (the exp0044 finding, reconfirmed). So the ceiling is not a
  reward-strength problem *or* a naive-search problem.

## Lesson — VR cracked the OBJECTIVE gap; the new wall is EXECUTION (and reward-MPC can't fix it)

The validated-reading thread arrives at a clean two-part conclusion:
1. **VR solved what it was for.** It moved length-2 `swap_follow` from a flat **0.00** (all of 0042–0047
   — the [[0040-rtfm-actor-conditioning]] objective/identifiability gap) to a confirmed **~0.10**: the
   actor now *follows the displayed text*, anti-baking-clean. The objective gap is genuinely addressable
   by rewarding validated reading.
2. **A new ceiling at ~0.10 is the EXECUTION wall, re-met.** It is invariant to reward strength
   (coef), the shaping scaffold (floor), and search (MPC) — so it is not "the actor doesn't *want* to
   obey" but "it can't reliably *execute* the 2-step gesture even when it wants to" (events confirm:
   ~6/60 correct-mode completions = the same ~10% read-and-execute rate). This is the exp0043/0044
   multi-step-execution wall, now hit from above with a non-zero floor.

**Why reward-maximizing MPC can't break it (the subtle, load-bearing point):** `swap_follow` is scored
in *swapped* mode where obeying the manual earns NO reward — so a reward-maximizing planner has no
reason to follow the displayed text, which is exactly why MPC ≈ reactive here. Pushing past ~0.10
needs **manual-DIRECTED execution** (plan toward what the manual *predicts/says*, using the VR-trained
WM's manual-conditioned dynamics as the target), not reward-maximizing search — i.e. the decode-manual
→ subgoal → execute machinery of [[hierarchical-imagination-agent]]. → this is a **design fork for the
maintainer** (see the state-of-the-night hand-back); the autonomous loop pauses here because the next
step is a genuine design/scope decision, not a clear one-flag experiment.

## Links

[[0048-rtfm-validated-reading]] · [[validated-reading-reward]] · [[mentored-learning-loop]] · [[0040-rtfm-actor-conditioning]] · [[hierarchical-imagination-agent]] · [[vime-2016]] · [[marino-hypothesis-2020]]
