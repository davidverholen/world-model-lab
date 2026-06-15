---
status: planned
owner: world-model
scope: local
verified: false
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

## Links

[[0048-rtfm-validated-reading]] · [[validated-reading-reward]] · [[mentored-learning-loop]] · [[0040-rtfm-actor-conditioning]] · [[hierarchical-imagination-agent]] · [[vime-2016]] · [[marino-hypothesis-2020]]
