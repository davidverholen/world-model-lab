---
status: draft
owner: world-model
scope: local
verified: false
last_reviewed: 2026-06-16
---

# 0051: optimizing flat validated-reading on the 2-step tutorial (no hierarchy)

After the hierarchy thread concluded negative ([[0050-rtfm-hierarchy]] — wrong tool for a 2-step
gesture; and, the maintainer's key insight, the *worker had no grounding incentive*: every acting
level needs it), the direction is to keep optimizing the **flat** validated-reading architecture
(where the incentive sits directly on the actor) and push the length-2 `swap_follow` ceiling past the
confirmed ~0.10 ([[0048-rtfm-validated-reading]] / [[0049-rtfm-sustained-vr]]).

## Budget probe — REAL ceiling, not compute-bound

`runs/exp0051budget`: flat VR coef 0.3, **2× budget** (40 rounds, curriculum 12, 800 ac-updates) vs the
30-round/400-update 0.105 baseline. 2 seeds, last-5 length-2:

| | swap_follow | correct |
|---|---|---|
| baseline (30r/400u) | 0.105 | 0.073 |
| **2× budget (40r/800u)** | **0.115** | 0.115 |

`swap_follow` is **flat (~0.11) despite doubling training** → the grounding ceiling is NOT compute-bound
at this scale. (correct rose a little — more budget buys slightly better task completion, but not
grounding.) So the lever is a **stronger/denser grounding incentive**, not more compute.

## In flight + next (the incentive lever)

- **coef 0.5** (`runs/exp0051c05`, 2 seeds): cheap probe filling the dose-response gap (exp0048 tested
  0.1/0.3 ≈ 0.10, 1.0 dark-rooms; 0.5 untested). Low-odds (likely ~0.10), but closes the coef question.
- **On-deck — decoupled/denser actor-VR:** apply the maintainer's insight directly. Currently VR is
  folded into the collection reward `r_read`, so the actor feels it only through the (blurred, shared)
  reward head. Give VR its **own reward head** (predict the stored VR component separately) and have the
  imagination actor optimize `task_reward + λ·VR_head` — a dedicated, undiluted grounding signal on the
  acting policy, still reality-judged (VR is computed on real transitions at collection). Moderate
  build; reviewer-gate. If coef 0.5 is flat, this is the next test.

## Links

[[0048-rtfm-validated-reading]] · [[0049-rtfm-sustained-vr]] · [[0050-rtfm-hierarchy]] · [[validated-reading-reward]] · [[0040-rtfm-actor-conditioning]]
