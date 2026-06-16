---
status: planned
owner: world-model
scope: local
verified: false
last_reviewed: 2026-06-16
---

# 0050: manual-directed hierarchy — break the ~0.10 execution ceiling (Director × manual conditioning)

**Status: PRE-REGISTERED — the capability build chosen 2026-06-16 (maintainer picked the full hierarchy
over a minimal MPC tweak). Substantial multi-session build; implement carefully with a reviewer.** Design:
[[hierarchical-imagination-agent]] (§1–3) + [[director-2022]]. Lit gate (scout 2026-06-16): [[director-2022]]
(anchor), FuN, HAC, THICK.

## Hypothesis

[[0048-rtfm-validated-reading]]/[[0049-rtfm-sustained-vr]] established: validated-reading cracked the
*objective* gap (the actor now follows the displayed text, length-2 swap_follow 0.00 → ~0.10) but a new
**~0.10 execution ceiling** is invariant to reward strength, scaffold, and reward-maximizing MPC — i.e.
the flat reactive actor can't reliably *sequence* the 2-step gesture (the exp0043/0044 multi-step
credit-assignment wall). A **Director-style manager/worker hierarchy trained in imagination** supplies
the missing temporal abstraction → should execute the gesture far more reliably than the flat actor.

## Two-phase design (the load-bearing experimental choice)

**Phase A — hierarchy WITHOUT the manual (isolate the cause).** State-conditioned manager only. Does a
hierarchy *alone* lift length-2 `correct` (achievement completion) above the flat actor's ~0.10? This
disambiguates: if YES → the ceiling was **credit assignment** (hierarchy is the fix); if NO → it's
**representation/grounding** and the hierarchy needs the manual. Run this FIRST — it's the cheaper,
decisive isolation (scout's key recommendation).

**Phase B — manual-DIRECTED manager (the novel axis).** Soft-condition the manager's subgoal
distribution on the manual embedding (Dynalang-style, NOT hard text→subgoal decode — anti-baking).
Headline: does held-out length-2 `swap_follow` lift past ~0.10 toward the ~0.3+ "cracked" bar, with
`swapped ≪ correct`? This is the manual-as-decomposition-tree thesis ([[hierarchical-imagination-agent]] §3):
the manual directs *which* subgoals the manager proposes.

## Architecture (minimal first cut — scout-recommended)

- **Worker:** the existing imagination-AC actor + one extra input — a **goal feature vector**; trained
  on a goal-similarity reward (cosine in RSSM feature space, FuN/Director), K=8-step imagination rollouts.
- **Manager:** small MLP over (RSSM belief [+ manual embedding in Phase B]) → softmax over a **discrete
  codebook of RSSM states** (K-means over a replay buffer first — no full VQ-VAE for the probe; Director
  §design), trained on task reward, proposing a new goal every K steps.
- **Subgoal representation:** discrete codebook (Director's tractable-search choice over FuN's continuous
  vectors), every code = a real reachable RSSM state.

## Stability (the HAC lesson — design around it from the start)

Co-training non-stationarity (worker changes → manager's subgoal→outcome mapping drifts) is the top
failure mode (HAC). Mitigations, all designed in up front:
1. **Hindsight relabel** a fraction of manager transitions with the RSSM state the worker *actually*
   reached (HAC) — guaranteed signal.
2. **Separate replay buffers** for manager vs worker.
3. **Keep the Dreamer alternation** (real data → WM update → imagined rollouts → AC update); do NOT
   interleave WM and actor updates.
4. **Pre-seed the codebook** with RSSM states from the VR-trained agent's rollouts (it reaches the
   gesture endpoint ~10% of length-2 episodes — exp0048/0049 checkpoints `runs/exp0049f02/s*_s*.pt`),
   so the gesture-completion state is in the codebook from the start (Director codebook-coverage fix).

## Implementation plan (phased; each step smoke-tested before the next)

1. Goal-conditioned worker (actor + goal input + similarity reward) — CPU shape/grad tests.
2. K-means codebook over replay RSSM states + manager MLP (state-only) + two-level imagination loop with
   hindsight relabel — CPU smoke.
3. **Dispatch Phase A** (multi-seed; bar below). Reviewer (opus) on the diff before dispatch.
4. Add manual embedding to the manager → **dispatch Phase B**.

## Counter-outcomes (named)

- Phase A null (hierarchy alone doesn't beat ~0.10): ceiling is grounding, not credit assignment → Phase
  B is the real test (manual must direct the subgoals).
- Co-training diverges (manager/worker oscillate): the HAC instability → check hindsight relabel + buffer
  separation; if persistent, fall back to the recursive decompose-or-execute MPC operator
  ([[hierarchical-imagination-agent]] §3) over the VR-WM instead of a trained manager.
- Phase B bakes (swapped rises with swap_follow): the manual conditioning leaked a shortcut → audit.

## Standing bar (success) — tag: `LADDER-EXIT`

Phase A: length-2 `correct` clearly above the flat ~0.10 (credit-assignment confirmed). Phase B: held-out
length-2 `swap_follow` sustained **> ~0.25** with `swapped ≪ correct`, multi-seed — the read→act grounding
strong enough to call the exp-0040 wall cracked. Unblocks the directable-competence milestone
([[0007-crafter-mastery-milestone]]) and the next rung.

## Links

[[hierarchical-imagination-agent]] · [[director-2022]] · [[0049-rtfm-sustained-vr]] · [[0048-rtfm-validated-reading]] · [[0043-rtfm-execution-wall]] · [[validated-reading-reward]] · [[mentored-learning-loop]] · [[hierarchy-and-credit]]
