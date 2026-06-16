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

## Result — Phase A v1: DIVERGENT (hierarchy degenerate; fixed → v2 running)

First Phase-A dispatch (4 seeds, `--hierarchy`, state-only, K=5; same WM recipe as the flat exp0048ctl
0.105 baseline). NEGATIVE, but as a *bug*, not a clean test:

| | length-1 (r3–9) | length-2 (r10–29) | flat baseline |
|---|---|---|---|
| correct | up to 0.35 | **0.00** (all rounds) | 0.073 |
| swap_follow | ~0.05 | **~0.01** | 0.105 |

**Diagnosis (decisive, from the s0 trajectory):** the hierarchy *partially worked at length-1*
(correct → 0.35) then **collapsed and diverged at length-2**: `worker_loss` exploded to **±10 → ±80**
(unbounded policy-gradient — the two-hot value's symexp output blew up the raw advantage), and the
manager's imagined macro-reward sat at 0.3–0.7 while real reward was ~0 (**imagination exploitation**,
no critic-on-replay anchor). Compounding it, the hierarchy **drove its own collection from round 3**, so
once it went degenerate it **poisoned the buffer** with junk data (vs the flat baseline, whose actor
collected competently). So this is NOT "grounding is the ceiling" — the hierarchy never functioned at
length-2.

**Phase A v2 (advantage-norm + grad-clip + `--hier-flat-collect`):** ALSO NEGATIVE. 4 seeds, final
length-2: **correct 0.000, swap_follow 0.000 on EVERY seed** (vs flat 0.073 / 0.105). The advantage
normalization + grad-clip *reduced* but did not eliminate the worker instability — `worker_loss` still
oscillates ±30–90 across all seeds (policy collapse: peaked worker → extreme log-probs, not advantage
explosion). And decisively: **even with clean flat-collection (a good WM + competent data), the
hierarchy executes at zero.**

## STATUS — hierarchy thread STALLED (2 failed attempts); STOP per the stall rule

Two pre-registered attempts (v1 divergent, v2 stable-but-zero), headline (length-2 swap_follow/correct)
**did not move off ~0** — below even the flat ~0.10. Per PROCESS.md §Efficiency guardrails (countable
stall rule), the thread is a wall; a 3rd attempt needs written justification that it is not a
known-class repeat. **Consolidated diagnosis:** the *minimal first cut* — goals in **raw belief space**,
a **K-means codebook**, a **cosine** worker reward, and **no learned goal autoencoder** — is the likely
culprit. This is exactly what [[director-2022]]'s own ablation flags as load-bearing ("removing the goal
autoencoder causes failure in most environments"): raw-RSSM cosine goals give the worker an
uninformative/unstable target, so it never learns to reach subgoals and the manager's codes are
meaningless. We skipped the VQ-VAE goal autoencoder for the first probe; that skip appears fatal.

**→ Handed back to the maintainer** (a genuine scope/design fork, not a clear one-flag experiment):
(A) build the real **VQ-VAE goal autoencoder** (Director's load-bearing piece) — the principled fix,
bigger build; (B) the lighter **recursive decompose-or-execute MPC** over the VR-trained WM
([[hierarchical-imagination-agent]] §3) — no trained manager, sidesteps the co-training instability;
(C) **bank validated-reading** (the confirmed 0→0.10 objective-gap result) and pause the hierarchy. NOT
Phase B (manual-directed manager): manual-directing a non-functional executor can't help — the executor
must work first. The hierarchy code (models/hierarchy.py + imagine_hierarchy_rtfm, reviewer-cleared) is
banked for whichever direction resumes it.

## exp0050b/c — diagnostic-driven iteration (maintainer re-engaged 2026-06-16)

Maintainer reopened the thread ("test more, propose prior"). Three diagnostic-driven tests localized
the root cause:

- **exp0050b (instrumented):** added `worker_sim` (goal-similarity), `code_entropy`, `codes_used`
  logging. Found: worker_sim ~0.6, manager code-entropy healthy (diverse codes), worker_loss now
  stable — yet correct = 0. *Initially* read as "machinery works, codebook lacks a success goal."
- **exp0050c (success-biased codebook, `--hier-codebook-success 0.5`):** oversample reward-earning
  beliefs into the codebook. **Still correct = 0**; the manager now *collapses* (entropy → 0, 1–2
  codes used on s1).
- **Consolidated diagnosis (the real one):** `worker_sim` is **flat ~0.6 and never rises**, invariant
  to the codebook change — i.e. it is the **ambient belief-cosine** (any belief ≈ 0.6-similar to any
  K-means centroid in raw 288-d belief space), NOT learned goal-reaching. **The goal space is
  uninformative**: raw-belief cosine does not separate states, so codebook goals are meaningless
  targets, the worker can't meaningfully reach a *specific* goal, and the manager rationally collapses
  (all goals ≈ equivalent). This is exactly the piece [[director-2022]] §2 calls load-bearing — the
  **learned goal autoencoder** (VQ-VAE) that gives a separable, low-dim goal space — which the minimal
  first cut skipped. Three cheap tests cleanly establish: *the shortcut (raw-belief goals) cannot work;
  the goal autoencoder is required.*

**Decision point (not a quick flag-test anymore):** the next real fix is building the goal autoencoder
(Director's load-bearing component) — a focused build, not a one-line change. So the fork is genuine:
(A) build the goal-encoder/autoencoder properly, or (C) bank validated-reading (the confirmed 0→0.10
result) and schedule the full Director build deliberately. Surfaced to the maintainer.

## Links

[[hierarchical-imagination-agent]] · [[director-2022]] · [[0049-rtfm-sustained-vr]] · [[0048-rtfm-validated-reading]] · [[0043-rtfm-execution-wall]] · [[validated-reading-reward]] · [[mentored-learning-loop]] · [[hierarchy-and-credit]]
