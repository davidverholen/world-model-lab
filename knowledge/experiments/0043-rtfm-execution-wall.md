---
status: draft
owner: human
scope: local
verified: true
last_reviewed: 2026-06-15
---

# 0043: length-2 is a PURE EXECUTION wall — the WM reads it, the actor can't do it

## Hypothesis

exp0041/0042: length-2 won't ignite (cold or curriculum). Two failure modes were confounded: maybe
the WM can't *read* a 2-step gesture (reading-compositionality), or maybe it reads it but the actor
can't *execute* it (execution). The `inv_ratio` diagnostic (aux on) separates them: run the length-2
curriculum **with `--manual-aux-coef 1.0`** and compare the belief's manual-specificity at length-2
vs length-1. inv_ratio>1 + correct≈0 ⇒ pure execution failure; inv_ratio≈1 ⇒ reading-compositionality.

## Result — inv_ratio is the SAME at length-2 as length-1, but correct collapses

| phase | inv_ratio (WM reads?) | correct | grounding |
|---|---|---|---|
| **length-1 (round 9)** | 1.13 / 1.27 / 1.23 / 1.27 | ~0.35 (s2,s3 ground) | up to 0.35 |
| **length-2 (round 29)** | 1.13 / 1.17 / 1.19 / 1.25 | 0.05–0.10 (all seeds) | ~0.05 |

The WM's manual-specificity (`inv_ratio`) at length-2 is **indistinguishable from length-1** on every
seed — the belief encodes the 2-step gesture just as well as the 1-step one. Yet `correct` collapses
from ~0.35 to ~0.05. **The world model reads the multi-step recipe; the actor cannot execute it.**

## Lesson — the wall is multi-step EXECUTION, and it's the same wall as the rung-3 plateau

This is the decisive, confounder-free localization. Combined with exps 0037–0042 it closes the
diagnosis with four converging results:
1. swap_follow stuck ~0.25 even at length-1 (actor under-executes — 0040).
2. length-2 won't ignite cold (0041).
3. length-2 won't ignite warm-started (curriculum — 0042).
4. **length-2: WM reads it (inv_ratio>1, = length-1), actor can't execute (correct≈0) — 0043.**

**Reading is solved; multi-step / compositional EXECUTION is THE wall.** A length-2 gesture is the
simplest multi-step action: sequence two specific actions to a sparse, one-shot-gated reward. The
reactive actor + imagination-AC learns this poorly — a sequential credit-assignment problem under
sparse reward.

**The unification (important):** this is the *same* bottleneck as the rung-3 Crafter depth plateau,
which exps 0026–0031 localized to *actor/discovery*, not perception. Plain Crafter stalls on the deep
tech tree (multi-step: stone→iron→diamond); rtfm stalls on the 2-step gesture. Both are
multi-step-execution / credit-assignment / **hierarchy** ([[hierarchy-and-credit]],
[[temporal-abstraction]]) — the capability we have repeatedly found missing and have not yet built.
It also echoes Dreamer-4, which needed explicit per-subtask staging to get *its* deep tree
([[dreamer4-2025]]).

→ This is not another rtfm lever — it is the **next major capability** (multi-step execution:
hierarchy / temporal abstraction / goal-conditioned credit assignment), and it pays off on BOTH the
reading-execution AND the plain-Crafter-depth fronts at once. Maintainer's strategic call + ADR-0007
ratification (this is its later-pillar work). Park rtfm length-scaling here; reading is demonstrated.

## Reproduction

- Commit 991ed32@world-model (`--curriculum-rounds`) + aux (0b72bcd).
- `dispatch_rtfm.sh exp0043 4 --rounds 30 --curriculum-rounds 10 ... --length 2 --one-shot --reading-shaping-coef 1.0 --manual-aux-coef 1.0`
- Artifacts: `runs/exp0043/` (inv_ratio in each round line).

## Links

[[0042-rtfm-length-curriculum]] · [[0040-rtfm-actor-conditioning]] · [[hierarchy-and-credit]] · [[temporal-abstraction]] · [[0007-crafter-mastery-milestone]] · [[dreamer4-2025]]
