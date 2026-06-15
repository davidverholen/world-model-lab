---
status: draft
owner: world-model
scope: local
verified: true
last_reviewed: 2026-06-15
---

# 0044: MPC ≈ reactive at length-2 — planning doesn't crack it; the wall relocates to WM rollout fidelity

## Hypothesis

exp0043 localized the rung-4 wall to *execution*: the WM reads a length-2 gesture (inv_ratio>1) but
the reactive imagination-actor can't execute it (correct≈0). The hierarchical-imagination design
([[hierarchical-imagination-agent]] §5,§7) proposes imagination-**planning** as the execution
mechanism. exp0044 tests its bottom rung: CEM-MPC over the SAME trained manual-conditioned WM
(`agents/rtfm_mpc.py`, prior-mean rollout, discounted reward-head return, refit-to-elites) vs the
reactive actor — a clean A/B. Pre-registered branches: MPC ≫ reactive ⇒ wall was credit-assignment,
planning is the fix; MPC ≈ reactive ⇒ wall is WM rollout-accuracy. Curriculum + aux so the WM reads
length-2; `--mpc-eval`; 4 seeds.

## Result — MPC marginally > reactive, but neither cracks length-2

Final round (29, length-2, coef=0):

| seed | reactive correct | MPC correct | inv_ratio (WM reads?) |
|---|---|---|---|
| s0 | 0.05 | 0.10 | 1.22 |
| s1 | 0.10 | 0.05 | 1.10 |
| s2 | 0.05 | 0.10 | 1.23 |
| s3 | 0.05 | 0.10 | 1.17 |
| **mean** | **0.06** | **0.09** | reads ✓ |

- **MPC is consistently but marginally ahead** — MPC ≥ reactive on 3/4 seeds at both round 23
  (mean 0.14 vs 0.075) and round 29 (0.09 vs 0.06). A small, repeatable directional signal that
  planning helps a little.
- **But the gap is within eval noise** (n=20 seeds) and **neither executes length-2** — both ~0.05–0.10
  vs length-1's ~0.4. Planning does **not** solve the execution wall.
- The WM **reads** length-2 throughout (inv_ratio 1.10–1.23), so reading is not the limiter.

## Lesson — the wall relocates from policy to WORLD-MODEL imagination fidelity

This is the second pre-registered branch: swapping a *search* planner for the reactive policy barely
moves the needle, so the binding constraint is **not** the policy's credit assignment — it is the
**world model's multi-step rollout / reward fidelity**. The WM's *belief* encodes the manual
(inv_ratio>1), but planning over its imagined 2-step rollout + reward-head prediction isn't sharp
enough to reliably find the gesture. The wall moves from *"the actor can't learn the 2-step sequence"*
(exp 0040/0043 framing) to *"the WM can't imagine the 2-step payoff accurately enough to plan through
it."* Even a 2-step horizon — short by design — is apparently past the WM's reliable-rollout range at
one_shot length-2.

**Implication for the design.** Imagination-planning (and Director-style hierarchy, and the whole
hierarchical-imagination agent) all *plan through the WM* — they inherit its rollout fidelity as a
hard ceiling. So **WM multi-step fidelity is upstream of the execution capability**: no planner can
plan its way out of an inaccurate imagined rollout. This reframes the next work: before more planning
machinery, sharpen the WM's short-horizon predictive accuracy (and the reward head's multi-step
calibration). It also echoes the exp 0017–0025 theme (imagination accuracy/calibration is the recurring
limiter) and the calibration dependency flagged in the design (§5).

**Caveats / confounds to rule out before concluding "fundamental".** The marginal MPC result could be
partly an MPC-setup limit, not pure WM fidelity: (a) prior-**mean** rollout (vs sampled/averaged) may
blur the 2-step consequence; (b) no value tail; (c) reward head trained on tutorial+shaping may
predict a soft reward shape; (d) receding-horizon replan each step vs committing the planned pair.

→ next: a **localizing diagnostic** — does the WM/reward-head predict high return for the *oracle*
2-action gesture from the current belief (privileged `manual_facts`, eval-only)? If yes, the gap is
MPC-search and is tunable; if no, it's WM rollout/reward fidelity and that's the real next target.
(Analogous to how inv_ratio localized reading-vs-execution in exp 0043.) Maintainer checkpoint before
investing further — this is a meatier fork than another rtfm lever.

## Reproduction

- Commit c679a4a@world-model (`agents/rtfm_mpc.py`, `--mpc-eval`).
- `dispatch_rtfm.sh exp0044 4 --rounds 30 --curriculum-rounds 10 ... --length 2 --one-shot
  --reading-shaping-coef 1.0 --manual-aux-coef 1.0 --mpc-eval` (MPC horizon 5, samples 200, iters 3).
- Artifacts: `runs/exp0044/` (reactive `correct=` and `MPC correct=` lines per round).

## Links

[[0043-rtfm-execution-wall]] · [[hierarchical-imagination-agent]] · [[director-2022]] · [[imagination-training]] · [[0007-crafter-mastery-milestone]]
