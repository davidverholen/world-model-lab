---
status: draft
owner: human
scope: local
verified: true
last_reviewed: 2026-06-15
---

# 0040: actor-conditioning doesn't crack swap_follow — PARK the strengthening thread (3 levers exhausted)

## Hypothesis

exp0038/0039 diagnosed swap_follow (~0.25) as **actor-bound, not WM-bound** (the WM reads — inv_ratio
>1, swapped≪correct — but the actor under-executes). The design §6.2 pre-authorizes conditioning the
policy directly as a fallback "if planning through the m-conditioned WM is too weak, and re-verify
the swap test." exp0040: exp0036 config (shaping anneal) + `--actor-cond` (the belief cross-attends
the manual tokens → the actor; commit 43a77c5), aux OFF, held-out eval. Predict: a direct
text→action path lifts swap_follow; held-out swap test guards baking.

## Result — null/negative + high variance

Final round (coef=0, held-out eval):

| seed | correct | none | swapped | grounding | swap_follow |
|---|---|---|---|---|---|
| s0 | 0.30 | 0.05 | 0.25 | 0.25 | 0.20 |
| s1 | 0.50 | 0.10 | 0.15 | 0.40 | 0.05 |
| s2 | 0.30 | 0.30 | 0.30 | 0.00 | 0.05 |
| s3 | 0.65 | 0.15 | 0.05 | 0.50 | 0.40 |

- **swap_follow did NOT rise** — mean ~0.18, *below* shaping-only (0.24) and shaping+aux (0.30).
  s1/s2 collapsed to 0.05; s2 failed to ground at all (none=correct=swapped).
- **But high variance with a strong tail:** s3 is the best single run in the whole rung-4 series
  (correct 0.65, grounding 0.50, swapped 0.05, swap_follow 0.40) — capacity *is* there, but the
  outcome is seed-dependent (different basins), not a reliable lift.
- **No baking:** swapped never exceeds correct on any seed; the held-out guard held. Actor-
  conditioning didn't cheat — it just didn't help, and destabilized training.

## Lesson — swap_follow is bounded by the training OBJECTIVE, not the architecture

Three levers have now failed to lift swap_follow off ~0.25, each ruling out an architectural cause:

| lever | exp | swap_follow | what it rules out |
|---|---|---|---|
| 2× training | 0037 | ~0.25 (null) | undertraining |
| reading aux (coef 1, 3) | 0038/0039 | ~0.30 (flat) | weak WM reading (inv_ratio rose, swap_follow didn't) |
| actor-conditioning | 0040 | ~0.18 (worse) | actor lacks a text→action path |

The reading is **genuine and robust** throughout (swapped≪correct, grounding ~0.4, inv_ratio>1).
What resists every architectural fix is *execution under a swap*. The coherent explanation is an
**objective/identifiability gap**: collection is always `ManualMode.CORRECT`, where "follow the
displayed manual" and "do the episode's true recipe" are the **same** reward-optimal policy. Nothing
in the reward pressures the agent toward *obeying the displayed manual per se*; it learns "use the
manual to find the reward-earning gesture," which **diverges from obedience exactly at the swap**
(where the displayed manual is anti-correlated with this episode's reward). A scripted reader gets
swap_follow=1.0 because it is *hardwired* to obey; a reward-maximizing RL agent has no such pressure.
So swap_follow ~0.25–0.40 may be a **ceiling of correct-mode reward training**, not a model defect.

## Decision — PARK, per the stall rule; maintainer call on the objective

Per the autonomous stall rule (a failure mode surviving multiple redesigns → stop the thread, write
it up, don't keep throwing levers), I am **parking the swap_follow-strengthening thread**. It is not
an architecture problem; the remaining moves change the *task/objective* and need the maintainer's
research-direction judgment (they touch the anti-baking design and what swap_follow should measure):

1. **Partial-anneal the reading-shaping** (don't take coef fully to 0): the HO-0007 shaping already
   rewards *following the displayed gesture* — keeping a residual could retain "obey the manual"
   pressure, with eval still at coef=0 to test internalization. (Risk: blurs the honest-eval line —
   is high swap_follow then "reading" or "trained to obey"? It IS reading, but the maintainer should
   decide if that counts.)
2. **Train on mixed/swapped modes** so obeying the *displayed* manual is what's rewarded even when it
   differs from the true recipe — directly creates the follow-displayed pressure correct-mode lacks.
   (Changes task semantics; needs a crafter-rtfm handoff.)
3. **Accept the result as-is.** The headline — *genuine, content-sensitive reading-to-learn-dynamics,
   ignited from scratch* (swapped≪correct on held-out, grounding ~0.4) — is already secured. Treat
   swap_follow ~0.3 as "partial grounding," move to length-2 / the next rung, and revisit obedience
   later. (My default recommendation: the reading thesis is demonstrated; swap_follow→1.0 is a
   separate "instruction-obedience" question.)

## Reproduction

- Commit 43a77c5@world-model (`--actor-cond`, off by default; `ConditionedActor`).
- `dispatch_rtfm.sh exp0040 4 --rounds 20 ... --reading-shaping-coef 1.0 --actor-cond`
- Artifacts: `runs/exp0040/` (4 seeds + logs).

## Links

[[0038-rtfm-aux-grounds-wm]] · [[0036-rtfm-shaping-ignition]] · [[rung4-manual-conditioned-agent]] · [[hierarchy-and-credit]] · [[no-hardcoded-env]]
