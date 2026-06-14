---
status: draft
owner: human
scope: local
verified: true
last_reviewed: 2026-06-14
---

# 0034: one_shot re-test — base-reward farming (fixed), then a sparse-reward ignition wall

## Hypothesis

0033 caught the manual-conditioned agent solving by **looking, not reading** (swap_follow=0,
swapped≈correct) — a vision shortcut off the staged state. crafter-rtfm's fix (HO-0006): `one_shot`
forfeits the tutorial after the first gesture-length window, killing the within-episode **search**
leak so reading is the only path. Re-test under `make_recipe_env(CORRECT, length=1, one_shot=True)`,
4 seeds: with looking/searching dead, does the WM-conditioned agent now **ground** (swap_follow
rises, swapped≪correct)?

## Result — two findings, in sequence

**(1) A reward-composition confound: the agent farmed readingless reward (run `runs/exp0034`).**
The re-test sat at `correct = none = swapped = 0` for 8+ rounds while collection reported ~120
"reward events" per round — *not* a no-signal bug. Probe of the env step reward: under `one_shot`
CORRECT a *random* agent gets `+1.0` step rewards (1–2/episode) while `tutorial_score` stays 0. Root
cause: the env's step reward `r` (env.py:186+196) **mixes** base-Crafter achievements
(wood/food/drink — farmable WITHOUT reading) **with** the sparse tutorial bonus, both magnitude 1.0.
`collect_rtfm` trained the WM + actor on `r`, so the agent optimized the dense readingless base
reward and never needed the manual. The honest signal was being drowned.

**Fix:** train on the **tutorial-only** reward — `r_read = len(info["tutorial_newly"])` — so reading
the manual is the *sole* reward source; the env's mixed `r` is discarded (train_rtfm.py
`collect_rtfm`). This is the necessary correction, but it was **not sufficient** —

**(2) The residual wall is pure `one_shot` sparsity: no ignition gradient (run `runs/exp0035`).**
With the tutorial-only reward, the reading-event count collapses to **chance**: round 0 (random
collection) ~2 events / 60 episodes; rounds 1+ (agent collection) bounce **0–6**, i.e. there is no
climb. With ~2–6 reward-positive transitions in an ~11k buffer, the learner has **no gradient toward
reading**, and eval stays at chance every round (`correct ≈ none ≈ 0.0–0.1`, `swap_follow` noise
≤0.15, `grounding ≈ 0`). The agent never ignites.

| run | reward stored | reward events/round | eval | verdict |
|---|---|---|---|---|
| exp0034 | env `r` (mixed) | ~120 (farmed base reward) | all 0 | confound — farms readingless reward |
| exp0035 | tutorial-only | 0–6 (chance) | chance, swap_follow noise | ignition deadlock (sparsity) |

## Lesson

**`one_shot` is correct but creates a sparse-reward bootstrap deadlock for a *from-scratch*
learner.** It does its job — search/no-text policies provably fail and a *scripted* reader still
solves it with swap_follow=1.0 (so HO-0006 is **accepted**: reading is necessary AND sufficient at
the env level). But the only reward-positive behavior is reading correctly on the **single** scored
attempt, which an agent that doesn't yet know how to read reaches only at chance → no gradient to
climb. one_shot removed the search reward (the point) but left nothing dense to bootstrap reading
from. This is exactly the curriculum/shaping need crafter-rtfm pre-flagged when answering HO-0006.

**Methodology note:** the base-reward-farming confound is the same class of trap as 0033's vision
shortcut — a dense, readingless signal the optimizer prefers over the sparse honest one. The fix
pattern is the same: starve every readingless reward channel until reading is the *only* way to be
rewarded. Having done that, the remaining problem is no longer "the agent cheats" but "the agent
can't bootstrap" — a cleaner, more honest failure.

→ next: **HO-0007** (open) — a training-time curriculum/shaping signal that gives a dense
reading-gated gradient without reopening the search leak; eval stays the honest `one_shot` +
swap_follow on held-out manuals. Re-test once crafter-rtfm answers.

## Reproduction

- Diagnosis + fix commit: (this milestone) @world-model; env fix `6394c6f@crafter-rtfm` (`one_shot`).
- Dispatch: `scripts/dispatch_rtfm.sh exp0035 4 --rounds 20 --episodes-per-round 60
  --updates-per-round 400 --ac-updates-per-round 400 --seq-batch 16 --window 12 --burn-in 4
  --horizon 10 --n-train-seeds 400 --n-eval-seeds 20 --max-steps 48 --length 1 --one-shot`
- Artifacts: `runs/exp0034/` (mixed-reward, dead), `runs/exp0035/` (tutorial-only, ignition wall).

## Links

[[0033-rtfm-length1-grounding]] · [[0032-rtfm-grounding]] · [[rung4-manual-conditioned-agent]] · [[grounding-env-spec]] · [[no-hardcoded-env]]
