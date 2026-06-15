---
status: draft
owner: world-model
scope: local
verified: true
last_reviewed: 2026-06-14
---

# 0033: Length-1 grounding diagnostic — the manual-conditioned WM agent does NOT read content

## Hypothesis

0032 (length-3 recipes) never lifted `correct` off zero even with the full calibrated recipe.
To separate "can't ground" from "can't do a 3-step sequence," drop to **length-1 recipes**
("perform action X", X randomized per episode) — the purest grounding test, trivial credit
assignment. 4 seeds (parallel, for a clear read past the seed variance). Predict: if the
architecture grounds at all, length-1 should show it; the gold-standard metric is **swap_follow**
(does the learner perform the *swapped* manual's action), NOT correct−none.

## Result

**Decisive: the agent does NOT read manual content — caught cleanly by the swap test.** The
`correct − none` "grounding" signal *appeared* (s1 reached correct=0.40 vs none=0.05) and would
have looked like success — but the swap test exposed it as a confound. Stable across rounds 8/10/11
and consistent across all 4 seeds:

| metric | s1 (r8–11) | meaning |
|---|---|---|
| correct (true manual) | 0.25–0.40 | does the true recipe with a manual present |
| none (no manual) | 0.00 | **fails without a manual** → manual *presence* matters |
| **swapped** (wrong manual) | **0.30–0.35 ≈ correct** | does the TRUE recipe just as well with a SWAPPED manual |
| **swap_follow** | **0.00** | **never performs the swapped manual's action** |

`swapped ≈ correct` + `swap_follow ≈ 0` on every seed = the agent **ignores the manual's content**.
It exploits manual *presence* (none=0) and infers the recipe from **another channel**, but does NOT
read the text — a swapped manual changes nothing about its behaviour.

## Lesson

**The swap test worked — it caught a non-reading shortcut that `correct − none` could not.** This
is the methodology payoff of insisting on swap-following as the success metric (rung-4
design §1/§4): `correct − none = 0.35` screamed "grounding!", but `swap_follow = 0` revealed the
agent never reads content. Had we used reward-gap alone we would have declared a false win. The
anti-baking design did its job.

**Likely cause — a VISION shortcut, and it's a two-domain finding.** crafter-rtfm's own learned
stub grounded (swap_follow=1.00, [[HO-0004]]) — but that stub is **text-only** (manual→encoder→
action, no image). **Our agent has vision (frozen DINO frames) AND text.** With vision it can infer
the recipe from the **staged visual state** ([[HO-0002]] env-staging arranges the recipe's
preconditions) and never needs to read. The text-only stub had no such shortcut, so it was *forced*
to read. Hence:
- **world-model side:** the WM-conditioned vision+text agent finds the visual shortcut → never
  learns to read (the WM-only conditioning, chosen for baking-resistance, gives the actor no
  incentive to read when vision already solves it).
- **crafter-rtfm side:** the reading-necessity gate (scripted no-text fails; text-only stub grounds)
  **does not hold for a vision+text learner** — the staging leaks the recipe visually. A real gap →
  **a handoff to crafter-rtfm** (need a mode where the recipe is NOT visually inferable).

→ next: (1) handoff to crafter-rtfm re: vision-shortcut reading-necessity; (2) once the env can't be
solved by looking, re-test — and consider direct actor-conditioning (the text→action path
crafter-rtfm proved learnable), keeping swap_follow as the gate.

## Links

[[0032-rtfm-grounding]] · [[rung4-manual-conditioned-agent]] · [[grounding-env-spec]] · [[no-hardcoded-env]]
