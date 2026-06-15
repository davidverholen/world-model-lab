---
status: proposed
owner: world-model
scope: local
sources: []
verified: false
last_reviewed: 2026-06-15
---

# 0007: Crafter mastery = "directable competence" (the rung-3 exit milestone)

**Status: PROPOSED (2026-06-15) — pending ratification.** Drafted from a design
conversation; captures a major-milestone commitment so the autonomous loop has a concrete target
to optimize against instead of a vibe.

## Context

We have validated **reading-to-learn-dynamics** at length-1 on crafter-rtfm (exp 0036: genuine,
content-sensitive, `swapped ≪ correct` on held-out, ignited from scratch). The strategic
frame: **master Crafter — the purpose-built 2D RL env — before advancing to 3D (Phase 4).**
This is the no-rung-skipping discipline of [[0003-environment-ladder]], applied: don't climb to 3D
before we've actually finished the 2D game built for exactly this.

"Finish Crafter" needs a crisp, non-gameable definition. Two framings were rejected:
- **"All 22 achievements in one episode"** — near-impossible for *anyone* (survival and deep-crafting
  compete for the same episode-time budget; the standard metric is per-achievement success rate, not
  a single perfect run).
- **"Beat DreamerV3's aggregate Crafter score"** — gameable: the score is a geometric mean, so
  farming the *easy* achievements inflates it without touching the deep tree that is the real
  frontier. (Note: no published agent *finishes* Crafter — DreamerV3 tops the leaderboard at ~14%
  score and only occasionally reaches the iron→diamond chain. So this milestone is frontier-
  *advancing*, not a reproduction target.)

## Decision (proposed)

Define the **rung-3 exit milestone** as **DIRECTABLE COMPETENCE**:

> For any target achievement we point the agent at, it reliably reaches that achievement within the
> episode budget — measured as per-achievement **success-rate-on-demand across all 22**, with the
> bar biting on the **deep tree** (iron, diamond) that no agent reaches reliably today.

The bar is **competence/reachability, not aggregate score** — it cannot be farmed by the easy
achievements; it directly demands general, controllable mastery.

**Mechanism = the project thesis (three pillars):**
1. **Dynamics manual** — the agent *reads the rules of the game* (recipes + combat + survival), which
   condition the world model's dynamics. This is reading-to-learn-dynamics, validated in rtfm; it is
   what could let us surpass DreamerV3, which has no reading channel.
2. **Goal / task conditioning** — the agent is *directed at a target* ("get diamond this episode").
   This is a DISTINCT mechanism from the dynamics manual (rules vs target) and is **not yet built**.
3. **Imagination** — the agent plans the path through the now-grounded world model. This is the
   bridge from *knowing* (what to do) to *doing* (executable plan), and is where our current results
   are weakest (the actor-bound swap_follow ceiling, exp 0040).

Two tiers: **we** set the goal → *directable* competence (the **measurable milestone**); the **actor**
sets its own goal → *autonomous* competence (the open-ended stretch BEYOND this milestone).

**Why this is *ours*, not a reproduction:** surpassing DreamerV3 on the deep tree *is* the thesis
ablation — Dreamer has no manual/reading. If reading-conditioned imagination is what gets us where
pure-RL Dreamer stalls, a swap/ablation that holds (depth with manual ≫ depth without, swap-proven)
demonstrates the reading is the cause, not baked priors.

**Staircase (each rung gated on a thesis-checkpoint so we never grind blind):**
1. **Reading → executable plan** in clean rtfm: fix the read→imagine→execute link (the parked
   actor/objective work + length-2 ignition). *If reading can't yield a length-2 plan, it won't yield
   a 10-step diamond plan.*
2. **Reading lifts Crafter depth**: manual-conditioned Crafter beats a matched no-manual agent on
   tech-tree depth (the ablation), swap-test-proven.
3. **Whole-game manual**: broaden from recipes to recipes + combat + survival.
4. **Goal-conditioning**: add the target channel → directable competence.
5. **The milestone**: reliable per-achievement reachability across all 22, deep tree included.

## Consequences

- **Phase 4 (continuous/3D) stays parked** until this is met — the explicit point of the milestone.
- The **swap-test anti-baking guard** ([[no-hardcoded-env]], [[rung4-manual-conditioned-agent]] §3)
  carries up every rung: any claimed reading/goal gain must survive held-out swap/ablation.
- **Moat-aligned:** Crafter is a sample-efficiency benchmark (~1M-step budgets), so reading-as-
  efficiency plays *to* the home-lab compute moat, not against it.
- This is a **multi-month north star**, not a next-experiment. It refines the vague Phase-3 exit
  criterion ("DreamerV3-class score comparison") into a competence definition.
- **Naming note:** the reading/manual work is sometimes called "rung-4" in the rtfm design docs
  ([[rung4-manual-conditioned-agent]]); that is the *mechanism* for completing **rung-3 (Crafter)
  mastery**, distinct from the env-ladder's rung-4 (3D). No rename proposed here, just the
  clarification.

## Links

[[0003-environment-ladder]] · [[environment-ladder]] · [[rung4-manual-conditioned-agent]] · [[hierarchy-and-credit]] · [[0036-rtfm-shaping-ignition]] · [[0040-rtfm-actor-conditioning]] · [[no-hardcoded-env]]
