---
status: draft
owner: agent
scope: local
sources: []
verified: false   # design brainstorm; literature ids in QUEUE, verify at ingest
last_reviewed: 2026-06-13
---

# Hierarchy & Credit Assignment: from flat value to emergent subgoals

## The question (Dave, 2026-06-13)

DoorKey pays reward only at the green goal; key-pickup and door-opening yield
nothing. Is the agent's solution therefore "random / acausal"? Will causality
emerge? LeCun's layered abstraction suggests a high-level goal (reach green)
decomposing into context-conditional subgoals (get key → open door → goal, the
first two needed ONLY if a door blocks). We must NOT hardcode the sequence — it
has to be learned.

## What we already have (don't mistake it for nothing)

The agent does NOT act from rewards; it acts from the **value head**, which is the
credit-assignment mechanism. MC returns-to-go propagate backward from rare goal
successes, so the belief "holding key, near door" acquires high value despite zero
local reward. Direct evidence: exp 0004's planning diagnostic — imagined return
rose monotonically with proximity to goal (≈ −0.001 far → +0.027 near) WITHOUT
subgoal supervision. This implicit, local credit is **already working** (drives the
63% result) and improves with training + encoder-freeze stability.

## What does NOT emerge from a flat value function

Explicit, reusable, compositional subgoals. A flat V(s) is a smooth scalar field; it
never factors into discrete "subgoal achieved" events. So the flat agent cannot name
"get a key", transfer "open a door" to a new context, or plan abstractly over
subgoals. This is the H-JEPA / options gap — and it is the part that needs new
architecture, not just more training.

| question | answer |
|---|---|
| Will the agent reliably do key→door→goal? | Largely YES already (value gradient; 63%) |
| Will it learn "get key" as a reusable, nameable subgoal? | NO with flat architecture — needs explicit temporal abstraction |

## Unsupervised, context-conditional subgoal emergence (no hardcoding)

Honoring Dave's constraint (subgoals needed only when the door blocks → must be
discovered, latent, conditional):

1. **Bottleneck discovery** — subgoals = states most successful trajectories funnel
   through (key-possession is a graph bottleneck). Find via successor-centrality or
   prediction-error spikes ([[temporal-abstraction]], Zacks event segmentation).
   Conditional for free: no door → no bottleneck.
2. **Manager–worker (Director/feudal)** — manager emits a latent subgoal belief;
   worker rewarded for reaching it; manager rewarded by real sparse reward.
   Subgoals are end-to-end-learned latent vectors. Director (Hafner 2022, QUEUE)
   does exactly this inside a Dreamer world model.
3. **Empowerment / intrinsic motivation** — key-possession is valuable INDEPENDENT
   of the goal (it expands reachable states). Would make key-seeking emerge before
   any goal is reached — also a fresh angle on our ignition problem.

## The synthesis (design decision in the making)

The Mode-1 actor we plan to distill (Crafter prerequisite) should likely be
**hierarchical (manager-worker), not flat**. One choice unifies three threads:
- the **actor** (rung-3 prerequisite),
- the **hierarchy/credit** decomposition (this page),
- subgoals == the "events" of [[temporal-abstraction]] == the units text tutorials
  name ([[language-grounding]]) — the shared docking point for the language thread.

## Strategic framing

Flat credit suffices for DoorKey (proven). Hierarchy becomes NECESSARY at Crafter
(~10-level tech tree, subgoals reused across procedural worlds, horizons too long
for flat MC variance). So this is not a detour from the Crafter pivot — it is a
central rung-3 design question, to be folded into the actor work rather than run
separately.

## Links

[[temporal-abstraction]] · [[language-grounding]] · [[agent-architecture]] ·
[[lecun-2022-path]] · [[imagination-training]] · [[value-equivalent-planning]] ·
exp 0004 (value-gradient evidence) · exp 0006 (value head carries beyond-horizon credit)
