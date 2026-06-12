---
status: draft
owner: agent
scope: local
sources: []
verified: false   # design-space page; literature ids in QUEUE, verify at ingest
last_reviewed: 2026-06-12
---

# Language → World Model: the Binding-and-Installation Problem

## What it is

The text-signal goal (ADR [[0005-minecraft-milestone]]) requires two capabilities
our agent lacks entirely (formulated by Dave, 2026-06-12):

1. **Binding**: map text entities ("furnace") onto visual/latent concepts learned
   from play — needs a shared or aligned embedding geometry (VL-JEPA/CLIP
   question).
2. **Installation**: convert a *declarative* statement ("smelting ore yields
   ingots") into a *dynamics belief* — a new edge in the transition model,
   composed from existing latent nodes, WITHOUT having experienced the transition.
   Today all knowledge enters via gradients on experienced transitions; text
   demands knowledge that arrives before experience.

## The three installation architectures (testable spectrum)

| # | mechanism | "new model" lives in | nearest literature | cost |
|---|---|---|---|---|
| 1 | **text-as-context** — tutorial embedding carried in the belief state; dynamics head conditions on it | in-context (no weight change) — world-model ICL | Dynalang; EMMA's entity attention (Messenger) | low — first Messenger experiment |
| 2 | **text-as-data** — text induces imagined rollouts; world model trains on its own text-conditioned dreams | gradients, via synthetic experience | (apparently unexplored in latent WMs — original-idea candidate, run wall protocol before claiming) | medium |
| 3 | **text-as-weights** — hypernetwork: tutorial → Δθ of dynamics head | explicit fast weights | Schmidhuber 1991 fast-weight programmers ([[intellectual-lineage]]); knowledge-editing (ROME/MEMIT) | high |

Prediction worth pre-registering when the thread opens: (1) suffices for
Messenger-scale (manual fits in context); (2)/(3) become necessary when tutorials
exceed context or must persist across episodes (Minecraft wiki scale).

## Why it matters here

This is the staircase's scientific payload — (i) Messenger/RTFM, (ii) text-Crafter,
(iii) Minecraft — and the place where our JEPA thread (aligned latent spaces) meets
the agent thread. Also interacts with [[retention]]: installation method 2 trains
on synthetic data — another distribution-shift source the flywheel lessons apply to.

## Links

[[0005-minecraft-milestone]] · [[intellectual-lineage]] · [[retention]] ·
[[jepa]] · [[agent-architecture]]
