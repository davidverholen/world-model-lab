---
status: draft
owner: agent
scope: local
sources: []
verified: false   # ids in QUEUE; verify at ingest
last_reviewed: 2026-06-12
---

# Temporal Abstraction: Events, Not Ticks

## What it is

The critique (raised by the maintainer via Stephen E. Robbins, Bergson lineage): current
world models — ours included — factorize experience as z_{t+1} = f(z_t, a_t) at a
fixed tick. A real *event* ("stirring coffee") is (a) temporally extended with
internal unity, (b) driven by an extended action (a motor policy with feedback,
not a keypress), (c) multimodally bound (sight/sound/feel as one). Flip-book
models represent none of that directly. The Bergsonian core (lived duration ≠
sequence of snapshots) survives any discretization; the engineering bet (LeCun,
and the maintainer's synthesis) is that a HIERARCHY of timescales captures enough: high
levels treat whole time-series as units, low levels keep frame detail.

## The converging research lines (rarely cite each other)

| field | mechanism | the "event" is... |
|---|---|---|
| RL: **options** (Sutton-Precup-Singh 1999) | policy + termination as one action | an extended action |
| LeCun: **H-JEPA** ([[lecun-2022-path]]) | multi-timescale latent prediction | an abstract state transition |
| Dreamer line: **Director** (Hafner 2022) | manager sets latent subgoals, worker executes | a subgoal episode |
| Cog-sci: **event segmentation theory** (Zacks) | humans chunk at prediction-error spikes | a surprise-bounded segment |
| Robotics: **action chunking** (ACT) | predict action sequences as units | a motor chunk |

The Zacks mechanism is directly implementable in our stack: we already compute
per-step prediction error; segment where it spikes → boundaries for free → the
segments become the higher level's alphabet. Options *discovered from* world-model
surprise rather than hand-designed.

## Why it matters here

1. **The deep fix for horizon blindness** (exp 0005): planning over events
   multiplies effective horizon by event length — DoorKey is 3 events vs 25+
   ticks. The value head was the patch; this is the structure.
2. **Multimodal binding** (sound/feel into events) is the same shared-latent
   problem as [[language-grounding]]'s binding — one geometry, multiple payoffs;
   text tutorials naturally describe EVENTS ("smelt the ore"), not ticks — the
   abstraction levels must meet for installation to work.
3. Honest scope note: our GRU belief already integrates history (a weak
   "duration"), but actions remain ticks. Cheap first probe at rung 2–3:
   prediction-error segmentation on DoorKey/Crafter trajectories — do the
   discovered boundaries align with human-obvious events (key pickup, door)?
   Diagnostic only, before any hierarchical agent.

## Links

[[lecun-2022-path]] · [[language-grounding]] · [[agent-architecture]] ·
[[imagination-training]] · [[intellectual-lineage]] (Bergson joins Craik in the
mental-models thread) · exp 0005 (horizon)
