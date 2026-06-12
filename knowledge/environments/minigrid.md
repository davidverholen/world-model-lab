---
status: draft
owner: agent
scope: local
sources: [web:minigrid-docs]
verified: false
last_reviewed: 2026-06-12
---

# MiniGrid

## What it is

Farama's collection of tiny, fast, procedurally generated gridworld environments
(https://minigrid.farama.org). Discrete 7-action space (turn, forward, pickup, drop,
toggle, done), sparse rewards, optional egocentric partial observation, language
mission strings. Runs thousands of steps/second on CPU.

## Why it matters here

Ladder rungs 1–2. Cheap enough to iterate on world-model architecture in minutes;
DoorKey + partial observability is the smallest setting where prediction and memory
genuinely matter. Our wrapper (`world_model.envs.make_minigrid_env`) emits CHW float
RGB; fully-observable for rung 1, partial view (`fully_observable=False`) for rung 2.

Useful env ids: `MiniGrid-Empty-5x5-v0`, `MiniGrid-Empty-8x8-v0`,
`MiniGrid-DoorKey-6x6-v0`, `MiniGrid-MultiRoom-N2-S4-v0`, `MiniGrid-KeyCorridorS3R1-v0`.
3D sibling when we need it: Miniworld (https://miniworld.farama.org).

## Open questions

- Linear-probe targets available from env internals (agent pos, direction, carrying):
  wire them up as representation-quality metrics.

## Links

[[gymnasium]] · [[environment-ladder]]
