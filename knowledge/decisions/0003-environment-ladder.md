---
status: current
owner: world-model
scope: local
sources: []
verified: true
last_reviewed: 2026-06-12
---

# 0003: Climb an environment ladder, MiniGrid first

**Status:** accepted (2026-06-12)

## Context

Goal: autonomous game-playing agents that learn world models from their own play, with
eventual real-world transfer. Risk: jumping to impressive environments (Atari,
Minecraft) before the basic latent-prediction machinery is understood, burning compute
on unattributable failures.

## Decision

Progress strictly along the ladder defined in [[environment-ladder]]: MiniGrid full-obs
→ MiniGrid partial-obs/DoorKey → Crafter → Atari100k subset → continuous/3D → real-world
transfer. Each rung has an explicit exit criterion; no rung-skipping.

## Consequences

- First experiments are unglamorous (5x5 gridworlds) but every component is verifiable.
- The 8 GB GPU stays sufficient through rung 3 at least.
- Real-world ambitions stay parked until the ladder earns them.

## Links

[[environment-ladder]] · [[0001-pytorch-over-jax]]
