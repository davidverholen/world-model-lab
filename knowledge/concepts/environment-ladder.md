---
status: current
owner: human
scope: local
sources: [web:minigrid-docs, web:gym-docs, arxiv:2109.06780]
verified: false
last_reviewed: 2026-06-12
---

# The Environment Ladder

## What it is

Our staged progression of environments, simplest first. Each rung adds exactly one
difficulty so failures are attributable. Accepted in [[0003-environment-ladder]] (ADR).

| Rung | Environment | New difficulty | Exit criterion |
|---|---|---|---|
| 1 | MiniGrid Empty-5x5 / 8x8 | none — pipeline bring-up | ✅ **met 2026-06-12** ([[0003-probe-protocol-8x8]]: no collapse, dynamics 4–5× better than copy baseline, 3 seeds) |
| 2 | MiniGrid DoorKey, partial view | sparse reward, partial observability, memory | model-based agent beats model-free baseline on sample efficiency |
| 3 | Crafter | long horizon, many subgoals, procgen | nontrivial achievement count within 1M steps |
| 4 | Atari 100k subset | pixels at scale, diverse dynamics | competitive with published WM agents on ≥3 games |
| 5 | Continuous control (dm_control) or 3D (Miniworld) | continuous actions / 3D | TBD |
| 6 | Real-world transfer (sim2real, robot or camera feed) | reality gap | TBD — far future |

## Why it matters here

The ladder *is* the project roadmap (see ROADMAP.md at repo root). It keeps experiment
scope honest: no rung-skipping; a technique that needs rung-4 compute to show value
doesn't belong in rung-1 experiments.

## Links

[[minigrid]] · [[gymnasium]] · [[world-models]] · [[0003-environment-ladder]]
