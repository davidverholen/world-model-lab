---
status: current
owner: world-model
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
| 2 | MiniGrid DoorKey, partial view | sparse reward, partial observability, memory | 2a ✅ **met 2026-06-12**: DoorKey-5x5 solved under partial obs ([[0006-value-head-doorkey]]); 2b open: beat model-free (PPO) on sample efficiency *on the smallest env where PPO struggles at budget* — split approved 2026-06-12 after [[0007-ppo-baseline]] |
| 3 | Crafter | long horizon, many subgoals, procgen | nontrivial achievement count within 1M steps |
| 4 | Atari 100k subset | pixels at scale, diverse dynamics | competitive with published WM agents on ≥3 games |
| 5 | Continuous control (dm_control) or 3D (Miniworld) | continuous actions / 3D | TBD |
| 5b | **Minecraft** ([[0005-minecraft-milestone]]) | multi-modal training: self-play + action-labeled video (VPT-style) + text-in-world-model (open research); long horizons at scale | diamond at a published sample-efficiency tier + task-suite breadth (refine at entry) |
| 6 | Real-world transfer (sim2real, robot or camera feed) | reality gap | TBD — after 5b |

## Why it matters here

The ladder *is* the project roadmap (see ROADMAP.md at repo root). It keeps experiment
scope honest: no rung-skipping; a technique that needs rung-4 compute to show value
doesn't belong in rung-1 experiments.

**Rung 3 → 4/5b is proxy → real target** (added 2026-06-13, full writeup in [[crafter]]):
Crafter is a *fast Minecraft-shaped proxy* (it exists because real Minecraft is too slow to
iterate on). Across the gap, the **method/recipe** and the **frozen encoder** transfer (the
encoder literally — it's game-agnostic; [[frozen-encoder-lean]]); **world-model weights do
not** (game-specific, retrained). New signals (VPT video, MineDojo wiki/tutorials) switch on
at Minecraft. Corollary: real envs can't be JAX-fused, so Crafter is the *last* rung where the
Craftax speedup could even apply.

## Links

[[minigrid]] · [[gymnasium]] · [[world-models]] · [[0003-environment-ladder]]
