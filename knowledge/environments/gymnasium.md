---
status: draft
owner: agent
scope: local
sources: [web:gym-docs]
verified: false
last_reviewed: 2026-06-12
---

# Gymnasium

## What it is

Farama's maintained fork of OpenAI Gym (https://gymnasium.farama.org) — the standard RL
environment API: `reset()/step()` with `(obs, reward, terminated, truncated, info)`,
typed observation/action spaces, wrapper system, vectorized envs.

## Why it matters here

Our environment abstraction layer. Everything on the [[environment-ladder]] speaks this
API (MiniGrid natively; Atari via `ale-py`; dm_control via shimmy), so models and
training code stay environment-agnostic. Use `gymnasium.vector` for parallel data
collection when collection becomes the bottleneck.

## Links

[[minigrid]] · [[environment-ladder]]
