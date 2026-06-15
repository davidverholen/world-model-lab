---
status: draft
owner: world-model
scope: local
sources: [arxiv:1911.08265, arxiv:2310.16828]
verified: false
last_reviewed: 2026-06-12
---

# Value-Equivalent Planning (MuZero, TD-MPC2)

## What it is

A third way between reconstruction (Dreamer) and latent self-prediction (JEPA): the
learned model only has to be accurate about *decision-relevant* quantities — reward,
value, policy — not about observations. MuZero (arxiv:1911.08265) plans with MCTS over
such a model and mastered Go/Chess/Atari. TD-MPC2 (arxiv:2310.16828) is the modern
continuous-control version: decoder-free implicit world model + model-predictive
control, one hyperparameter set across 100+ tasks.

## Why it matters here

For game-playing specifically, value-equivalence is a serious contender: games have
clear rewards, and we may not need visually faithful prediction at all. It also defines
the falsifiable question for our representation experiments: do JEPA-style latents
actually help *control*, or is decision-relevant prediction all you need?

## Open questions

- MCTS (discrete, MiniGrid-friendly) vs CEM/MPPI (continuous) as our first planner?
- How badly does value-equivalence generalize to *new* tasks in the same world,
  where JEPA/Dreamer latents should transfer better?

## Links

[[world-models]] · [[imagination-training]] · [[jepa]]
