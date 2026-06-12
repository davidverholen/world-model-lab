---
status: draft
owner: agent
scope: local
sources: [arxiv:1803.10122, openreview:lecun-path, arxiv:2506.01622, arxiv:2411.14499]
verified: false
last_reviewed: 2026-06-12
---

# World Models

## What it is

A world model is a learned predictive model of an environment: given the current state
(or a latent representation of it) and an action, it predicts what happens next. The
research bet — argued most forcefully by LeCun (openreview:lecun-path) — is that
predicting *world states* rather than tokens/pixels is the path to agents that
understand, plan, and generalize. Instead of "what word comes next?", the question is
"what does the world look like after I act?".

## Why it matters here

This project's goal is autonomous game-playing agents that learn from their own
experience: collect trajectories → learn a model of the game's dynamics → use the model
(planning or imagination training) to act better, climbing the [[environment-ladder]]
from MiniGrid toward more complex games and eventually real-world transfer.

## The main families

- **Latent-prediction (JEPA)** — predict the *representation* of the future, never
  reconstruct observations. See [[jepa]]. Champion: Meta FAIR / [[ami-labs]].
- **Latent dynamics + imagination (Dreamer/RSSM)** — learn a recurrent latent dynamics
  model with reconstruction, train an actor-critic on imagined rollouts. See
  [[imagination-training]]. Champion: Hafner et al. ([[dreamerv3-2023]], [[dreamer4-2025]]).
- **Value-equivalent (MuZero, TD-MPC2)** — the model only needs to predict what matters
  for decisions (reward/value/policy), not observations. See [[value-equivalent-planning]].
- **Generative video world models (Genie, GAIA, Cosmos, DIAMOND)** — large generative
  models of pixels/tokens used as simulators or neural game engines; the
  "scale-first" branch.

Theory note: Richens et al. (arxiv:2506.01622) argue any agent that generalizes to
long-horizon goals must implicitly contain a world model — so the question is not
*whether* to have one but whether to learn it explicitly.

## Open questions

- Which family wins at our scale (one consumer GPU, simple games)? Likely
  Dreamer-style for control performance, JEPA-style for representation quality —
  experiments should compare both on MiniGrid.
- What's the cheapest setup where a learned model demonstrably beats model-free RL?

## Links

[[jepa]] · [[imagination-training]] · [[value-equivalent-planning]] ·
[[environment-ladder]] · [[ami-labs]] · [[world-models-1803.10122]] (paper page, queued)
