---
status: stub
owner: agent
scope: local
sources: [arxiv:2301.04104]
verified: false
last_reviewed: 2026-06-12
---

# DreamerV3: Mastering Diverse Domains through World Models (Hafner et al., 2023)

**Lab:** DeepMind / U Toronto · **Code:** https://github.com/danijar/dreamerv3 · **Read state:** abstract-only

## One-paragraph summary

RSSM world model + actor-critic trained on imagined rollouts, with robustness tricks
(symlog predictions, two-hot return targets, KL balancing, free bits) that let a single
configuration work across 150+ tasks — first agent to collect Minecraft diamonds from
scratch without human data. Later published in Nature (2025). The reference
general-purpose world-model RL algorithm and our designated baseline.

## Relevance to our experiments

Ladder rungs 2–4 baseline. Also a parts bin: symlog and two-hot are reusable in any of
our training loops regardless of architecture choice.

## Links

[[imagination-training]] · [[dreamer4-2025]] · [[environment-ladder]]
