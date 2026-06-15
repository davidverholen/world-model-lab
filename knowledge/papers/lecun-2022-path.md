---
status: draft
owner: world-model
scope: local
sources: [openreview:lecun-path]
verified: true
last_reviewed: 2026-06-12
---

# A Path Towards Autonomous Machine Intelligence (LeCun, 2022, v0.9.2)

**Venue:** OpenReview position paper (title/author/abstract verified 2026-06-12)
· **Read state:** skimmed — abstract + secondary literature; full-PDF read pending
(claims below are well-established summaries, but promote to `current` only after
a full read)

## One-paragraph summary

A deliberately *compositional* proposal (it synthesizes ideas from decades of work
rather than reporting one result): autonomous intelligence should be built as a
modular cognitive architecture — **perception** (encode observations), **world
model** (predict consequences in representation space; the centerpiece, proposed as
a hierarchical JEPA predicting at multiple abstraction levels and timescales),
**cost module** (hard-wired intrinsic costs + trainable critic), **actor**,
**short-term memory**, and a **configurator** that orchestrates the modules per
task. Behavior runs in two modes: **Mode-1** (fast reactive policy) and **Mode-2**
(deliberate planning by optimizing action sequences through the world model). The
whole stack is framed energy-based, trained largely self-supervised, with JEPA
(predict latents, not pixels) as the anti-collapse, abstraction-friendly choice.

## Relevance to our experiments

This is our project's umbrella program, instantiated bottom-up:

| LeCun module | our component | status |
|---|---|---|
| perception | ConvEncoder (L2) | from-scratch; frozen-DINOv3 candidate at rung 3 |
| world model | GRU belief + next-latent head (L2) | deterministic; hierarchy/stochastic latents future |
| cost + critic | reward head + MC value head | TD/intrinsic cost future |
| actor (Mode-1) | — (missing) | = the planned policy distillation |
| Mode-2 planning | CEM over imagined rollouts (L1) | working |
| configurator | — (out of scope) | — |
| short-term memory | belief state (cheap version) | episodic memory future |

What the paper does NOT cover — and where our experiments actually live — is the
*training dynamics* of such a system under continual data collection ([[retention]]).

## Limitations / critiques

Position paper: no experiments, no training recipes; H-JEPA hierarchy remains
largely unrealized publicly (LeJEPA/V-JEPA 2 are partial steps). Mode-2→Mode-1
amortization is asserted, not demonstrated.

## Links

[[agent-architecture]] · [[jepa]] · [[world-models]] · [[lejepa-2025]] ·
[[vjepa2-2025]] · [[retention]] · [[ami-labs]]
