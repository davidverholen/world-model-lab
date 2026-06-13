---
status: draft
owner: agent
scope: local
sources: []
verified: false   # backlog concept (maintainer, 2026-06-13); Hoel ref verified, queued for ingest
last_reviewed: 2026-06-13
---

# Dreaming: relaxed-constraint generation as augmentation + creativity

Backlog concept (maintainer, 2026-06-13). Possibly unifies the creativity thread
([[generative-vs-predictive]] §novelty, [[capability-map]] #7) and the retention
thread ([[retention]]).

## The maintainer's model of dreaming

Two properties: (1) the established world model is applied LESS STRICTLY (relaxed
constraints → wild recombination, "purely creative"); (2) dream results are written
back to the persistent mental model VERY WEAKLY (low trust). I.e. high-temperature /
weak-conditioning generation + near-zero write-back.

## Maps onto the Overfitted Brain Hypothesis (Hoel 2021, arXiv:2007.09560, Patterns)

Verified 2026-06-13. Hoel: dreams = DATA AUGMENTATION to reduce overfitting. They are
deliberately less detailed and more fantastical (= the maintainer's relaxed constraints) to
pull representations AWAY from overfitting narrow waking experience; not written as
fact (= weak write-back). Striking: **the most reliable way to trigger a dream about
something is to OVER-TRAIN on a novel task** — i.e. dreaming is a response to the
exact overfitting we call primacy bias ([[retention]], exps 0009-0015).

## Two functions in our system

1. **Creativity-seeding:** relaxed-constraint generation proposes novel
   recombinations → weakly flagged → verified while awake (variation+selection).
2. **Anti-overfitting augmentation (testable, no generative AI):** generate
   high-entropy, constraint-relaxed LATENT rollouts (not pixels), train WEAKLY on
   them as augmentation against primacy bias. A cheap MiniGrid-scale retention
   experiment that would test Hoel in our own agent. Contrast: pixel-dreaming (full
   generative video) is the richer, expensive version for the creativity/tutorial
   side.

## Status

Backlog. The latent-dream-augmentation version is a concrete candidate retention
regularizer (cheaper than, or complementary to, encoder-freeze) — promote to an
experiment if the actor/hierarchy thread stalls or retention needs a fresh angle.

## Links

[[generative-vs-predictive]] · [[retention]] · [[capability-map]] ·
[[language-grounding]] · [[intellectual-lineage]]
