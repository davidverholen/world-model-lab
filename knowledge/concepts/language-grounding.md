---
status: draft
owner: agent
scope: local
sources: []
verified: false   # design-space page; literature ids in QUEUE, verify at ingest
last_reviewed: 2026-06-12
---

# Language → World Model: the Binding-and-Installation Problem

## What it is

The text-signal goal (ADR [[0005-minecraft-milestone]]) requires two capabilities
our agent lacks entirely (formulated by Dave, 2026-06-12):

1. **Binding**: map text entities ("furnace") onto visual/latent concepts learned
   from play — needs a shared or aligned embedding geometry (VL-JEPA/CLIP
   question).
2. **Installation**: convert a *declarative* statement ("smelting ore yields
   ingots") into a *dynamics belief* — a new edge in the transition model,
   composed from existing latent nodes, WITHOUT having experienced the transition.
   Today all knowledge enters via gradients on experienced transitions; text
   demands knowledge that arrives before experience.

## The three installation architectures (testable spectrum)

| # | mechanism | "new model" lives in | nearest literature | cost |
|---|---|---|---|---|
| 1 | **text-as-context** — tutorial embedding carried in the belief state; dynamics head conditions on it | in-context (no weight change) — world-model ICL | Dynalang; EMMA's entity attention (Messenger) | low — first Messenger experiment |
| 2a | **text-as-data, own dreams** — world model imagines rollouts from text; trains on them | gradients, via synthetic experience | (apparently unexplored in latent WMs — original-idea candidate, run wall protocol before claiming) | medium — but circular: requires text understanding first |
| 2b | **text-as-data, external imagination engine** (Dave, 2026-06-12) — text → domain video generator (Oasis / Dreamer-4-WM / Genie-3 class) → IDM action labels (VPT) → replay as trust-weighted synthetic experience → **corroboration gate**: real play validates/reinforces or discounts ("verstärkt oder nicht") | gradients, via externally generated experience | Oasis, Cosmos (synthetic-data platform), VPT IDM; gate ≈ MBPO's model-trust lessons | medium; breaks 2a's circularity — the external model carries the text grounding; unifies ADR-0005 signals 2+3 (text rides the video path) |
| 3 | **text-as-weights** — hypernetwork: tutorial → Δθ of dynamics head | explicit fast weights | Schmidhuber 1991 fast-weight programmers ([[intellectual-lineage]]); knowledge-editing (ROME/MEMIT) | high |

Prediction worth pre-registering when the thread opens: (1) suffices for
Messenger-scale (manual fits in context); (2)/(3) become necessary when tutorials
exceed context or must persist across episodes (Minecraft wiki scale).

The cognitive analogy (Dave): humans convert text into mental imagery/simulation
([[intellectual-lineage]]: Craik's "try it in the head"), which becomes belief only
through corroboration against experience. Human imagination is not exact simulation
either — it works *because of* constant verification and reinforcement, not despite
inexactness. So hallucination is not disqualifying; the gate IS the mechanism.

**Unified trust-weighted replay (Dave, 2026-06-12):** every transition carries a
trust weight w. The SOURCE sets the prior (own validated play: high; others'
IDM-labeled video: medium; generated video: low; own dreams: lowest); CORROBORATION
updates it (consistency with verified experience raises w, contradiction lowers it);
training loss scales with w. "Real" vs "imagined" stops being categorical — just
different priors on one scale; imagination must earn the weight real experience
gets at birth. Engineering neighbors: prioritized replay (different objective),
MBPO's model-trust lessons, Bayesian source priors. Note the symmetry with this
KB's own epistemics (sources enter unverified, trust is earned by verification).
Cheap prototype (pre-Minecraft): inject deliberately corrupted synthetic
transitions into Crafter replay with low priors; verify the gate discounts them
and clean synthetic data earns weight. Gate failure modes are
[[retention]]-adjacent (synthetic data = another distribution shift).

## Why it matters here

This is the staircase's scientific payload — (i) Messenger/RTFM, (ii) text-Crafter,
(iii) Minecraft — and the place where our JEPA thread (aligned latent spaces) meets
the agent thread. Also interacts with [[retention]]: installation method 2 trains
on synthetic data — another distribution-shift source the flywheel lessons apply to.

## Links

[[0005-minecraft-milestone]] · [[intellectual-lineage]] · [[retention]] ·
[[jepa]] · [[agent-architecture]]
