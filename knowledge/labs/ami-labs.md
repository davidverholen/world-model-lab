---
status: draft
owner: agent
scope: local
sources: [news:ami-funding, arxiv:2511.08544, arxiv:2605.26379, openreview:lecun-path]
verified: true
last_reviewed: 2026-06-12
---

# AMI Labs (Advanced Machine Intelligence)

## What it is

Yann LeCun's startup, founded after he left Meta (announced Nov 2025). Official launch
March 2026 with a $1.03B raise at $3.5B pre-money (co-led by Cathay Innovation,
Greycroft, Hiro Capital, HV Capital, Bezos Expeditions). Explicit thesis: world models
over LLMs — JEPA-based systems that learn from reality, targeting industrial, robotic,
and healthcare applications. Committed to publishing research.

## Research output so far (as of 2026-06-12)

- **LeJEPA** (arxiv:2511.08544, Nov 2025, Balestriero & LeCun — pre-dates the company
  but defines its technical core): provable JEPA training objective. See [[lejepa-2025]].
- **Late-May 2026 preprint pair** from LeCun's research circle — "a theorem and a
  stress test": (1) "When Does LeJEPA Learn a World Model?" (arxiv:2605.26379)
  proves linear identifiability — LeJEPA linearly recovers the world's latent
  variables from nonlinear observations, for worlds with stationary additive-noise
  transitions; (2) **WBench** (arxiv:2605.25874, id confirmed 2026-06-12) —
  interactive video world-model benchmark (5 dimensions, 289 cases) whose headline
  finding is that current models collapse under minor visual shifts. Together: the
  destination is provable, the field isn't there yet.
- Intellectual foundation: "A Path Towards Autonomous Machine Intelligence"
  (openreview:lecun-path) — H-JEPA, energy-based models, configurator, intrinsic cost.

## Why it matters here

AMI Labs is the strongest institutional bet on this project's premise. Their
publications are our primary watch-list (skill: `/scout-sources`); LeJEPA's
exploration-breadth condition is already a design input for our data collection.

## Links

[[jepa]] · [[lejepa-2025]] · [[world-models]] · [[vjepa2-2025]] (Meta-era lineage)
