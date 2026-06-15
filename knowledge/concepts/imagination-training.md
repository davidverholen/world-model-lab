---
status: draft
owner: agent
scope: local
sources: [arxiv:1811.04551, arxiv:2301.04104, arxiv:2509.24527, arxiv:1803.10122]
verified: true
last_reviewed: 2026-06-15
---

# Imagination Training (Dreamer lineage)

## What it is

Train a latent dynamics model of the environment, then train the *policy inside the
model*: roll out imagined trajectories in latent space and run actor-critic on them.
The real environment is only used to collect data and ground the model. Origin: Ha &
Schmidhuber's "training inside the dream" (arxiv:1803.10122); industrialized by the
Dreamer line.

Lineage: PlaNet introduces the RSSM — recurrent state-space model with deterministic
GRU path + stochastic latents (arxiv:1811.04551) → Dreamer v1 (actor-critic through
imagination) → DreamerV2 (discrete/categorical latents) → DreamerV3: one configuration
across 150+ tasks via robustness tricks (symlog targets, two-hot returns, free bits);
first to get Minecraft diamonds from scratch (arxiv:2301.04104, later in Nature) →
Dreamer 4: 2B-parameter block-causal transformer world model replacing the RSSM;
trained with "shortcut forcing" (diffusion with step-size conditioning + x-prediction
to prevent denoising shortcuts; 4 steps ≈ 64-step quality → 21 FPS on one H100);
agent trained purely *offline* by RL inside the model from a fixed video dataset;
first offline-dataset agent to obtain Minecraft diamonds, beating VPT with 100× less
data; policy stabilized via KL constraint to BC prior (RLHF-style) rather than
critic-on-replay (arxiv:2509.24527, verified; see [[dreamer4-2025]] for method depth).

## Why it matters here

DreamerV3 is the reference algorithm for exactly our use case: sample-efficient
learning of games from pixels on modest hardware. It is the strongest baseline to
benchmark any JEPA-flavored alternative against, and its tricks (symlog, KL balancing,
free bits) are reusable in our own training loops. Dreamer 4's offline result matters
for the endgame: learning to act without expensive environment interaction is the
bridge to real-world transfer where interaction is costly.

Contrast with [[jepa]]: Dreamer keeps a decoder (reconstruction grounds the latents,
sidestepping [[latent-collapse]]) but pays for it by modeling pixel detail; JEPA drops
the decoder but needs anti-collapse machinery.

## Open questions

- Smallest DreamerV3 config that solves MiniGrid DoorKey? (There are public configs
  for small tasks — check danijar.com/dreamerv3 at ingest time.)
- Can we swap the RSSM's reconstruction loss for a LeJEPA objective and keep
  imagination training working? That hybrid would be a genuinely interesting experiment.

## Links

[[world-models]] · [[jepa]] · [[value-equivalent-planning]] · [[environment-ladder]] ·
[[dreamerv3-2023]] · [[dreamer4-2025]]
