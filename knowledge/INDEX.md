# Knowledge Base Index

One line per page. Updated on every ingest. Conventions: `_schema/SCHEMA.md`,
process: `_schema/PROCESS.md`, raw sources: `sources/SOURCES.md`, backlog: `sources/QUEUE.md`.

## Concepts

- [world-models](concepts/world-models.md) — root page: what world models are, the four families, why they're our bet (draft)
- [jepa](concepts/jepa.md) — latent-prediction architectures I-JEPA→V-JEPA 2→LeJEPA; collapse problem and fixes (draft)
- [imagination-training](concepts/imagination-training.md) — Dreamer lineage: RSSM, actor-critic in imagination, Dreamer 4 offline (draft)
- [value-equivalent-planning](concepts/value-equivalent-planning.md) — MuZero/TD-MPC2: model only what decisions need (draft)
- [environment-ladder](concepts/environment-ladder.md) — our staged env progression with exit criteria (current)

## Papers

- [vjepa2-2025](papers/vjepa2-2025.md) — V-JEPA 2 + action-conditioned variant, zero-shot robot planning (stub)
- [lejepa-2025](papers/lejepa-2025.md) — provable JEPA objective (SIGReg); companion world-model theory paper (stub)
- [dreamerv3-2023](papers/dreamerv3-2023.md) — reference world-model RL baseline, one config for 150+ tasks (stub)
- [dreamer4-2025](papers/dreamer4-2025.md) — offline imagination training at scale, Minecraft diamonds (stub)

## Labs

- [ami-labs](labs/ami-labs.md) — LeCun's world-model startup: $1.03B raise, LeJEPA program (draft)

## Environments

- [minigrid](environments/minigrid.md) — rungs 1–2: fast gridworlds, our wrapper notes (draft)
- [gymnasium](environments/gymnasium.md) — the env API everything targets (draft)

## Decisions (ADRs)

- [0001-pytorch-over-jax](decisions/0001-pytorch-over-jax.md) — PyTorch default (accepted)
- [0002-knowledge-architecture](decisions/0002-knowledge-architecture.md) — Karpathy wiki × context architecture (accepted)
- [0003-environment-ladder](decisions/0003-environment-ladder.md) — MiniGrid first, no rung-skipping (accepted)

## Experiments

- [0001-naive-latent-regression](experiments/0001-naive-latent-regression.md) — reproduce latent collapse on purpose (planned)

## Wanted pages (linked but not yet written)

- latent-collapse — referenced by jepa, imagination-training, experiment 0001
- world-models-1803.10122 — Ha & Schmidhuber paper page (queued for ingest)
