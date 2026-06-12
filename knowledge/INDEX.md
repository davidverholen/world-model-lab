# Knowledge Base Index

One line per page. Updated on every ingest. Conventions: `_schema/SCHEMA.md`,
process: `_schema/PROCESS.md`, raw sources: `sources/SOURCES.md`, backlog: `sources/QUEUE.md`.

## Concepts

- [world-models](concepts/world-models.md) — root page: what world models are, the four families, why they're our bet (draft)
- [jepa](concepts/jepa.md) — latent-prediction architectures I-JEPA→V-JEPA 2→LeJEPA; collapse problem and fixes (draft)
- [imagination-training](concepts/imagination-training.md) — Dreamer lineage: RSSM, actor-critic in imagination, Dreamer 4 offline (draft)
- [value-equivalent-planning](concepts/value-equivalent-planning.md) — MuZero/TD-MPC2: model only what decisions need (draft)
- [environment-ladder](concepts/environment-ladder.md) — our staged env progression with exit criteria (current)
- [latent-collapse](concepts/latent-collapse.md) — the JEPA failure mode, fixes table, measurement pitfalls (draft)
- [compute-strategy](concepts/compute-strategy.md) — predicted local-vs-desktop-vs-rented GPU trade-offs per rung (draft, unvalidated)

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

- [0001-naive-latent-regression](experiments/0001-naive-latent-regression.md) — collapse baseline confirmed: latent_std 0.0003, probe R² −0.49 (done)
- [0002-sigreg-anti-collapse](experiments/0002-sigreg-anti-collapse.md) — SIGReg λ=0.05 prevents collapse (latent_std 0.74, probe R² 0.25); probe-protocol pitfalls documented (done)
- [0003-probe-protocol-8x8](experiments/0003-probe-protocol-8x8.md) — rung-1 exit: dynamics 4–5× better than copy baseline (3 seeds); linear probe demoted to diagnostic (done)
- [0004-mpc-agent-empty8x8](experiments/0004-mpc-agent-empty8x8.md) — first acting agent: latent CEM-MPC 19/20 vs random 3/20; multi-step training + reward calibration lessons (done)
- [0005-doorkey-memory](experiments/0005-doorkey-memory.md) — negative: belief-MPC 0/20 on DoorKey; but collection flywheel showed 8.5%→14.3%, and pure MPC's horizon limit identified → value head next (done)

## Wanted pages (linked but not yet written)

- world-models-1803.10122 — Ha & Schmidhuber paper page (queued for ingest)
