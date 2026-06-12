# Knowledge Base Index

One line per page. Updated on every ingest. Conventions: `_schema/SCHEMA.md`,
process: `_schema/PROCESS.md`, raw sources: `sources/SOURCES.md`, backlog: `sources/QUEUE.md`,
training assets: `sources/ASSETS.md`.

## Concepts

- [world-models](concepts/world-models.md) — root page: what world models are, the four families, why they're our bet (draft)
- [jepa](concepts/jepa.md) — latent-prediction architectures I-JEPA→V-JEPA 2→LeJEPA; collapse problem and fixes (draft)
- [imagination-training](concepts/imagination-training.md) — Dreamer lineage: RSSM, actor-critic in imagination, Dreamer 4 offline (draft)
- [value-equivalent-planning](concepts/value-equivalent-planning.md) — MuZero/TD-MPC2: model only what decisions need (draft)
- [environment-ladder](concepts/environment-ladder.md) — our staged env progression with exit criteria (current)
- [latent-collapse](concepts/latent-collapse.md) — the JEPA failure mode, fixes table, measurement pitfalls (draft)
- [compute-strategy](concepts/compute-strategy.md) — measured local-vs-desktop-vs-rented GPU trade-offs per rung; parallel-seeds corollary (draft)
- [agent-architecture](concepts/agent-architecture.md) — OUR system as 4 optimization layers (mermaid), per-layer failure/diagnosis table (current)
- [retention](concepts/retention.md) — plasticity loss/primacy bias/interference: fix families, literature↔our status (draft)
- [intellectual-lineage](concepts/intellectual-lineage.md) — 80 years of background in 4 threads: mental models, predictive brain, LeCun's arc, model-based RL; our flywheel = Dyna (draft, unverified ids)
- [language-grounding](concepts/language-grounding.md) — binding + installation: 3 architectures for text→world-model (context/data/weights); the text-staircase design space (draft)
- [temporal-abstraction](concepts/temporal-abstraction.md) — events-not-ticks: Robbins/Bergson critique × options/H-JEPA/event-segmentation; the deep horizon fix (draft)

## Papers

- [vjepa2-2025](papers/vjepa2-2025.md) — V-JEPA 2 + action-conditioned variant, zero-shot robot planning (stub)
- [lejepa-2025](papers/lejepa-2025.md) — provable JEPA objective (SIGReg); companion world-model theory paper (stub)
- [dreamerv3-2023](papers/dreamerv3-2023.md) — reference world-model RL baseline, one config for 150+ tasks (stub)
- [dreamer4-2025](papers/dreamer4-2025.md) — offline imagination training at scale, Minecraft diamonds (stub)
- [ijepa-2023](papers/ijepa-2023.md) — the first JEPA: multi-block masking, EMA-teacher anti-collapse (vs our SIGReg), method depth (draft)
- [nikishin-primacy-2022](papers/nikishin-primacy-2022.md) — primacy bias + reset protocol, implementation-grade (draft)
- [ma-plasticity-2024](papers/ma-plasticity-2024.md) — FAU module diagnostics; critic-bottleneck in model-free; frozen encoders suffice → legitimizes our fenc finding; Adaptive-RR lever (draft)
- [qiao-model-primacy-2023](papers/qiao-model-primacy-2023.md) — MBRL primacy bias lives in the WORLD MODEL; world-model resets help only at high UTD; confirms our 0011 negative; redesigned exp 0012 (draft)
- [lecun-2022-path](papers/lecun-2022-path.md) — the umbrella position paper; module↔our-layer mapping (draft, skim depth)

## Labs

- [ami-labs](labs/ami-labs.md) — LeCun's world-model startup: $1.03B raise, LeJEPA program (draft)

## Environments

- [minigrid](environments/minigrid.md) — rungs 1–2: fast gridworlds, our wrapper notes (draft)
- [gymnasium](environments/gymnasium.md) — the env API everything targets (draft)

## Decisions (ADRs)

- [0001-pytorch-over-jax](decisions/0001-pytorch-over-jax.md) — PyTorch default (accepted)
- [0002-knowledge-architecture](decisions/0002-knowledge-architecture.md) — Karpathy wiki × context architecture (accepted)
- [0003-environment-ladder](decisions/0003-environment-ladder.md) — MiniGrid first, no rung-skipping (accepted)
- [0004-remote-dispatch](decisions/0004-remote-dispatch.md) — SSH + git push over Tailscale to the 5070 Ti desktop; no orchestrator (accepted)
- [0005-minecraft-milestone](decisions/0005-minecraft-milestone.md) — rung 5b: Minecraft via self-play + action-labeled video + text staircase (accepted)

## Experiments

- [0001-naive-latent-regression](experiments/0001-naive-latent-regression.md) — collapse baseline confirmed: latent_std 0.0003, probe R² −0.49 (done)
- [0002-sigreg-anti-collapse](experiments/0002-sigreg-anti-collapse.md) — SIGReg λ=0.05 prevents collapse (latent_std 0.74, probe R² 0.25); probe-protocol pitfalls documented (done)
- [0003-probe-protocol-8x8](experiments/0003-probe-protocol-8x8.md) — rung-1 exit: dynamics 4–5× better than copy baseline (3 seeds); linear probe demoted to diagnostic (done)
- [0004-mpc-agent-empty8x8](experiments/0004-mpc-agent-empty8x8.md) — first acting agent: latent CEM-MPC 19/20 vs random 3/20; multi-step training + reward calibration lessons (done)
- [0005-doorkey-memory](experiments/0005-doorkey-memory.md) — negative: belief-MPC 0/20 on DoorKey; but collection flywheel showed 8.5%→14.3%, and pure MPC's horizon limit identified → value head next (done)
- [0006-value-head-doorkey](experiments/0006-value-head-doorkey.md) — DoorKey solved: value head + flywheel → 90%/75% vs 8% random; eval-hygiene incident (tile-size obs mismatch) resolved + guarded (done)
- [0007-ppo-baseline](experiments/0007-ppo-baseline.md) — honest loss: PPO 100% @ 80k on DoorKey-5x5, ahead at every budget; rung 2 stays open → escalate env + faster flywheel (done)
- [0008-doorkey6x6-vs-ppo](experiments/0008-doorkey6x6-vs-ppo.md) — flywheel fails to ignite on 6x6 (1 reward event in round 0); PPO ~37% > WM 10%; ignition + round-instability are the real blockers → exp 0009 (done)
- [0009-ignition-mechanics](experiments/0009-ignition-mechanics.md) — ignition solved (6 examples → 60% after round 0); retention now the isolated bottleneck (60%→0% under continued training); parity with PPO, rung 2b open → exp 0010 stability (done)
- [0010-retention-mechanics](experiments/0010-retention-mechanics.md) — clean 12-run negative: lr decay + EMA both fail (crash is directional interference = primacy bias); resets (Nikishin) are exp 0011 (done)
- [0011-nikishin-resets](experiments/0011-nikishin-resets.md) — naive reset transfer fails (9 runs, all arms < ctrl): mis-mapped "last layers" (we reset the world model itself); primacy-bias diagnosis confirmed in-setting; corrected mapping = exp 0012 (done)
- [0012-utd-and-targeted-resets](experiments/0012-utd-and-targeted-resets.md) — UTD ×4 wins: 63% mean, first PPO defeat at equal env budget (80% peak); resets conclusively dead; stability bar still open → 0013 trunk-freeze (done)
- [0013-trunk-freeze](experiments/0013-trunk-freeze.md) — BREAKTHROUGH: encoder freeze stops all crashes (6/6 seeds); interference = encoder drift; GRU must stay plastic; ceiling unconverged → 0014 freeze-round sweep (done)
- [0014-freeze-round-sweep](experiments/0014-freeze-round-sweep.md) — when to pin the encoder: fenc@3 vs fenc@4, both bars in reach (running)

## Wanted pages (linked but not yet written)

- world-models-1803.10122 — Ha & Schmidhuber paper page (queued for ingest)
