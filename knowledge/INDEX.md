# Knowledge Base Index

One line per page. Updated on every ingest. Conventions: `_schema/SCHEMA.md`,
process: `_schema/PROCESS.md`, raw sources: `sources/SOURCES.md`, backlog: `sources/QUEUE.md`,
training assets: `sources/ASSETS.md`.

## Concepts

Field knowledge — techniques, architecture families, and phenomena that hold
independent of this project. (Our own design/strategy artifacts live under Design &
strategy, below.)

- [world-models](concepts/world-models.md) — root page: what world models are, the four families, why they're our bet (draft)
- [jepa](concepts/jepa.md) — latent-prediction architectures I-JEPA→V-JEPA 2→LeJEPA; collapse problem and fixes (draft)
- [imagination-training](concepts/imagination-training.md) — Dreamer lineage: RSSM, actor-critic in imagination, Dreamer 4 offline (draft)
- [value-equivalent-planning](concepts/value-equivalent-planning.md) — MuZero/TD-MPC2: model only what decisions need (draft)
- [latent-collapse](concepts/latent-collapse.md) — the JEPA failure mode, fixes table, measurement pitfalls (draft)
- [retention](concepts/retention.md) — plasticity loss/primacy bias/interference: fix families, literature↔our status (draft)
- [intellectual-lineage](concepts/intellectual-lineage.md) — 80 years of background in 4 threads: mental models, predictive brain, LeCun's arc, model-based RL; our flywheel = Dyna (draft, unverified ids)
- [hierarchy-and-credit](concepts/hierarchy-and-credit.md) — flat value (have it) vs emergent compositional subgoals (need it for Crafter); the actor should be hierarchical (draft)
- [dreaming](concepts/dreaming.md) — relaxed-constraint generation + weak write-back = Hoel overfitted-brain (dreams as anti-overfitting augmentation); unifies creativity + retention; latent-dream-augmentation is a cheap testable retention experiment (draft, backlog)
- [generative-vs-predictive](concepts/generative-vs-predictive.md) — LeCun vs Xing: latent-abstract vs generative-full are one axis; hallucination is a SEPARATE intrinsic axis (exp 0017 proves latent imagination also hallucinates); generation = knowledge import, not better prediction (draft)
- [language-grounding](concepts/language-grounding.md) — binding + installation: 3 architectures for text→world-model (context/data/weights); the text-staircase design space (draft)
- [temporal-abstraction](concepts/temporal-abstraction.md) — events-not-ticks: Robbins/Bergson critique × options/H-JEPA/event-segmentation; the deep horizon fix (draft)

## Design & strategy

This project's own synthesis — system description, roadmaps, and operating strategy
(distinct from the field-knowledge concepts above; ADRs in Decisions record the frozen
calls these elaborate).

- [capability-map](design/capability-map.md) — the distinct capabilities (world-model/credit/hierarchy/grounding/installation/imagination/hypotheses), what each solves, dependency order; the anti-conflation index (current)
- [agent-architecture](design/agent-architecture.md) — OUR system as 4 optimization layers (mermaid), per-layer failure/diagnosis table (current)
- [architecture-strategy](design/architecture-strategy.md) — modes (configurator) + the real decision is the INTERFACE not model-count + reusable pretrained components (Minecraft WM); decide empirically at rung 3/5b (draft)
- [environment-ladder](design/environment-ladder.md) — our staged env progression with exit criteria; living spec for ADR 0003 (current)
- [compute-strategy](design/compute-strategy.md) — measured local-vs-desktop-vs-rented GPU trade-offs per rung; parallel-seeds corollary (draft)
- [rung4-manual-conditioned-agent](design/rung4-manual-conditioned-agent.md) — the reading-to-learn-dynamics agent: condition the WORLD MODEL (not policy) on the manual; anti-baking as a layered env+eval+probe+arch strategy (draft)
- [grounding-env-spec](design/grounding-env-spec.md) — the read-to-learn-dynamics benchmark spec (authored here → crafter-rtfm); the env side rung-4 consumes (current)

## Papers

- [vjepa2-2025](papers/vjepa2-2025.md) — V-JEPA 2 + action-conditioned variant, zero-shot robot planning (stub)
- [lejepa-2025](papers/lejepa-2025.md) — provable JEPA objective (SIGReg); companion world-model theory paper (stub)
- [dreamerv3-2023](papers/dreamerv3-2023.md) — reference world-model RL baseline, one config for 150+ tasks (stub)
- [dynalang-2023](papers/dynalang-2023.md) — DreamerV3-RSSM + streaming language tokens; auxiliary text-prediction loss as grounding ignition; static-manual mismatch analysis for rung-4 (draft)
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
- [crafter](environments/crafter.md) — rung 3: survival/tech-tree, our wrapper + first contact + Minecraft-proxy role (draft)
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
- [0014-freeze-round-sweep](experiments/0014-freeze-round-sweep.md) — frontier mapped: 40/47/53/60/63% across freeze timings; residual crashes = GRU drift → staged freeze (done)
- [0015-staged-freeze](experiments/0015-staged-freeze.md) — counter-outcome: GRU freeze caps at 30–37%; stability↔performance is a real frontier; rung-2b MET at 0012, pivot to actor recommended (done)
- [0016-actor-distillation](experiments/0016-actor-distillation.md) — BC planner-distillation fails (O(εT²) compounding error + stochastic teacher); eval-variance surprise (80%→25-30% on fresh seeds); on-policy needed → 0017 imagination actor-critic (done)
- [0017-imagination-actor-critic](experiments/0017-imagination-actor-critic.md) — Dreamer AC built; model exploitation (imagined_return 2-3 vs eval 0) — missing continue predictor → 0018 (done)
- [0018-continue-predictor](experiments/0018-continue-predictor.md) — continue predictor partially tames exploitation (s0 2.06→1.13) but actor exploits other off-distribution errors; deterministic WM can't imagine the 25-step chain → stochastic latents; fork (deepen vs DAgger) (done, partial)
- [0019-stochastic-latents](experiments/0019-stochastic-latents.md) — RSSM stochastic latents; partial win, eval off zero (s1 0.55) but value still inflated → fixed policy quality not calibration (done)
- [0020-ac-reset](experiments/0020-ac-reset.md) — REFUTED: per-round actor-critic reset suppressed learning (s1 0.55→0); ruled out behaviour-layer plasticity as the collapse cause (done)
- [0021-critic-on-replay](experiments/0021-critic-on-replay.md) — CONFIRMED: DreamerV3 critic-on-replay calibrates imagined value (2–7→~1); kept as recipe default (--repval 0.3) (done)
- [0022-frozen-dino-probe](experiments/0022-frozen-dino-probe.md) — frozen DINOv2 features encode Crafter state (linear probe 0.98 vs 0.80 baseline); frozen-encoder bet validated (done)
- [0026-replay-ratio-scaling](experiments/0026-replay-ratio-scaling.md) — 2x updates buys BREADTH (survival achievements, count 3.9-4.0) not DEPTH (no crafting tech-tree) + over-training hint (less movement); depth is data/exploration-bound not under-training (done)
- [0025-reward-head-patch](experiments/0025-reward-head-patch.md) — combined fix (patch tokens + two-hot reward head) RESOLVES the exploitation: all 3 seeds move/explore/craft (behavior_report PASS), s2 reaches 6 achievements; ceiling now compute-bound (done)
- [0024-dino-patch-tokens](experiments/0024-dino-patch-tokens.md) — patch tokens unlock navigation (s0 moves + builds table) but seed-dependent; representation IS part of the bottleneck; behavior gate caught the s0/s1 split (done)
- [0023-twohot-value](experiments/0023-twohot-value.md) — two-hot distributional critic bounds Crafter value; first rung-3 learning run climbs to ~3 achievements (2 seeds), plateaus at shallow tree; reward head still inflates imagination (done)
- [0032-rtfm-grounding](experiments/0032-rtfm-grounding.md) — rung-4 first grounding attempt (length-3 recipes): correct never lifts off zero; can't separate "can't ground" from "can't do 3-step sequence" → drop to length-1 (done)
- [0033-rtfm-length1-grounding](experiments/0033-rtfm-length1-grounding.md) — DECISIVE NEGATIVE: the swap test caught a non-reading shortcut correct−none missed (swapped≈correct, swap_follow=0); vision reads the staged state → handoff HO-0006 (done)
- [0034-rtfm-oneshot-ignition](experiments/0034-rtfm-oneshot-ignition.md) — one_shot re-test: fixed a base-reward-farming confound (train on tutorial-only reward), then hit a pure sparse-reward ignition wall (0–6 chance events, no gradient) → curriculum handoff HO-0007 (done)
- [0036-rtfm-shaping-ignition](experiments/0036-rtfm-shaping-ignition.md) — FIRST genuine reading-to-learn-dynamics: HO-0007 reading-shaping (anneal→0) IGNITES reading; 3/4 seeds correct≫none AND swapped≪correct (content-sensitive, no LRS shortcut); grounding partial (swap_follow 0.15–0.40, s0 failed) → strengthen next (done)

## Wanted pages (linked but not yet written)

- world-models-1803.10122 — Ha & Schmidhuber paper page (queued for ingest)
