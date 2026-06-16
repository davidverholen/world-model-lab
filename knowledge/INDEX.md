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
- [hierarchical-imagination-agent](design/hierarchical-imagination-agent.md) — dual-process planning: System-1 reactive default / System-2 deliberate (recursive read-grounded decomposition + imagination/MPC) / compilation / confidence-gated arbitration; the next major capability for the multi-step-execution wall (draft)
- [validated-reading-reward](design/validated-reading-reward.md) — intrinsic reward to TEST what you read: pay the agent for the manual's marginal next-state predictive value, validated against the real transition (reality as un-fakeable judge; marginal framing dodges dark-room/noisy-TV). Targets the [[0040]] objective/identifiability gap; motivates exp0048 (draft, proposal)
- [mentored-learning-loop](design/mentored-learning-loop.md) — NORTH STAR: teach an agent by talking to it — instruct → instant in-context memory → act+validate against reality → consolidate by dreaming (imagination) / practice → skill. Key insight: validation GATES dreaming (don't consolidate hallucinations). Why rung-4 matters (draft, vision)

## Papers

- [vjepa2-2025](papers/vjepa2-2025.md) — V-JEPA 2 + action-conditioned variant, zero-shot robot planning (stub)
- [lejepa-2025](papers/lejepa-2025.md) — provable JEPA objective (SIGReg); companion world-model theory paper (stub)
- [dreamerv3-2023](papers/dreamerv3-2023.md) — reference world-model RL baseline, one config for 150+ tasks (stub)
- [dynalang-2023](papers/dynalang-2023.md) — DreamerV3-RSSM + streaming language tokens; auxiliary text-prediction loss as grounding ignition; static-manual mismatch analysis for rung-4 (draft)
- [dreamer4-2025](papers/dreamer4-2025.md) — transformer WM + shortcut forcing, offline imagination RL, Minecraft diamonds; strategic analysis for RSSM-vs-transformer rung-3 decision (draft)
- [ijepa-2023](papers/ijepa-2023.md) — the first JEPA: multi-block masking, EMA-teacher anti-collapse (vs our SIGReg), method depth (draft)
- [nikishin-primacy-2022](papers/nikishin-primacy-2022.md) — primacy bias + reset protocol, implementation-grade (draft)
- [ma-plasticity-2024](papers/ma-plasticity-2024.md) — FAU module diagnostics; critic-bottleneck in model-free; frozen encoders suffice → legitimizes our fenc finding; Adaptive-RR lever (draft)
- [qiao-model-primacy-2023](papers/qiao-model-primacy-2023.md) — MBRL primacy bias lives in the WORLD MODEL; world-model resets help only at high UTD; confirms our 0011 negative; redesigned exp 0012 (draft)
- [lecun-2022-path](papers/lecun-2022-path.md) — the umbrella position paper; module↔our-layer mapping (draft, skim depth)
- [director-2022](papers/director-2022.md) — Director: manager proposes VQ-VAE subgoals every K=8 steps, worker reaches them in imagination; NeurIPS 2022; direct answer to our multi-step execution wall (draft)
- [cql-2020](papers/cql-2020.md) — Conservative Q-Learning: push-down Q on OOD actions, push-up on data; NeurIPS 2020; foundational offline-RL conservatism; portable to reward heads (draft)
- [crop-2023](papers/crop-2023.md) — CROP: CQL-style push-down applied directly to the learned REWARD estimator in MBRL; conservative Q lower-bound; exp0047 lit anchor (draft)
- [vime-2016](papers/vime-2016.md) — VIME: intrinsic reward = KL(posterior||prior) over BNN dynamics-model parameters (information gain); NeurIPS 2016; single-source IG ancestor of exp0048 marginal-IG validated-reading reward (draft)
- [marino-hypothesis-2020](papers/marino-hypothesis-2020.md) — Empirically Verifying Hypotheses Using RL: agent acts to confirm/refute pre/action/post triplets against real environment; 2020; closest structural analog to exp0048 act-to-verify loop; missing: reading, marginal-value framing, contrastive dual pass (draft)

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
- [0004-remote-dispatch](decisions/0004-remote-dispatch.md) — SSH + git push over Tailscale to the desktop GPU; no orchestrator (accepted)
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
- [0038-rtfm-aux-grounds-wm](experiments/0038-rtfm-aux-grounds-wm.md) — 2× training NULL (0037); Dynalang masked-manual aux GROUNDS the WM (inv_ratio 0.87→1.1–1.4, all seeds ground, s0 fixed) but swap_follow stays ~0.25 → bottleneck is ACTOR execution, not WM reading (correct-mode identifiability) (done)
- [0040-rtfm-actor-conditioning](experiments/0040-rtfm-actor-conditioning.md) — actor-conditioning (§6.2 fallback) does NOT lift swap_follow (mean ~0.18, high variance, s3 best-ever 0.40); 3rd failed lever → swap_follow is OBJECTIVE-bound not architecture-bound → PARK thread, open call on the objective (partial-anneal / mixed-mode train / accept ~0.3) (done)
- [0041-rtfm-length2-generalization](experiments/0041-rtfm-length2-generalization.md) — the length-1 ignition recipe does NOT generalize to length-2 (events 3–6/round, correct ~0); length-1 is the current ceiling → lit-backed length curriculum (RTFM/Messenger) is the next step, pending human sign-off (done)
- [0042-rtfm-length-curriculum](experiments/0042-rtfm-length-curriculum.md) — length-1→2 curriculum does NOT crack length-2 (warm-up ignites len-1, collapses at the switch, never recovers); 2nd failed len-2 lever → converges with swap_follow: reading is solved, the wall is multi-step EXECUTION (done)
- [0043-rtfm-execution-wall](experiments/0043-rtfm-execution-wall.md) — DECISIVE: at length-2 inv_ratio>1 (= length-1, WM reads the 2-step gesture) but correct≈0 → PURE EXECUTION wall. Same bottleneck as the rung-3 Crafter plateau → next major capability = multi-step execution / hierarchy (done)
- [0044-rtfm-mpc-execution](experiments/0044-rtfm-mpc-execution.md) — CEM-MPC ≈ reactive at length-2 (mean 0.09 vs 0.06, within noise; neither cracks it) → planning doesn't solve the wall; it relocates to WM multi-step rollout/reward FIDELITY (upstream of all planning/hierarchy). Next: oracle-gesture probe to localize (done)
- [0045-rtfm-oracle-probe](experiments/0045-rtfm-oracle-probe.md) — oracle probe: WM reads+VALUES the gesture (pct≈0.88, ret 5–6× random) — NOT a fidelity wall. But ~12% of wrong sequences are overrated → reward-head OOD false positives the planner exploits = exp0017–0025 at the planning layer. Fix = reward calibration / robust planning (done)
- [0046-rtfm-robust-planning](experiments/0046-rtfm-robust-planning.md) — v1 sampled-rollout MPC (K=10, average): PARTIAL. Oracle pct 0.88→0.92, MPC correct 0.05→~0.11 (~2× floor), swapped≈0 (anti-baking holds) — but pct plateaus ~0.91, not 1.0. Averaging shaves SOME prior-mean optimism; the bulk of the ~12% OOD false positives is a starved/miscalibrated reward head (collection-signal: ~5/60 length-2 positives). Next: compose pessimism / continue-gating / reward-OOD-reg (done)
- [0047-rtfm-conservative-reward](experiments/0047-rtfm-conservative-reward.md) — CROP/CQL conservative reward head (push down predicted reward on OOD actions): NEGATIVE. 6-coef sweep → monotonic over-suppression (oracle_ret +0.13→−0.13 as α 0.1→3.0); no coef beats the 0.88 baseline. Conservatism can't separate OOD junk from the sparse true gesture. CLOSES the execution/calibration thread (0044–0047); swap_follow≈0 throughout = the [[0040]] objective gap → attack the objective (done)
- [0048-rtfm-validated-reading](experiments/0048-rtfm-validated-reading.md) — intrinsic "validated-reading" reward (manual's marginal next-state predictive value, confirmed by the REAL transition; deter_noctx baseline; reality = judge → anti-baking). WEAK CONFIRM (4-seed control): FIRST lever to move length-2 swap_follow off 0 — all 4 seeds 0.05–0.13 (mean 0.105) vs flat 0.00 control, swapped≈0, events ~2× control. But phase-1's 0.18/oracle-0.92 were INFLATED (1-seed + GPU nondeterminism; oracle washes to baseline). Real but underpowered → strengthen via [[0049-rtfm-sustained-vr]] (verified)
- [0049-rtfm-sustained-vr](experiments/0049-rtfm-sustained-vr.md) — NULL: shaping-floor 0.2 gives NO lift (0.094 vs 0.105 control); swap_follow is a robust ~0.10 across α∈{0.1,0.3}, floor∈{0,0.2}, AND MPC (≈reactive). VR cracked the OBJECTIVE gap (0→0.10) but ~0.10 is the EXECUTION ceiling re-met (exp0043/44) — invariant to reward strength/scaffold/search. Reward-MPC can't fix it (swap_follow earns no reward in swapped mode) → needs manual-DIRECTED execution / hierarchy. DESIGN FORK for maintainer (verified)
- [0050-rtfm-hierarchy](experiments/0050-rtfm-hierarchy.md) — Director hierarchy for the length-2 execution wall: CONCLUDED NEGATIVE. ~0 across BOTH goal spaces (raw-belief K-means + learned VQ autoencoder) + every stability fix; worker never becomes goal-directed (worker_sim flat ~0.5), manager collapses. Leading read: hierarchy is the WRONG tool for a 2-step gesture (nothing to decompose; Director is for long-horizon). Recommend banking validated-reading; revisit execution as a short-horizon problem (verified)
- [0051-rtfm-flat-vr-optimization](experiments/0051-rtfm-flat-vr-optimization.md) — push flat-VR past the ~0.10 length-2 swap_follow ceiling (no hierarchy). Budget probe (2x training): REAL ceiling, NOT compute-bound (0.105→0.115). coef-0.5 ≈0.10 → coef lever EXHAUSTED. On-deck decoupled head → [[0052-rtfm-decoupled-vr-head]] (NEGATIVE). Whole grounding-incentive axis saturates/collapses → pivot to localizing the wall ([[0053-rtfm-imagination-fidelity]]) (verified)
- [0052-rtfm-decoupled-vr-head](experiments/0052-rtfm-decoupled-vr-head.md) — decoupled validated-reading reward HEAD (dedicated two-hot VR head, actor optimises task + λ·VR_head). CONCLUDED NEGATIVE: λ=1.0 COLLAPSED the policy to manual-blind (swap_follow 0.00–0.02 frozen, dead flywheel, imagined_return inflated 15–25) — the actor reward-hacks the VR head's OOD over-predictions (exp0045/0047 redux on a new channel; no conservative guard). Pushing grounding HARDER keeps failing → bottleneck is downstream (verified)
- [0053-rtfm-imagination-fidelity](experiments/0053-rtfm-imagination-fidelity.md) — imagination-fidelity probe (imagine_report.py): from a shared real s0, imagine the plan then replay it real, compare belief/h divergence vs a real-vs-real sampling floor + side-by-side GIF. FINDING: the WM imagines FAITHFULLY (h-div ~1.5–2.7× floor, both baseline + collapsed) → the ~0.10 wall is NOT the world model, it's DOWNSTREAM (policy/reward-readout). Baseline WM is manual-sensitive (CORRECT 1.5× vs SWAPPED 2.7×); collapsed model lost that. Next: reward-gap on exp0054 (verified)

## Wanted pages (linked but not yet written)

- world-models-1803.10122 — Ha & Schmidhuber paper page (queued for ingest)
