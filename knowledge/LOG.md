# Knowledge Base Log

Append-only. Entry format: `## [YYYY-MM-DD] <ingest|query|lint|curation> | <title>`

## [2026-06-12] curation | Knowledge base bootstrapped

Initial setup per ADR 0002 (Karpathy LLM Wiki structure × Context Architecture
whitepaper process). Created: schema (+4 templates), source registry (27 sources, 7
verified), intake queue, 5 concept pages, 4 paper stubs, 1 lab page, 2 environment
pages, 3 ADRs, 1 planned experiment, INDEX.

Research provenance: web verification done for V-JEPA 2 (2506.09985), Dreamer 4
(2509.24527), LeJEPA (2511.08544), When-Does-LeJEPA-Learn-a-World-Model (2605.26379),
AMI Labs funding (TechCrunch 2026-03-09), Karpathy LLM Wiki gist. All other arXiv ids
come from a training-data compilation and are marked `verified: no` in SOURCES.md —
verify at ingest time.

## [2026-06-12] curation | Smoke run confirms collapse hypothesis (exp 0001)

End-to-end pipeline verified on CUDA (`world_model.collect`, reduced settings):
naive joint latent regression collapses (latent_std 0.0005 @ 100 updates).
Preliminary result recorded on experiments/0001; formal run with default settings
still pending. Next: experiment 0002 (SIGReg per [[lejepa-2025]]).

## [2026-06-12] curation | milestone: project bootstrap

Checkpoint protocol adopted (PROCESS.md § Milestone checkpoint, /milestone skill):
after every meaningful milestone, curate the KB first, then commit everything in one
commit. Touched-page lint for this checkpoint: INDEX complete, frontmatter consistent,
known dangling links registered under "Wanted pages" (latent-collapse,
world-models-1803.10122). First commit follows this entry.

## [2026-06-12] ingest | LeJEPA method sections (arXiv:2511.08544 + reference repo)

SIGReg details extracted (Epps-Pulley on random slices, 17-point trapezoid, λ=0.05,
batch≥128, bounded gradients). lejepa-2025 page stub→draft, read state: skimmed.
Pages touched: papers/lejepa-2025, concepts/latent-collapse (new), concepts/jepa.

## [2026-06-12] curation | milestone: SIGReg fixes latent collapse on MiniGrid

Exp 0001 formal run recorded (full collapse, commit 6692aa2). Exp 0002 run and
recorded: SIGReg λ=0.05 holds latent_std at ~0.75 over 1000 updates (control: 0.0003);
final probe protocol reads SIGReg 0.25 vs control −0.49. Probe target 0.8 not met —
representation-quality measurement itself was the main lesson (amplitude vs
information; standardization pitfall) → documented in latent-collapse page; proper
probe protocol is exp 0003. INDEX updated (latent-collapse off wanted list, exps
done); touched-page frontmatter checked. Next: exp 0003 probe protocol + Empty-8x8,
then memory for partial observability (rung 2).

## [2026-06-12] curation | milestone: rung 1 complete — world model beats copy baseline

Exp 0003 run (Empty-8x8, 3 seeds x 2 arms): SIGReg world model predicts held-out
transitions at 0.20-0.30x the copy-baseline error (controls: 1.5-1.7x). Rung-1 exit
criterion marked met on environment-ladder (owner: human — flagged to Dave for
review). Linear probe demoted to diagnostic-only after persistent seed noise;
dynamics-vs-copy ratio promoted to primary metric (latent-collapse page updated).
New tooling: world_model.play viewer (human render / GIF record), checkpoint saving
(--save). Next: rung 2 — DoorKey partial obs, memory, then first acting agent.

## [2026-06-12] curation | milestone: first acting agent — world model plays Empty-8x8 at 19/20

Exp 0004 closed after three iterations (6/20 -> 0/20 -> 12/20 -> 19/20 with stronger
eval-time planning; random baseline 3/20). Root causes found by diagnostics, not
tuning: (1) compounding rollout error from 1-step-only training -> multi-step rollout
training (replay sequence windows, losses at every imagined step, reward head trained
on drifted latents); (2) reward-magnitude starvation under sparse rewards ->
pos_weight 100. New code: RewardHead, MPCAgent (CEM), play --checkpoint,
ReplayBuffer.sample_sequences, --rollout-length. ROADMAP phase-1 complete + first
acting agent ticked. Lessons routed to exp page; "compounding rollout error" is now
mentioned on 2 pages — concept-page candidate if it recurs. Next: DoorKey + memory
(rung 2), PPO baseline comparison.

## [2026-06-12] curation | milestone: recurrent world model — DoorKey negative result, flywheel signal

Exp 0005 closed as an instructive negative: belief-state CEM-MPC 0/20 on DoorKey-5x5
(hypothesis >=40% refuted), but the collect->train->collect flywheel showed its first
life (random 8.5% -> round-0-model+eps-MPC 14.3% during collection) and the failure
decomposed cleanly: (a) pure MPC cannot span DoorKey's ~25-step reward chain with
horizon 20 -> value head needed (exp 0006, TD-MPC/Dreamer lineage answer); (b)
continued training across rounds destabilized (pred_loss 0.46->0.59) -> per-round
checkpoints/eval + optimizer handling next time. New infra committed: RecurrentDynamics
GRU belief model, burn-in/open-loop sequence training, RecurrentMPCAgent,
play --epsilon + recurrent checkpoint auto-detection, RGB partial-obs wrapper.
Also: compute-strategy concept page (local vs 5070 Ti vs vast.ai prediction, prices
checked 2026-06-12). Next: exp 0006 value head + per-round eval.

## [2026-06-12] curation | thermal measurement: laptop GPU power-capped + throttling

90s load test after Dave noticed heat: 4070L capped at ~45W (TGP floor), 62->78C in
90s, clocks ~1.2GHz vs 3.1 max, SW thermal slowdown already active ~396s cumulative
today. compute-strategy page updated (5070 Ti advantage revised 2.5-3x -> 4-6x;
desktop-dispatch trigger lowered 4h -> 1h); CLAUDE.md hardware note updated.

## [2026-06-12] curation | milestone: remote GPU dispatch live + benchmark validates compute strategy

Windows desktop (RTX 5070 Ti) wired up end-to-end: OpenSSH + Git Bash default
shell (cmd.exe breaks git transport), bare-repo push dispatch (scripts/remote.sh
setup/gpu/run/pull), uv sync with marker-gated cu130 torch wheels, CUDA verified.
First real dispatch = gpu_bench.py: matmul 4.5x (prediction 4-6x confirmed),
our recurrent train-step 0.93x (latency-bound prediction confirmed) ->
compute-strategy page updated with measured table. Publishability pass on Dave's
request: no machine names/keys/paths in tracked files; config via env vars +
gitignored .env.remote (.env.remote.example committed); ADR 0004 records the
design. Setup gotchas (bare HEAD main-vs-master, administrators_authorized_keys,
DefaultShell) documented in docs/REMOTE.md.

## [2026-06-12] curation | milestone: DoorKey solved — value head + flywheel (exp 0006)

Exp 0006 closed: 90% per-round eval / 75% post-hoc (20 eps) vs 8% random — value
head makes beyond-horizon reward visible; collect->train->collect compounds
(collection 8.5%->5%->36%). First training dispatched through the remote pipeline
(5070 Ti, ~25 min). Eval-hygiene incident found and fixed: play.py tile_size=16
changed agent observations vs training tile 8 (4/20 -> 75% after fix); exp 0004
re-verified at 20/20 matched; obs_shape now stored in checkpoints + asserted at
load; minigrid.md gotcha section added. Process additions per Dave's request:
reviewer agent (opus, research-code charter — first pass verified 0006 paths
correct + caught sweep-scrapability gap), /sweep skill + scripts/sweep.py,
scripts/publish_check.sh wired into /milestone verify. Next: PPO baseline
(closes rung 2), then Crafter prep.

## [2026-06-12] curation | milestone: PPO baseline wins on DoorKey-5x5 — rung 2 stays open (exp 0007)

Pre-registered counter-outcome occurred: sb3 PPO (CnnPolicy, 8 envs, no tricks)
hits 100% on all 3 seeds by 80k steps, ahead of our agent at every budget; our
agent burns 60k steps on random collection before the flywheel spins. Honest
verdict recorded; rung-2 sample-efficiency criterion NOT met on this env.
Escalation pre-registered in exp page: DoorKey-6x6/8x8 + earlier/smaller
collection rounds (exp 0008); ladder-criterion split proposed for Dave's
sign-off (owner: human). Infra: baselines/ppo as isolated uv project (sb3 caps
gymnasium <1.3); first /sweep dogfood produced the results table directly.

## [2026-06-12] curation | milestone: 6x6 ignition failure — the real blockers identified (exp 0008)

Exp 0008 closed via pre-registered alternative (b): round-0 random collection drew
1 reward event in 20k steps and the flywheel never ignited (best 10% vs PPO mean
~37% at 110k; PPO breakthrough moved 60-80k on 5x5 -> 100k+ on 6x6). Second
confirmed sighting of round-over-round training instability (round 6: 28%
collection success -> 0% eval after retraining). Exp 0009 agenda: ignition
(adaptive round 0, success-episode oversampling in replay, intrinsic signal) +
stability (success-balanced sampling, lr schedule, EMA agent weights); world-model
runs need >=2-3 seeds (round-0 luck is decisive). Rung 2b open; scoreboard PPO 2:1.

## [2026-06-12] curation | milestone: ignition solved, retention isolated (exp 0009)

3 parallel seeds on the 5070 Ti (first remote.sh shell use; ~2h wall, 82% util,
49C). Adaptive round 0 + success-window oversampling fixed ignition everywhere
(seed 1: 6 success examples -> 60% greedy straight after round 0 — vs PPO 0% at
that budget). But continued round training destroyed and only partly rebuilt that
competence (60->0->50%): catastrophic interference under distribution shift is now
THE isolated bottleneck (3rd sighting, first clean). Best-checkpoint means: WM 40%
@ 115-130k vs PPO 37% @ 110k — parity via guard, not a win; rung 2b open. Exp 0010
(pre-registered): retention mechanics — EMA/snapshot acting weights, lr decay after
round 0, value target network, or frozen-trunk/head-only later rounds. Crafter
source verified (abstract) during the wait. remote.sh gained shell+kill; killed
runs survive ssh death on Windows — kill subcommand is the off switch.

## [2026-06-12] curation | timing correction: exp 0009 parallel sweep was 35 min, not ~2h

Log timestamps (12:04 launch -> 12:37-12:39 per-seed finish): 3 parallel seeds in
35 min wall, contention nil. Estimate was 3.5x pessimistic (MPC collection per-step
cost on the 5070 Ti overestimated). exp 0009 page + compute-strategy corrected;
rung-2 sweeps are coffee-break scale.

## [2026-06-12] curation + scout | agent-architecture page; plasticity-loss literature found

New concept page: agent-architecture — our 4 optimization layers (CEM planning /
belief-model / Adam joint training / data flywheel) as a mermaid diagram with a
per-layer failure->experiment diagnosis table. Scout (Dave's prompt): our retention
problem IS the literature's "plasticity loss / primacy bias / churn" — 6 sources
queued (Nikishin resets = top exp-0011 candidate: reset last layers, keep replay).
QUEUE gained a themed retention section.

## [2026-06-12] curation | milestone: retention 2x2 negative — primacy bias confirmed, resets next (exp 0010)

Full factorial (ctrl/lr03/ema/both x 3 seeds, 2x 6-wide parallel batches): no arm
prevents the post-round-0 crash; seed 1 crashes in ALL arms (deterministic
interference, not noise); EMA damps peaks (60->40) without protecting them; lr
decay kills late recovery. Conclusion: smoothing the weight trajectory can't fix
directional interference — matches primacy-bias literature; exp 0011 = Nikishin
resets (reinit heads each round, keep replay + oversampling). Ops lessons: 6-wide
batches page heavily (replay ~10GB/process -> uint8 storage queued); block-buffered
logs look empty mid-run; laptop suspend pauses monitors (runs unaffected).
Also today: agent-architecture diagram iterated to extension-safe form (no init
directive, classDef styling).

## [2026-06-12] ingest | Nikishin primacy bias (protocol depth) + LeCun path paper (skim) + retention concept page

papers/nikishin-primacy-2022 (implementation-grade protocol; underpins running exp
0011), papers/lecun-2022-path (skim depth, module<->our-layer mapping table; full
read queued), concepts/retention (fix-family table: smoothing ruled out by 0010,
resets running, churn/continual-backprop/frozen-trunk untested). DINOv3 queued with
frozen-encoder framing. PROCESS gained the literature-first rule (Dave's request —
flagged here since _schema is human-owned).
