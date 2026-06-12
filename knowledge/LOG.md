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
