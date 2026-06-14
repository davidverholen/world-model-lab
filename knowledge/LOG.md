# Knowledge Base Log

Append-only. Entry format: `## [YYYY-MM-DD] <ingest|query|lint|curation> | <title>`

## [2026-06-14] ingest | Curious Replay for Model-based Adaptation (Kauvar et al., ICML 2023) → papers/curious-replay-2023.md (method depth; priority formula, update rule, Crafter results, frozen-encoder applicability)

## [2026-06-14] curation | milestone: repo made public-ready — leaks scrubbed, README opened for third parties

Publication-prep pass on the maintainer's request (no research content changed). Internal/
personal leaks removed from tracked files: `.vscode/settings.json` (bypass-permissions
config) untracked + `.vscode/` gitignored; author email dropped from `pyproject.toml`;
README "whitepaper" link repointed from a LinkedIn profile to the context-architecture
GitHub repo. Depersonalized 142 "Dave" mentions → "the maintainer" across 28 files (LOG,
CLAUDE.md, skills, experiment/concept/design pages) + softened one proton.me mention here.
Owned-GPU model names (RTX 4070 Laptop / 5070 Ti) genericized to capacity/role descriptors
in CLAUDE.md and design/compute-strategy.md (all measurements/conclusions preserved; rental/
market GPUs 4090/5090/H100/3090 left intact). Added MIT LICENSE. README quick-start
restructured around the real third-party arc (collapse→SIGReg fix, then train DoorKey agent
→ watch it play via `--checkpoint`). `baselines/ppo/` kept — load-bearing for exp 0007/0008
(the rung-2 sample-efficiency claim). Verify: 21 tests pass, ruff clean, publish_check clean.
Note: `last_reviewed` dates deliberately NOT bumped on depersonalized pages — the edits were
mechanical (name swap), not content re-review.

## [2026-06-13] milestone+curation | exp 0021 CONFIRMED — critic-on-replay kept as recipe default

Critic-on-replay (DreamerV3 β_repval 0.3) confirmed on our stack: imagined_return pulled
from 0019's inflated 2–7 down to ~1.0 (s1 clean 3.2→1.0) — independent validation of the
Dreamer mechanism AND our implementation. Ignition still occurs (s2 0.55, s0 0.15) but
LATE (round 6 vs 0019's 3–5) and at honest value; endpoint trajectories opposite (0019
inflated→collapse, 0021 calibrated→rising). Best evals seed-reshuffled wash
(0019 0/0.55/0.35; 0021 0.15/0/0.55). DECISION (maintainer): mechanism confirms the Dreamer
design → keep it; default flipped `--repval` 0.0→0.3 (recipe default now); bank the
MiniGrid imagination loop as good-enough+calibrated, don't over-polish the stepping stone,
move to recipe-hardening (two-hot/symlog/percentile-norm) → Crafter. Caveat logged: core
mechanism confirmed on a sparse task; full DreamerV3 recipe still to harden on denser
rewards. Process meta-lesson (2nd time): I read rounds 0–5 as "ignition vanished"; the
final round flipped it — partial-data conclusions burned us again (cf. 0019 smoke-test).
Pages: experiments/0021 Result+Lesson (status CONFIRMED); code default flip.

## [2026-06-13] curation | language-grounding: symbols-as-thought ideas (maintainer)

Added an "Ideas to look into later" section to concepts/language-grounding.md capturing
three forward ideas from a conversation (flagged not-decisions): (1) pixels-for-percept /
symbols-for-thought boundary — text→pixels→frozen vision encoder is right for text the
agent SEES in-world, wrong for text-as-thought (rendering discards symbolic structure the
tokens already carry); distinct from text→video→latent which stays legit (generating
experience ≠ binding meaning). (2) The deep question is symbol→belief binding (a word =
pointer into a belief region), which makes language a capability jump (ape↔human gap):
compositional abstraction lets the agent imagine in abstractions not pixels (counterfactuals,
"what if I had a pickaxe") ⇒ systematic generalization. (3) LLM-as-aligned-source-into-belief
= PAN's leverage done experience-primary: keep frozen vision, add a language pathway mapping
LLM-seeded tokens into the SAME belief space, alignment learned by grounding; beyond Dynalang
on deep binding + LLM-prior reuse. Also doubles as why [[frozen-encoder-lean]] survives the
language rung. Rides into next milestone commit (with the exp 0021 result). No page cites
PAN/Dynalang as authority (QUEUE-only).

## [2026-06-13] curation | Queued imagination-module substrate thread

Conversation on modular reuse of big pretrained world models (could a Cosmos-scale
video model serve as the Dreamer imagination loop?). Verdict in-conversation: no for
the inner loop (latency/pixel/non-differentiable — wrong complexity class), but the
question reframes usefully as "reuse the representation, not the simulator." Added a
neutral OPEN-QUESTION block to sources/QUEUE.md bracketing the design space with two
already-in-KB items: DIAMOND (2405.12399, diffusion WM) vs V-JEPA 2-AC (in 2506.09985,
frozen-encoder + predictor). Explicitly flagged not-a-decision to avoid biasing the
future read. No wiki page asserts anything yet.

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
criterion marked met on environment-ladder (owner: human — flagged to the maintainer for
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

90s load test after the maintainer noticed heat: 4070L capped at ~45W (TGP floor), 62->78C in
90s, clocks ~1.2GHz vs 3.1 max, SW thermal slowdown already active ~396s cumulative
today. compute-strategy page updated (5070 Ti advantage revised 2.5-3x -> 4-6x;
desktop-dispatch trigger lowered 4h -> 1h); CLAUDE.md hardware note updated.

## [2026-06-12] curation | milestone: remote GPU dispatch live + benchmark validates compute strategy

Windows desktop (RTX 5070 Ti) wired up end-to-end: OpenSSH + Git Bash default
shell (cmd.exe breaks git transport), bare-repo push dispatch (scripts/remote.sh
setup/gpu/run/pull), uv sync with marker-gated cu130 torch wheels, CUDA verified.
First real dispatch = gpu_bench.py: matmul 4.5x (prediction 4-6x confirmed),
our recurrent train-step 0.93x (latency-bound prediction confirmed) ->
compute-strategy page updated with measured table. Publishability pass on the maintainer's
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
load; minigrid.md gotcha section added. Process additions per the maintainer's request:
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
collection rounds (exp 0008); ladder-criterion split proposed for the maintainer's
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
per-layer failure->experiment diagnosis table. Scout (maintainer's prompt): our retention
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
frozen-encoder framing. PROCESS gained the literature-first rule (maintainer's request —
flagged here since _schema is human-owned).

## [2026-06-12] scout | AMI pair completed (WBench id), Causal-JEPA + VL-JEPA queued

Watch-list sweep (3 searches): WBench arxiv:2605.25874 confirmed as the late-May
AMI-circle companion benchmark (theorem 2605.26379 + stress test pair; "current
models collapse under minor visual shifts") -> ami-labs page updated, QUEUE open
item resolved. New: Causal-JEPA (2602.11389, object-level latent interventions),
VL-JEPA (2512.10942, low priority). Dreamer/DeepMind side: nothing new beyond
known Genie/Dreamer state; 2026 survey blogs noted but not queued (secondary).

## [2026-06-12] curation | intellectual-lineage page (maintainer's request: decades of background)

New concept page organizing 4 threads: mental models (Craik 1943, Tolman, Kahneman
S1/S2 -> Mode-1/2), predictive brain (Rao-Ballard, Friston), LeCun's arc (LeNet ->
contrastive -> EBM -> cake -> 2022 path), model-based RL (Dyna 1991 — our flywheel's
true name — Schmidhuber 90/91, POMDP belief states, 2018 revival). 6 lineage
sources queued (background section). Page verified:false until ids checked on ingest.

## [2026-06-12] curation + scout | QUEUE ids batch-verified via arXiv API; rung-3 SOTA line queued

All ~34 queued arXiv ids verified in one API call (id->title). Catches: DINO-WM id
was WRONG (2411.04958 = astronomy survey; corrected to 2411.04983, ICML 2025) —
the "id uncertain" flag from intake did its job; two title corrections (General
agents CONTAIN world models; MAINTAINING Plasticity). SOURCES verified flags
flipped (25 rows). New rung-3 prep section: 2502.01591 (Craftax SOTA jump),
2605.16457 ITC (May 2026, current Craftax SOTA 72.5%, identifiability framing
converging with LeJEPA theorem), delta-IRIS + DART (low).

## [2026-06-12] ingest | DreamerV3 (implementation depth) + I-JEPA (method depth); local PDF archive

dreamerv3-2023 stub->draft: full trick kit extracted (symlog, two-hot K=255,
KL balancing 1/0.5/0.1 + free bits 1 nat, return percentile norm, imagination
H=15/lambda .95) — retention-relevant find: their critic-EMA is a TARGET
(regularize toward slow copy), not acting weights like our failed 0010 arm; +
training ratio up to 16 vs our 0.1. New page ijepa-2023: multi-block masking
params, EMA-teacher 0.996->1.0 — the heuristic LeJEPA replaced. QUEUE marks.
Maintainer's request: scripts/fetch_sources.sh -> 21 arXiv PDFs (183MB) in gitignored
knowledge/sources/files/ for personal reading; idempotent, documented in SOURCES.

## [2026-06-12] lint | 6-ingest checkpoint: graph closed, 1 INDEX gap fixed

Scripted pass over 45 pages: zero dangling [[links]] (cross-link graph fully
closed); INDEX was missing exp 0011 (fixed); remaining stubs (dreamer4, vjepa2)
honestly labeled, not cited as authority. Skills updated with today's craft:
ingest-source gained archive step + read-depth labels + ar5iv route;
scout-sources gained the arXiv-API batch-verification trick. Assessment: no new
skills/agents needed at current KB size; ingestion stays on-demand per the
literature-first rule (exceptions queued: Dyna, WBench).

## [2026-06-12] curation | research loop codified as top-level process (maintainer)

PROCESS.md gains "The research loop": experiment-verify <-> literature-first <->
route-to-KB <-> skills-carry-procedure-not-knowledge. Point 4 sharpened from
the maintainer's draft: maintenance trigger is process drift, not knowledge growth (thin
skills + INDEX navigation make new knowledge reachable without skill edits).

## [2026-06-12] curation | research-loop horizon defined (maintainer)

Process holds until a blocking problem has no published solution; then step 2
transforms (nearest-neighbor mapping + novel mechanism + our pages as primary
record). Noted: the model-based retention transfer (exp 0011) may already sit
on that edge.

## [2026-06-12] curation | deep-search protocol added to scout skill (maintainer's G-Scholar question)

Tested live: Semantic Scholar API = the right semantic+citation tool (anonymous
tier congested; free key recommended -> env S2_API_KEY); OpenAlex = instant but
weak ranking, good citation graphs; Google Scholar = manual-only (no API/ToS).
Local embeddings index DEFERRED with trigger: build it over OUR corpus (pages +
PDF archive) at ~150 pages, not over external papers (duplicates S2).

## [2026-06-12] curation | deep-search stack re-ranked: OpenAlex primary (S2 key gated)

S2 API-key form requires institutional affiliation + rejected a personal (non-institutional) email —
independent research not in their concept. Re-ranked: OpenAlex (truly open) as
programmatic primary with anchor-paper citation-walking to compensate ranking;
S2 anonymous as best-effort bonus. Project principle reinforced: the KB/process
depends on no gated service (arXiv, ar5iv, OpenAlex all open).

## [2026-06-12] curation | milestone: naive reset transfer fails — frontier confirmed (exp 0011)

9 runs, 3 arms, all below ctrl: resetting "heads" includes next-latent = world-model
amputation each round at 25x-too-low replay ratio (per-round amnesia). Bonus
finding: hr arm's s1 round-0 60->20 under doubled early training = primacy bias
reproduced in-setting (diagnosis confirmed, cure mis-mapped). Exp 0012
pre-registered in the page: value/reward-only reset, once at round 3, 4x post-reset
updates, shrink-perturb arm. Horizon rule active: no published work on this
question — our pages are now the primary literature for it. uint8 buffers held
6-wide at 3.7GB (paging fixed). retention.md fix-table updated.

## [2026-06-12] review | exp 0011 reviewer pass: latent reset+EMA bug found, negative result verified clean

Opus reviewer on the milestone diff: uint8 roundtrip provably exact, reinit
coverage complete, optimizer rebuild correct — and one important latent bug:
--reset never reaches the EMA shadow (acting weights), so reset+EMA combined
would silently no-op. Checked all 9 exp-0011 checkpoint configs: ema_decay=0
throughout -> negative result stands clean. Guard added (mutually exclusive
flags); lr-rebuild fragility commented.

## [2026-06-12] scout+ingest+correction | frontier claim overturned by the wall protocol (maintainer's prompt)

Ran the hitting-a-wall protocol properly for the first time (OpenAlex anchor walk +
S2 citation pages of Nikishin, ~300 citations triaged): found arXiv:2310.15017
"Mind the Model, Not the Agent" (2023) — directly on-topic, ingested at method
depth. It independently confirms our 0011 negative (agent resets harm MBRL),
locates MBRL primacy bias in the world model, and predicts reset failure at low
model-UTD (ours: ~0.1) — reframing our crashes as possible UNDER-training. Exp
0011 page corrected (frontier claim narrowed to sparse-reward low-UTD belief-MPC
flywheel); exp 0012 redesigned (Qiao shrink-perturb arm, UTD-scaling diagnostic
arm, surgical-heads arm); PROCESS horizon rule hardened (deep-search mandatory
for frontier claims); 6 more citation-walk finds queued. ./scripts/fetch_sources.sh
to be re-run for the new PDF.

## [2026-06-12] curation | ADR 0005: Minecraft milestone (maintainer) — three-signal training

New rung 5b before real-world transfer: self-play + VPT-style action-labeled video
+ text-in-world-model (open research; Dynalang/VL-JEPA direction). Ladder + ROADMAP
amended; 5 sources queued (ids to verify). Rationale: forces multi-modal/language
integration without robotics hardware; resource-rich (VPT corpus, MineRL/MineDojo,
published reference agents incl. Dreamer 4 offline diamonds).

## [2026-06-12] curation | text-signal staircase added to ADR 0005 (maintainer)

Messenger/RTFM (text necessary) -> text-augmented Crafter (text helpful; metric =
sample-efficiency delta) -> Minecraft tutorials. Steps i-ii run on current hardware.

## [2026-06-12] curation | language-grounding concept page (maintainer's binding/installation insight)

New page: text->world-model = binding (shared embedding geometry) + installation
(declarative -> dynamics belief). Three architectures: text-as-context (Dynalang),
text-as-data (text-induced imagination training — possibly unexplored; wall
protocol required before claiming), text-as-weights (Schmidhuber fast weights /
model editing). Pre-registered prediction: context suffices at Messenger scale;
data/weights needed at wiki scale. Linked into ADR 0005 + lineage + retention.

## [2026-06-12] curation | language-grounding 2b: external imagination engine (maintainer)

Text -> domain video generator (Oasis/Genie-3/Dreamer-4-WM class) -> VPT-IDM action
labels -> trust-weighted synthetic replay -> corroboration gate (real play
validates/discounts). Breaks 2a's circularity; unifies ADR-0005 signals 2+3;
cognitive analogy: Craik's mental simulation + belief corroboration. Gate is
mandatory (video hallucination) and retention-adjacent (synthetic distribution
shift).

## [2026-06-12] curation | unified trust-weighted replay (maintainer): source priors + corroboration

Real vs imagined experience unified on one trust scale: source sets the prior,
corroboration updates it, loss scales with it; imagination earns what real
experience gets at birth. Human analogy: imagination is inexact AND functional via
constant verification. Prototype path: corrupted-synthetic injection on Crafter.
Symmetry noted with the KB's own verified-flag epistemics.

## [2026-06-12] curation | horizon: self-generated hypotheses (maintainer) — agent internalizes the research loop

Text-installed beliefs -> imagination-generated beliefs -> active validation, all on
the trust-gate substrate. Near-term echo: hypothesis-driven exploration as the
principled ignition fix (exp 0008 thread); Plan2Explore queued.

## [2026-06-12] scout | language/imagination design space swept (4 searches, 14 queued)

Solved-vs-open verdict per thread: binding-by-conditioning substantially explored
(2511.22904 reads dynamics descriptions — read before Messenger work); generated-
video-as-experience active at platform level (survey 2603.28489); synthetic-
transition reweighting mature BUT all generator-self-confidence based — the maintainer's
reality-corroborated cross-source gate stays novel; uncertainty-driven exploration
covered (Plan2Explore, DreamerV3-XP) — compositional hypothesis generation beyond
state-novelty stays open. language-grounding page to absorb refs at ingest time.

## [2026-06-12] curation | paper-readiness check (maintainer): provenance chain confirmed, venue column added

Claim -> page -> source id -> verified registry row chain is manuscript-grade by
construction; read-depth labels gate citability (only `read` sources citable in a
manuscript). Gap closed: venue/peer-review status now recorded at ingest (backfill
at pre-paper lint); BibTeX export planned (mechanical from arXiv ids).

## [2026-06-12] curation | /research-cycle skill: autonomous loop with mandatory per-failure literature gate

Prepared for overnight autonomous operation (the maintainer's standing instruction, to be
activated later): harvest -> record -> mandatory failure-specific search ->
pre-register -> implement -> dispatch -> milestone; guardrails (scope, spend,
2-strike stall rule, honesty, state-of-the-night report). PROCESS gained the
per-iteration literature gate.

## [2026-06-12] curation | training-asset registry + tier-1 VPT retention (maintainer's availability insurance)

New ASSETS.md registry (tiered retention: T1 now / T2 at rung-5b entry / T3 never-
bulk). Tier-1 retained to desktop ~/minecraft-assets/ (~10GB, 1.28TB free): IDM 4x
(the video->actions labeler — load-bearing for architecture 2b), all VPT
foundation/fine-tuned/RL weights, contractor-data index JSONs, repo snapshot (MIT).
Key finding: the 70k-h YouTube corpus was NEVER released (recipe only) —
contractor data + IDM are the retainable substance. VPT id verified:
arXiv:2206.11795. T2 (contractor subsets, MineRL, MineDojo dumps) deferred to
rung-5b entry with availability re-check.

## [2026-06-12] curation | tier-1 retention confirmed: 21/21 OK, 9.8 GB

All VPT tier-1 assets verified on desktop ~/minecraft-assets/vpt/ (first detached
attempt died with its ssh session — Git Bash quoting; foreground-over-held-ssh
pattern worked). ASSETS.md claims now confirmed-true.

## [2026-06-12] curation | temporal-abstraction concept page (maintainer via Robbins/Bergson)

Events-not-ticks critique routed: converging lines table (options, H-JEPA,
Director, Zacks event segmentation, action chunking); Zacks mechanism directly
implementable (segment at prediction-error spikes — boundaries for free);
identified as the DEEP fix for exp-0005 horizon blindness and as the same
binding problem as language-grounding. Cheap diagnostic probe pre-sketched
(boundary alignment on DoorKey trajectories). 5 sources queued.

## [2026-06-12] curation | milestone: UTD x4 — first PPO defeat at equal env budget (exp 0012)

9 runs close the literature-recipe arc: resets conclusively dead (5 variants <=
ctrl across 0011/0012 — both Nikishin and Qiao assume high-UTD overfitting; we
were UNDER-trained); UTD x4 (no resets) = 63% mean / 80% peak vs PPO 37% and ctrl
40% — bar (ii) met, rung 2b performance criterion cleared at equal env steps.
Bar (i) no-crash still fails (80->30): interference persists at higher amplitude;
trunk remains the only unprotected component. Exp 0013 pre-registered: UTD
baseline + trunk-freeze arm (+optional round-0-at-x1 arm). retention.md fix table
updated. No paid resources needed (~80-min desktop batches).

## [2026-06-12] curation | AUTONOMOUS MODE ACTIVATED (maintainer, conditional)

Standing instruction: if the maintainer doesn't respond after exp 0013 concludes, continue
the research cycle autonomously (/research-cycle skill — identical loop, no
per-iteration go). Stop gates, per the maintainer: (1) hard wall that survives the
literature gate (no resolution found AND no further research to ingest — i.e.,
the skill's 2-strike stall rule across ALL queued threads), (2) token exhaustion.
All other guardrails unchanged: desktop-only spend, current-rung scope, honesty
rules, milestone discipline, PAID-RESOURCE flags recorded but not acted on,
owner:human pages get proposals only. Reports: milestone commits as journal +
state-of-the-night LOG entry.

## [2026-06-13] curation | milestone: interference localized to the encoder (exp 0013) [autonomous]

9 runs: encoder-freeze@r2 stops the crash phenomenon in all 6 frozen-arm seeds
(fenc monotone-rising, peaks at final round, mean 47% unconverged; ftrunk stable
but capped 40% — GRU plasticity needed; warm 43% — round-0 primacy dodged, crashes
return with encoder free). The 5-experiment retention arc resolves: interference =
ENCODER DRIFT. Literature gate: 2310.07418 (ICLR24) localizes to the critic in
model-free visual RL — tension recorded, ingestion queued (augmentation lever
noted). Exp 0014 pre-registered + launching: freeze-round sweep (fenc@3, fenc@4,
3 seeds each, same budget) — if both bars clear, propose rung-2b closure to the maintainer
and pivot to the actor thread.

## [2026-06-13] ingest | Ma et al. ICLR24 plasticity paper — the 0013 tension resolves [autonomous]

Their encoder-stays-healthy + frozen-pretrained-encoders-suffice findings
COMPLEMENT our encoder-drift result (different pathology — dormancy vs drift —
same prescription: stop training the encoder once competent). Critic-bottleneck
mechanism is TD-specific (our MC values dodge it). Imports queued for exp 0015:
FAU per-module logging, Adaptive-RR scheduling atop our UTD finding.

## [2026-06-13] STATE OF THE NIGHT [autonomous]

Threads advanced since the maintainer's last message:
- exp 0013 CLOSED: interference localized to the ENCODER (freeze @r2 -> 6/6 seeds
  crash-free; GRU must stay plastic; warm arm confirms round-0 primacy dodgeable).
  Milestone 70ff51d.
- Literature gates run: Ma ICLR24 ingested (tension resolved — frozen encoders
  legitimized; FAU + Adaptive-RR levers queued); progressive-freezing literature
  imported (FreezeOut/LayerLock principle).
- exp 0014 CLOSED: freeze-timing frontier mapped (40/47/53/60/63% across
  ftrunk/fenc@2/@3/@4/free); residual crashes = GRU drift. Milestone 6833495.
- exp 0015 RUNNING: staged freeze (enc@2+gru@4, enc@2+gru@5), both bars targeted;
  --freeze2 mechanics shipped + smoke-verified.

Decision queued for the maintainer: if 0015 clears both bars -> rung-2b closure proposal
(ladder is owner:human) + pivot to the actor thread. If it caps like ftrunk ->
budget-tier extension + PPO re-baseline proposal instead.
No paid resources used or needed. No stop gates approached.

## [2026-06-13] curation | milestone: staged freeze counter-outcome — frontier confirmed (exp 0015) [autonomous, pre-maintainer-return]

g4/g5 staged enc+GRU freeze caps at 30-37% (< ftrunk 40% < fenc@4 60% < utd 63%):
freezing the GRU always recovers the ftrunk ceiling. Stability<->performance is a
genuine frontier at this budget, not out-tunable. Desktop suspended mid-run
overnight, resumed clean (both freezes fired, no corruption). KEY STRATEGIC NOTE:
rung-2b exit criterion (beat model-free on sample efficiency) was already MET at
exp 0012 (63% vs PPO 37%); crash-free "both bars" was self-imposed extra rigor,
now characterized. Recommendation logged for the maintainer: declare 2b met, pivot to actor
thread. Autonomous mode PAUSED — the maintainer returned. Awaiting their strategic call.

## [2026-06-13] curation | hierarchy-and-credit concept page (maintainer brainstorm) + rung-2b MET accepted

The maintainer's credit-assignment question routed to a new concept page: flat value head
already does implicit local credit (exp-0004 monotonic value gradient = 63%), but
explicit reusable subgoals need hierarchy (H-JEPA/options). Conditional-subgoal
constraint -> unsupervised discovery (bottlenecks/Director/empowerment), never
hardcoded. KEY DESIGN DIRECTION: the Mode-1 actor we distill should be hierarchical
(manager-worker), unifying actor + hierarchy + temporal-abstraction + language
threads. 5 sources queued (Director, FuN, Option-Critic, empowerment, DIAYN).
The maintainer accepted rung-2b as MET (63% > PPO 37%, stable) — pivot to actor confirmed.

## [2026-06-13] curation | architecture-testing tier ladder added (maintainer)

hierarchy-and-credit page gains a 3-tier testing ladder: (1) Empty/DoorKey =
mechanism debugging (fast loop); (2) MiniGrid KeyCorridor/ObstructedMaze/MultiRoom
= cheap flat-vs-hierarchical credit A/B (negative = cheap kill, positive =
necessary-not-sufficient); (3) Crafter = the decisive skill-reuse claim. Principle:
cheapest env that reveals the effect; never debug architecture on Crafter.

## [2026-06-13] curation | compute efficiency elevated to stated operating principle (maintainer)

Generalized the tier-ladder into a lab strategy: architecture and compute-efficiency
are the same axis (judge architectures by capability-per-FLOP slope; our world-model
bet IS an efficiency bet). For a home lab efficiency is the entire moat. Rules:
minimum-SUFFICIENT-scale (compute analog of whitepaper's minimum-needed-context),
measure slopes not endpoints, literature-first/pre-registration as efficiency gates.
Added to compute-strategy.md + PROCESS.md research loop.

## [2026-06-13] curation | language-grounding sharpened (maintainer): order of operations + verification battery

Two refinements: (1) language is NOT the cause of the missing "get the key" subgoal
— temporal abstraction is; language LABELS pre-existing nameless abstractions, so
grounding rides on top of the hierarchical actor (order: concepts/events first,
words attach). (2) real language = learned binding to OUR latent geometry, not a
plugged-in LLM (whose words are grounded in text stats, not this agent's
experience). Added the grounding-verification battery (imagination match,
cross-modal probe, REFERENT SWAP = gold standard / why Messenger-RTFM shuffle,
compositional zero-shot, modality transfer). First grounding experiment = built
around referent swap on a cheap env; Crafter is payoff not proving-ground.

## [2026-06-13] curation | capability-map page — locking the capability separation (maintainer)

Consolidation: single orientation index of the 7 distinct capabilities (world
modeling, flat credit, temporal abstraction/hierarchy, language grounding, language
installation, trust-weighted imagination, self-generated hypotheses) — problem each
solves, dependency order, status, test env. Explicitly records the separations that
keep blurring (#3 subgoals != #4 language; #4 grounding != plugged LLM; #5
installation != #4 grounding). owner:human (a roadmap-level artifact).

## [2026-06-13] curation | run profiled; acceleration verdict (maintainer)

profile_run.py: training 79% (launch-bound), MPC collection 16%, eval 4%, env 0.4%.
Verdict: faster env is pointless (0.4%); no big easy wins (TF32 1.02x, bf16 1.23x,
compile 1.33x but CUDA-graph capture fights our freeze/reset optimizer rebuilds).
Already fleet-efficient via 6-wide parallelism. Shipped TF32 default + --amp opt-in
(off by default for fp32 comparability); deferred torch.compile to the actor/Crafter
phase (bigger models, no reset/freeze pattern). compute-strategy.md updated.

## [2026-06-13] milestone | exp 0016: planner distillation fails (BC compounding error) → on-policy

Flat actor distilled from the 80% planner via BC: train_acc ~54% (stochastic CEM
target), eval ~0% (O(eT^2) compounding error on the 25-step brittle chain). Clean
negative; literature gate confirms (DAgger/Ross-Bagnell). Also surfaced an
eval-rigor finding: teacher 80%(10 eps, seeds 10000+) vs 25-30%(20 eps, seeds 0-19)
— headline numbers are high-variance point estimates; standardize >=20 eval eps on
a fixed seed set. Verdict: go on-policy. Exp 0017 = imagination actor-critic
(Dreamer-style; on-policy, no teacher — fixes all three failure modes, and it's the
Crafter-path actor anyway). New code: Actor, ActorAgent, distill_actor.

## [2026-06-13] milestone | exp 0017: imagination AC exploits the model — missing continue predictor

Dreamer-style actor-critic flywheel built + ran (3 seeds, fast — actor-collection
cheap). Clean negative: imagined_return 2.0/3.0/0.48 vs eval 0% all seeds = model
exploitation. Root cause: we omitted Dreamer's CONTINUE PREDICTOR — imagination has
no termination, so the actor farms goal reward across the un-terminated horizon
(imagined return >> real max 1.0). Literature gate confirms (DreamerV2/3 discount/
continue head). The rest works (WM, on-policy AC, fast actor-collection). Exp 0018:
add continue predictor, train on replay dones, discount imagined returns by
cumulative continue prob. Single-variable fix.

## [2026-06-13] curation | generative-vs-predictive concept page (maintainer, LeCun vs Xing debate)

Sharpening: generative-vs-latent (what you produce) and hallucination/exploitation
(intrinsic to optimizing any learned model) are ORTHOGONAL axes. Empirical anchor:
exp 0017 proves latent-only imagination ALSO hallucinates (no decoder, still farmed
fake reward) → verification mandatory regardless. Generation isn't needed to ground
words (contrastive/CLIP suffices) but IS needed to imagine experience from text
(arch 2b); the latent→decode→re-encode round-trip's only honest justification is
knowledge import. Unifying principle "real outweighs imagined" links trust-weighted
replay + continue predictor + corroboration. Roadmap: LeCun-pure rungs 1-4,
generative/Xing layer at 5b gated by verification.

## [2026-06-13] curation | imagination-as-novelty (maintainer): hallucination is the raw material of ideas

Reframe added to generative-vs-predictive: an original idea = a hallucination that
survives verification (variation+selection = creativity; Popper/Campbell/Dennett/
Schmidhuber). Value is in what survives, not the generation (exp-0017 bad-idea
killed by 0018 = refutation). Creativity-safety tradeoff: never-hallucinating =
uncreative; target = regulated imagination (entropy/curiosity + verify). Closure:
agent creativity loop ≡ our research loop; capability #7 is this internalized.
Suppression and creativity are ONE mechanism, two aims.

## [2026-06-13] curation | architecture-strategy concept page (maintainer): modes, interfaces, reusable components

Design brainstorm routed. Key reframes: (1) external-learning is a MODE (source +
verification regime), not the default loop = LeCun's configurator; (2) monolithic-
vs-modular is the wrong axis — the real decision is the stable INTERFACE
(belief-state), which makes module-count reversible/per-component; (3) reuse a
pretrained Minecraft WM as a frozen module — backed by knowledge-import+compute AND
our own encoder-freeze finding (components want different lifecycles). Decide
joint-vs-staged empirically at rung 3/5b. Flagged: FULL read of lecun-2022-path is
now design-critical (the maintainer re-derived its modular architecture 3x tonight).

## [2026-06-13] curation | generative-vs-predictive: does imagination need generative AI? (maintainer)

Goals-divergence with LeCun (control-accuracy vs human-like-cognition). Sharpened:
imagination != pixel generation (exp 0017 imagines latent, no decoder); generative
is PRACTICALLY valuable for knowledge-import/tutorials + interpretability, and
STRUCTURALLY for 3D occlusion/amodal-completion (object permanence) — the strongest
real case. Corrected over-attribution (3D-hard != generative-key; DreamerV3 does 3D
latent). Resolution: latent core + optional generative module per mode; testable
(text->latent direct vs render-and-re-encode at Messenger scale).

## [2026-06-13] curation | dreaming concept (maintainer) + lazy-generation refinement + text→video→latent

Dreaming page (backlog): the maintainer's relaxed-constraints + weak-writeback = Hoel
overfitted-brain (verified arXiv:2007.09560) — dreams = anti-overfitting augmentation;
over-training triggers dreams = OUR primacy bias. Two functions: creativity-seeding +
latent-dream-augmentation (cheap testable retention regularizer, no gen-AI). Also:
generative-vs-predictive refined — generation is LAZY/attention-gated (latent tracks
persistence/object-permanence; render on demand only), demoting last night's
continuous-amodal-generative claim (maintainer's correction); text→latent solved by
composition text→video→encoder (seams: domain video-gen, encoder robustness to
generated frames, trust-gate). Hoel queued.

## [2026-06-13] milestone+curation | exp 0018 partial (deeper exploitation) + word-associations

Exp 0018: continue predictor partially tamed exploitation (s0 2.06->1.13) but s1
worsened (5.4) and eval still ~0 — actor exploits OTHER off-distribution model
inaccuracies; root = deterministic WM can't imagine the 25-step chain (compounding
error), no honest success gradient. Lack Dreamer's stochastic latents. Fork logged
for the maintainer: (a) deepen (stochastic latents / H=5 / uncertainty penalty) vs (b) DAgger
on-policy distillation (reuse working planner, sidesteps imagination exploitation) —
recommend cheap H=5 check then DAgger. Also: language-grounding gains word-association
section — association matrix = word embedding (Levy-Goldberg, queued); don't rebuild
as metadata (import + bind to grounded latent); associations enable TRANSITIVE
grounding (smelt↔furnace bridges ungrounded tutorial words to experience).

## [2026-06-13] lint | full-KB curation sweep | 11 issues, 9 fixed, 2 flagged

Full sweep (53 pages; no >90-day staleness — KB is ~1 day old). Concept clusters
audited for duplication/contradiction via 3 parallel readers (architecture /
imagination / grounding-hierarchy): **no contradictions; clean abstraction-level
separation** — the rapid 06-12/13 brainstorm pages do not duplicate each other
(agent-architecture=descriptive, architecture-strategy=design, capability-map=roadmap;
generative-vs-predictive=philosophy anchor, dreaming=testable retention regularizer).
Link graph re-closed: every wikilink resolves except the registered wanted page
world-models-1803.10122; only remaining orphan is the meta-ADR 0002 (acceptable).
Fixed: INDEX (added 0019, corrected 0018 running→done-partial, reordered 0016);
retention.md (fix-table stopped at 0013 → extended with 0014/0015 frontier + Ma 2024;
both "open questions" answered by 0011-0013; added 7 missing links incl. orphaned
[[ma-plasticity-2024]] and [[dreaming]]); agent-architecture L3 interference "(open)"
→ localized-to-encoder/0013; orphan inbound links for 0019 (from 0018), 0015 (from
0014), [[ijepa-2023]] (from jepa); language-grounding Links gained
[[hierarchy-and-credit]]/[[temporal-abstraction]]. FLAGGED for the maintainer (owner:human, no
silent rewrite): (1) "pension" terminology in exps 0014/0015 — reconstructed as a
deliberate metaphor from the [autonomous]-run agent ("pension off" = retire a network
component = freeze it permanently; encoder + GRU are the "two plasticity taps / drift
sources" each needing to be pensioned). Coherent but obscure; keep-or-normalize is
The maintainer's call. (2)
capability-map world-modeling status says "rungs 1-2 / SIGReg" — predates the
0017-0019 imagination thread, may want an exp-range refresh.

## [2026-06-13] curation | structural: new design/ directory (maintainer-approved plan)

Resolved a shelving seam the maintainer spotted: `concepts/` conflated field-knowledge
explainers ("what is X") with this project's own design/strategy artifacts ("OUR X"),
so titles like "capability-map" mismatched the generic-concept expectation. The
`scope: local|shared` flag was meant to carry this distinction but isn't visible when
browsing by directory. Fix: new top-level `knowledge/design/` for project synthesis.
`git mv`'d 5 unambiguous pages out of concepts/ → design/ (capability-map,
agent-architecture, architecture-strategy, compute-strategy, environment-ladder);
wikilinks survived (basename-resolved), 6 path-based refs updated (INDEX ×5 → new
"Design & strategy" section, ROADMAP, sweep SKILL, gpu_bench.py, CLAUDE.md). SCHEMA.md
(owner:human) directory table gained a `design/` row and `concepts/` was narrowed to
"field concept". Hybrid pages (retention, temporal-abstraction, generative-vs-predictive,
hierarchy-and-credit, language-grounding, dreaming) deliberately LEFT in concepts/ —
reclassifying them is a separate judgment call (deferred). environment-ladder keeps its
ADR-0003 (frozen decision) + design-page (living spec) split. Also resolved flag (1)
above: "pension" wording normalized to "freeze/frozen" across exps 0014/0015 (maintainer
approved a clearer word). Link graph re-verified closed post-move.

## [2026-06-13] curation | agent-architecture synced to code: 4 → 5 layers

Checked design/agent-architecture against the actual src/world_model/ (Explore map).
The page was an accurate snapshot of the exp-0001–0010 MPC+value system but stale
across all layers vs the 0016–0019 code: missing the learned actor (second acting
mode), the stochastic RSSM world model, the continue + reconstruction heads, and the
entire imagination actor-critic training loop. Per the maintainer ("the model should follow the
architecture, not stick to 4 layers"), promoted the imagination actor-critic to a real
**L4 BEHAVIOR** layer (policy gradient + λ-return critic w/ EMA target); the flywheel
moved L4 → **L5**. Updates: title now count-agnostic ("Nested Optimization Layers");
mermaid redrawn (5 subgraphs, two acting modes L1-search + L4-amortized feeding L5);
algorithm table gained the L4 row + RSSM/continue in L2/L3 + 0017-0019 failure modes;
prose contrasts Dreamer(no L1)/TD-MPC2/PPO(model-free) against our layering and locates
model-exploitation at L4. Renumber rippled to one cross-ref: capability-map "4 → 5
optimization layers" (the (L1)/(L2) maps in lecun-2022-path are stable, untouched).
FLAGGED (separate, not fixed): lecun-2022-path:38 calls stochastic latents "future" —
now built (RSSM, 0019); that page wants a refresh too.

## [2026-06-13] queue | PAN / GLP (Eric Xing) — LLM-as-latent-backbone

Queued two papers after a discussion sparked by the maintainer watching an Eric Xing talk:
PAN (arxiv:2511.09057) and its position paper "Critiques of World Models"
(arxiv:2507.05169), under a new QUEUE subsection "LLM-as-latent-backbone (PAN/GLP)".
Why they matter to us: PAN's Generative Latent Prediction (encoder→latents, an LLM
backbone as the latent dynamics model conditioned on language actions, diffusion
decoder→video) is the published answer to the LLM↔latent integration we'd only
sketched — and Xing argues FOR a generative decoder, the explicit counter to JEPA on
our generative-vs-predictive axis. Logged the key ingest question so it isn't lost:
does PAN show only (a) static pretrained knowledge in the backbone, or (b) test-time
acquisition (read a novel tutorial → new latents → changed behavior)? (b) is our edge
case; if PAN demonstrates it the roadmap shifts. No wiki page cites these yet (queue
only). Pairs flagged: concepts/generative-vs-predictive, design/architecture-strategy.

## [2026-06-13] queue | Teams to watch — Hassabis/DeepMind + Fan-Yun Sun/Moonlake

Added a new QUEUE subsection "Teams to watch" after the maintainer asked who Xing credits in the
DataCamp podcast "Will World Models Bring us AGI?" (youtube VNyLNZunv9E). Transcript-
confirmed: Xing names **Demis Hassabis / DeepMind** as near-perfectly aligned on what a
world model (and virtual cell) is and how to build/test it, and expects "something
fancier and disruptive in the next few months" (public releases trail internal work).
Second lead surfaced in the same discussion: **Fan-Yun Sun / Moonlake AI** (ex-Stanford
SAIL; Manning/Goodfellow orbit) — causal, multimodal, interactive, EFFICIENT world
models over blind scaling; aligns with our generative-vs-predictive + Causal-JEPA
threads. Both are groups-to-track, not papers; ingest specific outputs as they land.
First-ingest pointers logged (Genie line for DeepMind; latent.space/p/moonlake for
Moonlake).

## [2026-06-13] milestone+curation | exp 0019 RSSM — eval off zero (partial win)

Exp 0019 (RSSM stochastic latents) closed as a **partial success**, 3 seeds DoorKey-6x6,
commit b8fdb62. Eval rose off zero for the first time on the imagination line: best
s0=0.00, s1=0.55, s2=0.35 (both winners peak round 5, regress round 6). s1 checkpoint
reproduces 10/20 under play (new reactive RSSM agent path in play.py + RSSMActorAgent.
from_checkpoint). Key finding written to experiments/0019: **RSSM fixed policy quality,
not value calibration** — imagined_return stayed inflated 0.7–4.3 (hypothesis predicted
≤1.0; refuted), the SAME range as exploited 0017/0018, yet now coexists with real
competence. Stochasticity stops collapse onto one fake trajectory (relative action
ordering tracks reality) without bounding the absolute value scale. Two open problems:
peak-then-collapse round 6 (retention signature; freeze-round 2 insufficient → next
fork a) and seed variance (s0 never ignited). Process lesson recorded: a smoke confirms
mechanism wiring, not steady-state behaviour — the earlier "imagined_return ~0.1, killed"
claim was a premature smoke-test promotion (inflation re-emerges after 4k×7 AC updates).
Pages: experiments/0019 Result+Lesson filled (pre-registered hypothesis left intact).
Code: play.py + agents/rssm_agent.py (watchable checkpoints). Next: fork (a), the
round-6 collapse, via retention/plasticity levers.

## [2026-06-13] milestone+curation | exp 0020 actor-critic reset — REFUTED (fork a closed)

Fork (a) tested and falsified fast. Per-round shrink-perturb reset of the actor-critic
(α=0.5, `--ac-reset`) to attack 0019's round-6 collapse instead **suppressed learning**:
best_eval s0/s1/s2 = 0.00/0.05/0.10 vs 0019's 0.00/0.55/0.35 — the reset erased the
competence on the two seeds that had it. Mechanism: 0019's policy *consolidates*
gradually across rounds (s1 0.15→0.20→0.55), and an α=0.5 reset halves that each round.
So the round-6 collapse is NOT behaviour-layer plasticity loss (clean falsification).
Critic reset made `imagined_return` inflation worse (s2 → 7.3), confirming **value
miscalibration** is the load-bearing problem → redirect to fork (b): ground/bound the
imagined value (DAgger real-rollout anchoring / KL / shorter horizon), not the behaviour
layer. Pages: experiments/0020 Result+Lesson (status REFUTED); pre-reg hypothesis intact.
Caveats logged not chased: only aggressive α tested; 0019's 0.55→0 may be partly eval
variance. Code already committed d03eeba (the --ac-reset flag); this entry records the
outcome. Live logs worked this run (PYTHONUNBUFFERED in the dispatched commit).

## [2026-06-13] scout+queue+curation | value-calibration toolkit; exp 0021 critic-on-replay

After exp 0020 redirected us to value miscalibration, the maintainer asked to accelerate by reusing
established learnings. Lit sweep (4 searches) mapped the published toolkit for inflated
imagined value / model exploitation — THE central MBRL failure mode, not exotic:
b1 DreamerV3 **critic-on-replay** (β_repval 0.3, critic grounded in real returns) ·
b2 MBPO short *branched* rollouts (1906.08253) · b3 DreamerV3 percentile return-norm +
two-hot critic · b4 MOPO/MOReL/CBOP uncertainty pessimism. New QUEUE subsection
"Value calibration / model exploitation" (MBPO, MOPO, MOReL, CBOP queued; ids from search,
verify at ingest). **Curation fix:** papers/dreamerv3-2023.md had MISSED the critic-on-replay
detail (had two-hot/EMA-target/replay-ratio but not β_repval) — added the bullet; that was
the load-bearing fact for our fix. Pre-registered + implemented exp 0021 (b1): `--repval`
flag in train_rssm, critic also regressed on real burn-in returns (reuses beliefs already
computed; real_bel detached → grad to critic only). Flag defaults off → baseline ≡ 0019.
Prediction: imagined_return falls 2–4 → ~1 and ignition broadens. Smoke-confirmed wiring.

## [2026-06-13] decision | ADR 0006 accepted — Crafter (original) over Craftax for rung 3

The maintainer accepted ADR 0006. Crafter (original, PyTorch-native) for rung-3 first contact, NOT
Craftax. Three deciding facts: (1) Craftax 257x is a model-FREE PPO (1B-step) figure; we
are model-based/sample-efficient (~1e6 steps) so env-stepping is not our wall; (2) Craftax
is JAX-only, vs ADR 0001 PyTorch (interop friction / full rewrite), and 0001 pre-registered
the only revisit trigger as heavy parallel-env training; (3) we want pixel obs (frozen-
encoder thesis), and Craftax fast mode is symbolic. Craftax kept as documented escape hatch
with a measured trigger (env-stepping >=~30% wall-clock, or sample-hungry pivot). Next:
Crafter env wrapper behind the MiniGrid interface -> hello-Crafter baseline.

## [2026-06-13] curation | JAX rewrite revisit-triggers consolidated (maintainer asked)

Added a consolidated "JAX rewrite revisit triggers" subsection to design/compute-strategy.md
(extends ADR 0001; cross-links ADR 0006). Key framing: JAX advantage SHRINKS up the ladder —
real envs cannot be JAX-fused (Atari C++, Minecraft Java, real world), so the parallel-env
speedup exists only at the Crafter rung; after that the question is moot. Triggers (none
current): TPU access, sample-hungry pivot, env-stepping >=30% wall-clock. Even then: surgical
(per-experiment JAX env/agent via dlpack hybrid), not a stack rewrite. JEPA/WM stays PyTorch.

## [2026-06-13] milestone | rung-3 first contact — Crafter env wrapper

ADR 0006 accepted -> built the Crafter env wrapper (src/world_model/envs/crafter.py): a
gymnasium adapter over old-gym crafter.Env emitting CHW float32 obs in [0,1] like the
MiniGrid wrapper (same encoder/agents/training consume it unchanged). done split into
terminated (death, info.discount==0) vs truncated (length); seeded reset rebuilds the env
for reproducible eval worlds (~0.02s); info.achievements (22-dict) is the score basis.
crafter 1.8.3 added (light deps, no second CUDA framework). Smoke test added (19/19 pass).
Hello-Crafter random rollout: 283 steps to death, 1/22 achievements (wake_up), reward ~0.1
-- the expected random floor; env+reward+achievement+death/timeout all flow. New page
environments/crafter.md (incl. the Crafter->Minecraft proxy->target transfer relationship
The maintainer asked about: method+frozen-encoder transfer, WM weights do not; JAX corollary). INDEX
+ environment-ladder additive pointer. Next: 64x64 encoder (frozen DINO/JEPA per lean) +
port the calibrated imagination loop; harden DreamerV3 recipe here (denser rewards).

## [2026-06-13] experiment+curation | exp 0022 — frozen DINOv2 encodes Crafter state (encoder bet validated)

Before building the frozen-encoder pipeline, validated the gating risk (does natural-image
DINO transfer to Crafter pixel-art?). scripts/probe_dino_crafter.py: linear probe on frozen
DINOv2-S CLS features (600 random-play frames) -> "which materials in the 9x9 view" (free
labels from info.semantic). Result: mean test acc 0.981 vs majority 0.804 (+0.177 lift),
per-material 0.93-1.00; shuffled-label control collapses to 0.748 (signal real). Frozen DINO
VALIDATED as the Crafter encoder, no fine-tuning. Greenlights DINO-WM-style build (frozen
DINOv2 + small RSSM dynamics). Frozen => drop SIGReg, cache embeddings in replay, immune to
primacy/drift. Pages: experiments/0022, crafter.md encoder bullet, INDEX (also caught up
0019-0021 statuses). Next: FrozenDinoEncoder module + wire into the RSSM flywheel for Crafter.

## [2026-06-13] milestone | frozen DINO + RSSM imagination loop wired on Crafter (train_crafter v1)

Wired the validated frozen DINOv2 encoder (exp 0022) into the calibrated RSSM imagination
loop (0019/0021) for Crafter: new src/world_model/train_crafter.py reuses wm_train/imagine_ac/
critic-on-replay unchanged, swaps in FrozenDinoEncoder (embed_dim 384, frozen -> excluded from
opt), make_crafter_env (17 actions), and an achievement/reward eval. SIGReg dropped (frozen
cant collapse; guarded wm_train with lam>0). v1 encodes in the training loop (correctness
first); embedding caching (encode once at collection, skip encoder in WM updates) is the next
optimization and is needed before a full-scale run. Tiny CPU smoke: full flywheel runs end to
end, imagined_return sane (0.10/0.18 = calibrated, critic-on-replay carries to Crafter), ckpt
saved. MiniGrid train_rssm untouched (lam>0 guard preserves it). Next: embedding cache -> first
real desktop Crafter run; then harden DreamerV3 recipe (two-hot/symlog/percentile-norm).

## [2026-06-13] milestone | embedding cache for the frozen-encoder Crafter loop

Implemented freezing payoff: train_crafter now stores DINO EMBEDDINGS (float32) in replay,
not pixels, encoding each frame ONCE at collection (new collect_embed) and passing an Identity
encoder to wm_train/imagine_ac -> zero DINO forwards in WM/AC updates (~98% of forwards
eliminated; training-update cost no longer DINO-bound). ReplayBuffer gained an obs_dtype param
(float32 stores embeddings verbatim, no *255 round-trip; default uint8 unchanged -> MiniGrid
path untouched, 20/20 tests). Correctness-equivalent: cached round-0 smoke is NUMERICALLY
IDENTICAL to the v1 uncached run (0.393/0.1687/0.096) -- same embeddings, same training, just
encoded at collection. Float-buffer roundtrip test added. Embeddings are ~8x smaller than the
image buffer too. Makes a full-scale desktop Crafter run feasible. Next: first real run +
harden DreamerV3 recipe.

## [2026-06-13] validation | first Crafter learning signal + value inflation surfaced

Local-GPU validation of the full train_crafter pipeline (cached frozen DINO + RSSM + critic-
on-replay), rounds 0->1 in 75s on the throttled laptop: eval_reward 0.10->1.10, achievements
1->2 -- LEARNS above the random floor; cache works (no OOM, training not DINO-bound). BUT value
inflation resurfaced at Crafter scale: imagined_return 9.95->21.67, critic_loss 14->35 (vs ~1
calibrated on MiniGrid). Denser/larger-scale rewards overwhelm plain-MSE critic + critic-on-
replay (beta_repval 0.3) -- model exploitation (cf 0017/0018) at Crafter scale. Decision: do
NOT dispatch a long run yet (would just exploit the model); harden the DreamerV3 recipe FIRST
(symlog + two-hot distributional critic + percentile return-norm) -- the deferred step, now
empirically justified by this run. crafter.md pipeline-status + what-next updated.

## [2026-06-13] experiment | exp 0023 two-hot distributional critic — Crafter value bounded

Value inflation on Crafter (imagined_return 21, critic_loss 35) attacked with DreamerV3 two-hot
distributional critic (models/twohot.py: 255 symlog bins, two-hot CE; forward=symexp expectation,
drop-in for ValueHead). imagine_ac gated via _critic_loss (twohot_loss if available else MSE) ->
MiniGrid path byte-identical (21/21 tests), Crafter gets distributional loss. Local smoke: same
config that gave MSE imagined_return 21 now gives 6.1->5.4 STABLE, critic_loss 2.5->2.2 -- blow-up
gone. Eval flat in 2 noisy rounds (calibration-delays-ignition, cf 0021); climb test needs a long
run -> dispatching overnight. Pages: experiments/0023.

## [2026-06-13] experiment | exp 0023 result — first Crafter LEARNING run (two-hot critic)

Overnight desktop run (2 seeds, 10 rounds) of the full rung-3 pipeline (frozen DINO + RSSM +
critic-on-replay + two-hot critic + embedding cache). RESULT: learns and climbs off the random
floor (1 ach) to ~2.5-3 achievements / reward ~2 (20x random reward), no collapse, NO ignition
stall (the 0021 calibration-kills-exploration risk did not materialize -- Crafter reward dense
enough). Both seeds plateau at the shallow tree (~3 ach). Two-hot fixed the CRITIC (loss 1.5-2.7
stable vs MSE 35) but imagined_return stayed inflated/noisy (s0 20->5-10 self-calibrating; s1
17-22 w/ a round-8 collapse to 0.56) -- the REWARD head is the remaining over-prediction source.
Best reward s0 2.10@r6, s1 1.97@r2. Next (reordered by data): exp 0024 DINO patch tokens (richer
spatial features for navigation/gathering, attack the plateau) BEFORE the reward-head fix, since
the plateau (not value) is the headline blocker. Pages: experiments/0023 Result+Lesson, INDEX.

## [2026-06-13] diagnostic | exp 0023 agent is STATIONARY — reward-head exploitation (maintainer spotted it)

The maintainer noticed in the live viewer the player never moves relative to the (egocentric-scrolling)
world. Quantified: 3 episodes, unique_tiles=1, player_pos bbox_span=(0,0) -- the agent NEVER
changes position. Action histogram: do (chop adjacent), mk_iron_sword spammed 38-48x/ep (a no-op:
no iron), sleep, place_plant. All 3 achievements are reachable without moving (wood/table/plant
from spawn). The mk_iron_sword spam = MODEL EXPLOITATION: the reward head over-predicts reward
for that useless action in imagination -> THIS is the inflated imagined_return (17-31); the
actor maximizes broken imagined reward by spamming the fake-rewarding action instead of exploring.
RE-PRIORITIZES the plateau diagnosis: it is a degenerate non-exploratory policy driven by
reward-head inflation, NOT primarily representation. => reward-head fix (symlog/two-hot REWARD
predictor) jumps to top priority (kill spurious action value -> remove spam incentive -> free
exploration), likely + stronger entropy. exp 0024 (patch tokens) still a clean representation-axis
test; keep it running. Movement is not broken (random agent moves; actions pass through).

## [2026-06-13] experiment | exp 0024 patch tokens — navigation unlocked (seed-dependent); overturns the null

DINO patch tokens (cls+patch) vs the 0023 plateau. Aggregate looked like a wash (best reward
s0 2.47, s1 1.60) BUT behavior_report revealed a huge split: s0 MOVES (moved_frac 0.12, span
17.7 tiles, move_actions 0.42) and builds a TABLE (place_table -- real tech-tree progress);
s1 collapsed to the 0023 stationary policy (do:0.96, moves 0
## [2026-06-13] experiment | exp 0024 patch tokens — navigation unlocked (seed-dependent); overturns the null

DINO patch tokens (cls+patch) vs the 0023 plateau. Aggregate looked like a wash (best reward
s0 2.47, s1 1.60) BUT behavior_report revealed a huge split: s0 MOVES (moved_frac 0.12, span
17.7 tiles, 42pct movement) and builds a TABLE (place_table -- real tech-tree progress); s1
collapsed to the 0023 stationary policy (do:0.96, moves 0pct). => Representation IS part of the
bottleneck (overturns pre-reg null): global CLS cannot localize resources to navigate; spatial
patch tokens give the where-signal and s0 learned to walk + craft. Seed-dependent because the
reward exploitation is unfixed (imagined_return 8-22) and tips s1 into collapse. BOTH levers
matter -> exp 0025 combines patch tokens + two-hot reward head, 3 seeds. PROCESS WIN: the
behavior QA gate caught the s0/s1 split that aggregate eval masked. Pages: 0024, INDEX.

## [2026-06-13] curation | why Crafter onboards easier than DoorKey + plateau is a scaling story (maintainer)

Recorded an honest decomposition on crafter.md of why Crafter feels easier than the DoorKey
struggle: (1) reward DENSITY (built-in achievement curriculum vs DoorKey single sparse goal --
biggest factor), (2) frozen pretrained encoder (skips the from-scratch representation instability
that was half the DoorKey fight), (3) our algorithmic improvements (prevent failures, not the main
ease driver). Plus the maintainer correcting an over-dramatic "DoorKey wall returns" framing: the plateau
is NOT a fundamental wall -- (a) current ~3 plateau = the exploitation BUG (0025 fixes), (b)
mid-tree = mostly COMPUTE SCALING (DreamerV3 reaches it at ~1M steps; we are ~10x under-trained),
(c) only deepest chains might want hierarchy, as an efficiency lever. For a compute-efficiency lab
efficient scaling IS the research. Discipline: MEASURE the scaling slope once 0025 lands.

## [2026-06-14] curation | ROADMAP.md refreshed to current state (rung 3 / exp 0025 era)

ROADMAP was stale (stopped ~exp 0008). Rewrote to reflect the journey: rungs 1-2 done
(incl. the imagination actor-critic line + retention thread), rung 3 (Crafter) current
with the frozen-encoder + calibrated-imagination + value-hardening + behavior-QA progress
and the next steps (consistent navigation, scaling slope, hierarchy). Depersonalized for
the public repo (no identity/hardware leaks; proper KB links). Phases map to ladder rungs.

## [2026-06-14] scout+queue | does our grounding testbed already exist? — CrafText/RTFM/SILG/LDD

The maintainer asked whether a custom Crafter-with-tutorials env exists or we should build it. Checked
the literature first (frontier-claim protocol). Found: CrafText (2505.11962, Craftax instruction-
following — "Crafter+language" exists, but instructions not read-to-learn-dynamics, and JAX/Craftax
per ADR-0006 wall); RTFM (right read-to-learn-dynamics paradigm, toy scale); Messenger (referent-swap
gold standard); SILG (2110.10661, unified grounded-language-game benchmark); LDD (2210.00066, method).
KEY DISTINCTION recorded: instruction-following (text=goal) vs reading-to-learn-DYNAMICS (text=rules;
our installation interest). LIKELY GAP (UNVERIFIED, needs deep protocol): rich-env read-to-learn-
dynamics. Recommendation: literature-first — ingest this cluster BEFORE building any custom env
(gives task designs + baselines + decides reuse-vs-build). Queued under a new QUEUE subsection. No
build started.

## [2026-06-14] design | two-domain split: CrafterManual grounding-env as a separate repo; this project authors requirements

The maintainer framed the grounding testbed as a SECOND context-architecture DOMAIN: a separate
standalone GitHub repo (CrafterManual / Crafter-RTFM), built by its own Claude to spec, with THIS
research project iteratively AUTHORING requirements for it. Clean inter-domain contract:
requirements flow research->tool, capability reports flow tool->research; neither reaches into
the others internals. Wrote v0 requirements catalog at knowledge/design/grounding-env-spec.md
(gate=verify-gap-first; mandatory-read-to-learn-DYNAMICS, rich+PyTorch, built-in referent-swap
verification harness, phased build with a P1 mandatory-reading validation gate). The spec is the
boundary object, owned/versioned here; implementation lives in the separate repo. Downstream of
the Crafter foundation; not started.

## [2026-06-14] milestone | exp 0025 SUCCESS — combined fix resolves the exploitation (all seeds move/craft)

Combined fix (DINO patch tokens + two-hot reward head + two-hot critic + critic-on-replay), 3
seeds, 10 rounds. best_eval_reward s0 2.35 / s1 2.47 / s2 3.47. DECISIVE: behavior_report PASSES
on ALL 3 seeds (vs 0023 all-stationary, 0024 seed-dependent) — moved_frac 0.11-0.18 (all move,
span 18-22 tiles), action entropy 1.5-2.0 (no collapse), real tech-tree (wood/stone pickaxes,
furnace), s2 6 unique achievements / reward 3.77. imagined_return calibrated 1.3-1.7 (vs 8-22),
trended DOWN. Both levers needed: patch tokens (navigate) + bounded reward (dont hallucinate
reward). The exploitation arc 0017->0025 is CLOSED. Residual: s0 minor make_iron_sword spam.
Ceiling (~3-6 ach) now a COMPUTE-SCALING question (~10x under-trained vs DreamerV3) -> exp 0026
scaling slope. Process win: behavior_report (from the maintainer s it-doesnt-move catch) was the
decisive metric; eval_reward alone looked flat. Pages: experiments/0025 Result+Lesson, INDEX.

## [2026-06-14] experiment | exp 0026 replay-ratio scaling — breadth not depth (mixed)

2x updates/round (replay ratio), 0025 recipe, 2 seeds. End achievements ROSE (s0 3.88, s1 4.00
vs 0025 ~2.25-3.4) BUT the names show it is BREADTH: union all shallow/survival (collect_wood/
drink/sapling, place_plant, eat_cow, defeat_zombie, wake_up) -- NO crafting tech-tree (no
pickaxe/table/furnace that 0025 reached). Movement DROPPED (moved_frac 0.07-0.15 vs 0.11-0.18,
top place_plant/sleep) = mild over-training toward in-place farming (retention/primacy thread).
=> replay-ratio lifts breadth/reliability (we were mildly under-trained there) but is NOT the
depth lever; dont push it higher. Depth (crafting->stone->iron) is data/exploration-bound (the
DoorKey long-horizon credit-assignment problem returning), not under-training. Caveat: crafting
rare + 2 seeds = partial seed confound. -> exp 0027 step/data scaling (more rounds at 1x replay):
does more exploration data unlock depth? If not -> exploration/hierarchy-bound. Pages: 0026, INDEX.

## 2026-06-14 — exp 0027 dispatched + lit-gate prep for the depth fork
- Dispatched exp 0027 (step/data scaling: 20 rounds, 1× replay, 2 seeds) at commit 755d902 on the desktop GPU. Tests whether crafting DEPTH is data/exploration-bound (the 0026 follow-up: replay-ratio bought breadth not depth).
- Literature gate (autonomous research-cycle step 2) for the likely 0027 fork: queued 3 new on-target anchors under sources/QUEUE.md → new section "Crafter DEPTH / achievement-hierarchy fork (exp 0027 thread)": Achievement Distillation (2307.03486, contrastive achievement hierarchy — most on-target), Learning Achievement Structure (2305.00508, structured exploration via the dependency graph), Curious Replay (2306.15934, novelty-prioritized DreamerV3 replay, +1.33× Crafter — the contrast to 0026's failed uniform 2× replay). Ingest waits until 0027 picks the fork.

## [2026-06-14] state-of-the-night | autonomous research-cycle: depth thread (0026→0027→0028)

**Threads advanced tonight (all milestoned):**
- **exp 0026** (replay-ratio scaling, 2× updates) → MIXED: breadth/reliability up, crafting
  DEPTH unchanged, mild over-training (less movement). Milestone 207ff4a.
- **exp 0027** (step/data scaling, 2× data @ 1× replay, 20 rounds) → NULL for depth: reward/
  count rose (best 3.47/3.72, count 4.38/4.62 > 0025) with movement HEALTHY (behavior_report
  PASS both; 1× replay avoided 0026's collapse), but frontier did NOT exceed 0025 (s0 stalls
  at place_table, s1 one transient make_wood_pickaxe; no stone/furnace). Milestone b9fa258.
- **Verdict:** both compute axes (gradient-steps AND env-steps) are ruled out as the depth
  lever → depth is **exploration/hierarchy-bound**, not compute-bound. Stall rule fired
  (same failure survived 2 redesigns) → switched thread from scaling to structured exploration.

**Running now:** **exp 0028 — Curious Replay** (novelty/surprise-prioritized WM replay;
arxiv:2306.15934, ingested at method depth → papers/curious-replay-2023.md). 2 seeds,
dispatched at commit 997183a on the desktop GPU. One variable vs 0027: `--curious` (WM-train
windows sampled by p=c·β^visits+(|recon+KL|+ε)^α instead of uniform). Reviewer-confirmed the
optimized loss is identical to the 0027 baseline → clean attribution. Chosen because the paper
reports its Crafter gains SPECIFICALLY on the deep nodes we're stuck on (stone pickaxe, iron).

**Recommended next decision (for the maintainer):**
- If 0028's eval union gains frontier achievements (collect_stone / make_stone_pickaxe /
  place_furnace) with behavior_report still PASS → curiosity-on-WM IS a depth lever →
  exp 0029 = extend Curious Replay to the imagination-AC burn-in sampling too (compounding).
- If 0028 is NULL (frontier unchanged) → the bottleneck is the ACTOR not discovering the path,
  not the WM not learning it → escalate to structured exploration / hierarchy: Achievement
  Distillation (2307.03486) or achievement-graph structured exploration (2305.00508), both
  queued. This is a bigger design step (achievement-conditioned policy/contrastive head) and
  is a reasonable point to want maintainer input on direction.
- Caveat to watch: DINO-embedding recon error may be lower-variance than the paper's pixel
  loss → the curiosity signal could be weak. If 0028 shows priorities barely differentiating
  (recon+KL near-uniform), that's the likely culprit, not the method.

No paid resources used (desktop GPU only). No PAID-RESOURCE flag this cycle.
