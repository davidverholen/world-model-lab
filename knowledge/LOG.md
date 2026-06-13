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

## [2026-06-12] scout | AMI pair completed (WBench id), Causal-JEPA + VL-JEPA queued

Watch-list sweep (3 searches): WBench arxiv:2605.25874 confirmed as the late-May
AMI-circle companion benchmark (theorem 2605.26379 + stress test pair; "current
models collapse under minor visual shifts") -> ami-labs page updated, QUEUE open
item resolved. New: Causal-JEPA (2602.11389, object-level latent interventions),
VL-JEPA (2512.10942, low priority). Dreamer/DeepMind side: nothing new beyond
known Genie/Dreamer state; 2026 survey blogs noted but not queued (secondary).

## [2026-06-12] curation | intellectual-lineage page (Dave's request: decades of background)

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
Dave's request: scripts/fetch_sources.sh -> 21 arXiv PDFs (183MB) in gitignored
knowledge/sources/files/ for personal reading; idempotent, documented in SOURCES.

## [2026-06-12] lint | 6-ingest checkpoint: graph closed, 1 INDEX gap fixed

Scripted pass over 45 pages: zero dangling [[links]] (cross-link graph fully
closed); INDEX was missing exp 0011 (fixed); remaining stubs (dreamer4, vjepa2)
honestly labeled, not cited as authority. Skills updated with today's craft:
ingest-source gained archive step + read-depth labels + ar5iv route;
scout-sources gained the arXiv-API batch-verification trick. Assessment: no new
skills/agents needed at current KB size; ingestion stays on-demand per the
literature-first rule (exceptions queued: Dyna, WBench).

## [2026-06-12] curation | research loop codified as top-level process (Dave)

PROCESS.md gains "The research loop": experiment-verify <-> literature-first <->
route-to-KB <-> skills-carry-procedure-not-knowledge. Point 4 sharpened from
Dave's draft: maintenance trigger is process drift, not knowledge growth (thin
skills + INDEX navigation make new knowledge reachable without skill edits).

## [2026-06-12] curation | research-loop horizon defined (Dave)

Process holds until a blocking problem has no published solution; then step 2
transforms (nearest-neighbor mapping + novel mechanism + our pages as primary
record). Noted: the model-based retention transfer (exp 0011) may already sit
on that edge.

## [2026-06-12] curation | deep-search protocol added to scout skill (Dave's G-Scholar question)

Tested live: Semantic Scholar API = the right semantic+citation tool (anonymous
tier congested; free key recommended -> env S2_API_KEY); OpenAlex = instant but
weak ranking, good citation graphs; Google Scholar = manual-only (no API/ToS).
Local embeddings index DEFERRED with trigger: build it over OUR corpus (pages +
PDF archive) at ~150 pages, not over external papers (duplicates S2).

## [2026-06-12] curation | deep-search stack re-ranked: OpenAlex primary (S2 key gated)

S2 API-key form requires institutional affiliation + rejected proton.me —
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

## [2026-06-12] scout+ingest+correction | frontier claim overturned by the wall protocol (Dave's prompt)

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

## [2026-06-12] curation | ADR 0005: Minecraft milestone (Dave) — three-signal training

New rung 5b before real-world transfer: self-play + VPT-style action-labeled video
+ text-in-world-model (open research; Dynalang/VL-JEPA direction). Ladder + ROADMAP
amended; 5 sources queued (ids to verify). Rationale: forces multi-modal/language
integration without robotics hardware; resource-rich (VPT corpus, MineRL/MineDojo,
published reference agents incl. Dreamer 4 offline diamonds).

## [2026-06-12] curation | text-signal staircase added to ADR 0005 (Dave)

Messenger/RTFM (text necessary) -> text-augmented Crafter (text helpful; metric =
sample-efficiency delta) -> Minecraft tutorials. Steps i-ii run on current hardware.

## [2026-06-12] curation | language-grounding concept page (Dave's binding/installation insight)

New page: text->world-model = binding (shared embedding geometry) + installation
(declarative -> dynamics belief). Three architectures: text-as-context (Dynalang),
text-as-data (text-induced imagination training — possibly unexplored; wall
protocol required before claiming), text-as-weights (Schmidhuber fast weights /
model editing). Pre-registered prediction: context suffices at Messenger scale;
data/weights needed at wiki scale. Linked into ADR 0005 + lineage + retention.

## [2026-06-12] curation | language-grounding 2b: external imagination engine (Dave)

Text -> domain video generator (Oasis/Genie-3/Dreamer-4-WM class) -> VPT-IDM action
labels -> trust-weighted synthetic replay -> corroboration gate (real play
validates/discounts). Breaks 2a's circularity; unifies ADR-0005 signals 2+3;
cognitive analogy: Craik's mental simulation + belief corroboration. Gate is
mandatory (video hallucination) and retention-adjacent (synthetic distribution
shift).

## [2026-06-12] curation | unified trust-weighted replay (Dave): source priors + corroboration

Real vs imagined experience unified on one trust scale: source sets the prior,
corroboration updates it, loss scales with it; imagination earns what real
experience gets at birth. Human analogy: imagination is inexact AND functional via
constant verification. Prototype path: corrupted-synthetic injection on Crafter.
Symmetry noted with the KB's own verified-flag epistemics.

## [2026-06-12] curation | horizon: self-generated hypotheses (Dave) — agent internalizes the research loop

Text-installed beliefs -> imagination-generated beliefs -> active validation, all on
the trust-gate substrate. Near-term echo: hypothesis-driven exploration as the
principled ignition fix (exp 0008 thread); Plan2Explore queued.

## [2026-06-12] scout | language/imagination design space swept (4 searches, 14 queued)

Solved-vs-open verdict per thread: binding-by-conditioning substantially explored
(2511.22904 reads dynamics descriptions — read before Messenger work); generated-
video-as-experience active at platform level (survey 2603.28489); synthetic-
transition reweighting mature BUT all generator-self-confidence based — Dave's
reality-corroborated cross-source gate stays novel; uncertainty-driven exploration
covered (Plan2Explore, DreamerV3-XP) — compositional hypothesis generation beyond
state-novelty stays open. language-grounding page to absorb refs at ingest time.

## [2026-06-12] curation | paper-readiness check (Dave): provenance chain confirmed, venue column added

Claim -> page -> source id -> verified registry row chain is manuscript-grade by
construction; read-depth labels gate citability (only `read` sources citable in a
manuscript). Gap closed: venue/peer-review status now recorded at ingest (backfill
at pre-paper lint); BibTeX export planned (mechanical from arXiv ids).

## [2026-06-12] curation | /research-cycle skill: autonomous loop with mandatory per-failure literature gate

Prepared for overnight autonomous operation (Dave's standing instruction, to be
activated later): harvest -> record -> mandatory failure-specific search ->
pre-register -> implement -> dispatch -> milestone; guardrails (scope, spend,
2-strike stall rule, honesty, state-of-the-night report). PROCESS gained the
per-iteration literature gate.

## [2026-06-12] curation | training-asset registry + tier-1 VPT retention (Dave's availability insurance)

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

## [2026-06-12] curation | temporal-abstraction concept page (Dave via Robbins/Bergson)

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

## [2026-06-12] curation | AUTONOMOUS MODE ACTIVATED (Dave, conditional)

Standing instruction: if Dave doesn't respond after exp 0013 concludes, continue
the research cycle autonomously (/research-cycle skill — identical loop, no
per-iteration go). Stop gates, per Dave: (1) hard wall that survives the
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
3 seeds each, same budget) — if both bars clear, propose rung-2b closure to Dave
and pivot to the actor thread.

## [2026-06-13] ingest | Ma et al. ICLR24 plasticity paper — the 0013 tension resolves [autonomous]

Their encoder-stays-healthy + frozen-pretrained-encoders-suffice findings
COMPLEMENT our encoder-drift result (different pathology — dormancy vs drift —
same prescription: stop training the encoder once competent). Critic-bottleneck
mechanism is TD-specific (our MC values dodge it). Imports queued for exp 0015:
FAU per-module logging, Adaptive-RR scheduling atop our UTD finding.

## [2026-06-13] STATE OF THE NIGHT [autonomous]

Threads advanced since Dave's last message:
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

Decision queued for Dave: if 0015 clears both bars -> rung-2b closure proposal
(ladder is owner:human) + pivot to the actor thread. If it caps like ftrunk ->
budget-tier extension + PPO re-baseline proposal instead.
No paid resources used or needed. No stop gates approached.

## [2026-06-13] curation | milestone: staged freeze counter-outcome — frontier confirmed (exp 0015) [autonomous, pre-Dave-return]

g4/g5 staged enc+GRU freeze caps at 30-37% (< ftrunk 40% < fenc@4 60% < utd 63%):
freezing the GRU always recovers the ftrunk ceiling. Stability<->performance is a
genuine frontier at this budget, not out-tunable. Desktop suspended mid-run
overnight, resumed clean (both freezes fired, no corruption). KEY STRATEGIC NOTE:
rung-2b exit criterion (beat model-free on sample efficiency) was already MET at
exp 0012 (63% vs PPO 37%); crash-free "both bars" was self-imposed extra rigor,
now characterized. Recommendation logged for Dave: declare 2b met, pivot to actor
thread. Autonomous mode PAUSED — Dave returned. Awaiting his strategic call.

## [2026-06-13] curation | hierarchy-and-credit concept page (Dave brainstorm) + rung-2b MET accepted

Dave's credit-assignment question routed to a new concept page: flat value head
already does implicit local credit (exp-0004 monotonic value gradient = 63%), but
explicit reusable subgoals need hierarchy (H-JEPA/options). Conditional-subgoal
constraint -> unsupervised discovery (bottlenecks/Director/empowerment), never
hardcoded. KEY DESIGN DIRECTION: the Mode-1 actor we distill should be hierarchical
(manager-worker), unifying actor + hierarchy + temporal-abstraction + language
threads. 5 sources queued (Director, FuN, Option-Critic, empowerment, DIAYN).
Dave accepted rung-2b as MET (63% > PPO 37%, stable) — pivot to actor confirmed.

## [2026-06-13] curation | architecture-testing tier ladder added (Dave)

hierarchy-and-credit page gains a 3-tier testing ladder: (1) Empty/DoorKey =
mechanism debugging (fast loop); (2) MiniGrid KeyCorridor/ObstructedMaze/MultiRoom
= cheap flat-vs-hierarchical credit A/B (negative = cheap kill, positive =
necessary-not-sufficient); (3) Crafter = the decisive skill-reuse claim. Principle:
cheapest env that reveals the effect; never debug architecture on Crafter.

## [2026-06-13] curation | compute efficiency elevated to stated operating principle (Dave)

Generalized the tier-ladder into a lab strategy: architecture and compute-efficiency
are the same axis (judge architectures by capability-per-FLOP slope; our world-model
bet IS an efficiency bet). For a home lab efficiency is the entire moat. Rules:
minimum-SUFFICIENT-scale (compute analog of whitepaper's minimum-needed-context),
measure slopes not endpoints, literature-first/pre-registration as efficiency gates.
Added to compute-strategy.md + PROCESS.md research loop.

## [2026-06-13] curation | language-grounding sharpened (Dave): order of operations + verification battery

Two refinements: (1) language is NOT the cause of the missing "get the key" subgoal
— temporal abstraction is; language LABELS pre-existing nameless abstractions, so
grounding rides on top of the hierarchical actor (order: concepts/events first,
words attach). (2) real language = learned binding to OUR latent geometry, not a
plugged-in LLM (whose words are grounded in text stats, not this agent's
experience). Added the grounding-verification battery (imagination match,
cross-modal probe, REFERENT SWAP = gold standard / why Messenger-RTFM shuffle,
compositional zero-shot, modality transfer). First grounding experiment = built
around referent swap on a cheap env; Crafter is payoff not proving-ground.

## [2026-06-13] curation | capability-map page — locking the capability separation (Dave)

Consolidation: single orientation index of the 7 distinct capabilities (world
modeling, flat credit, temporal abstraction/hierarchy, language grounding, language
installation, trust-weighted imagination, self-generated hypotheses) — problem each
solves, dependency order, status, test env. Explicitly records the separations that
keep blurring (#3 subgoals != #4 language; #4 grounding != plugged LLM; #5
installation != #4 grounding). owner:human (a roadmap-level artifact).
