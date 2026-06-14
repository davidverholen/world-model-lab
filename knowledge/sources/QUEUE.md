# Intake Queue

Unread material waiting for proper ingestion (`/ingest-source`). Items here must
NOT be cited by wiki pages. **All arXiv ids batch-verified 2026-06-12 via the arXiv
API (id→title match)**, except the Background/lineage section (pre-arXiv classics).
Corrections found in that pass: DINO-WM is 2411.04983 (2411.04958 is an astronomy
paper); "General agents CONTAIN world models" (not "need"); Sutton-lab plasticity
paper is "MAINTAINING Plasticity in Deep Continual Learning".

## Rung-4: reading-ignition literature gate (scouted 2026-06-14; all ids verified via arXiv API or abstract fetch)

This cluster is the lit gate for HO-0007 (dense shaping to ignite reading) and the KEY RISK question
(does shaping create a non-reading shortcut?). Read in the order listed; RTFM+Messenger are the
mandatory anchors before any ignition experiment.

### Anchor papers — reading-to-learn-dynamics (ingest these first)

- [ ] arxiv:1910.08210 **RTFM** (Zhong, Rocktäschel & Grefenstette, ICLR 2020) — the paradigm paper
      for reading-to-learn-DYNAMICS (not goal-following). Key mechanism: curriculum learning was
      REQUIRED to ignite reading; without it, dense reward on simple subtasks forced grounding before
      sparse reward on full tasks. Dense reward = enemies drop 1 point on contact (not a single sparse
      bonus); curriculum graduates from 1-hop to 5-hop coreference. INGEST to confirm exact reward
      function. HIGHEST PRIORITY — direct ancestor to our env design + shaping question.

- [ ] arxiv:2101.07393 **Messenger / EMMA** (Hanjie, Zhong & Narasimhan, ICML 2021) — referent-swap
      evaluation built in as the grounding gold standard (entities are symbolic tokens whose text
      labels randomize per game; zero-shot eval = new entity names). Three-stage curriculum was
      REQUIRED — end-to-end training on the full task "proved too difficult". Reward is sparse binary
      (win=+1, lose=-1 per interaction). INGEST for: (a) exact curriculum stage design — compare to
      our shaping anneal plan; (b) confirmation that sparse reward WITHOUT curriculum fails.

- [x] arxiv:2308.01399 **Dynalang** (Lin, Du, Watkins, Hafner, Abbeel, Klein & Dragan, 2023) —
      INGESTED 2026-06-14 → papers/dynalang-2023.md (method depth: RSSM conditioning, aux loss
      formula, static-manual mismatch analysis + candidate adaptations for our setting).

- [ ] arxiv:2511.22904 **LED-WM** (Nguyen & Lee, Nov 2025) — DreamerV3 + cross-attention grounding
      on DYNAMICS descriptions (text = how-env-behaves, not goal). Tests on Messenger. The most
      recent text-conditioned dynamics world model; likely the closest published baseline to our
      rung-4 architecture. INGEST for: attention mechanism details, what reward structure was used,
      whether a shaping or curriculum was applied, and comparison results on Messenger vs Dynalang.

### Key risk: does shaped reward induce a non-reading shortcut?

- [ ] arxiv:2305.16621 **Language Reward Shaping May Hinder Learning** (Huang, Lipovetzky & Cohn,
      2023) — theoretical + empirical evidence that suboptimal LRS designs that reward PARTIALLY
      MATCHED trajectories converge SLOWER than pure RL, because the shaped signal creates a
      misaligned proxy. Directly relevant: our gesture-progress shaping signal could be exploited
      WITHOUT reading if the agent discovers a partial-match sequence (e.g., a stereotyped gesture
      that scores shaping reward on most manuals by coincidence). INGEST for: exact mechanism and
      diagnostic tests they propose; map to our anneal+eval protocol.

### Dense bridge / auxiliary loss alternatives (context before designing ignition fix)

- [ ] arxiv:2210.00066 **Language Dynamics Distillation — LDD** (Zhong, Mu, Zettlemoyer,
      Grefenstette & Rocktäschel, 2022) — two-stage: pretrain a model to PREDICT DYNAMICS from
      language-annotated demos, THEN fine-tune with RL. Motivated explicitly by "sparse, delayed
      rewards make grounding difficult." Provides a language-aware init that shortcuts the ignition
      wall. INGEST for: whether demonstration data is required, how much it helps, and whether
      the pretrain generalises to held-out manuals (swap-follow analog).

- [ ] arxiv:1707.01495 **HER: Hindsight Experience Replay** (Andrychowicz et al., NeurIPS 2017) —
      canonical technique for sparse-reward goal-conditioned RL: relabel failed episodes as successes
      toward the achieved state. Relevant because a language-HER variant (relabel episode with the
      gesture it actually performed) could give dense positives even when the manual is wrong.
      Read as BACKGROUND — one page; no full ingest needed unless we plan a HER variant.

### Theory: when does shaping preserve the optimal policy?

- [ ] icml:ng1999shaping **Ng, Harada & Russell 1999 "Policy Invariance Under Reward Transformations"**
      (ICML 1999) — proves potential-based shaping F = γΦ(s') - Φ(s) is necessary AND sufficient
      to guarantee the shaped and original MDPs share the same optimal policy. Our gesture-progress
      shaping is NOT obviously potential-based (it depends on the manual, which changes each episode).
      Read to determine: (a) whether per-episode non-stationarity violates the guarantee; (b) whether
      anneal-to-0 and eval-at-0 sidesteps the bias even if the guarantee fails during training.
      One-page read; background level.

### Unified benchmark for grounded language envs (context)

- [ ] arxiv:2110.10661 **SILG** (Zhong et al., NeurIPS 2021) — unified interface over RTFM,
      Messenger, NetHack, ALFWorld under one API. Useful as a map of the space (what tasks exist,
      which are dynamics-reading vs goal-following) but not a method paper. Low-priority read.

---

## High priority (directly on our experiment path)

### Retention / plasticity / interference (exp 0009–0011 thread; found 2026-06-12, ids from search results)

- [x] arxiv:2205.07802 Nikishin primacy bias — INGESTED 2026-06-12 →
      papers/nikishin-primacy-2022.md (protocol depth)
- [ ] arxiv:2411.04832 Plasticity Loss in Deep RL: A Survey (2025) — map of the field
- [ ] arxiv:2303.07507 Loss of Plasticity in Continual Deep RL (Abbas et al.)
- [ ] arxiv:2506.00592 Mitigating Plasticity Loss by Reducing Churn (ICML 2025) —
      NTK-rank view; churn reduction
- [ ] arxiv:2306.13812 Loss of Plasticity in Deep Continual Learning (Sutton lab;
      continual backprop / Nature 2024 version)
- [ ] arxiv:2307.04887 Measuring and Mitigating Interference in RL — diagnosis metrics


- [x] arxiv:2301.08243 I-JEPA — INGESTED 2026-06-12 → papers/ijepa-2023.md (method depth)
- [x] arxiv:2301.04104 DreamerV3 — INGESTED 2026-06-12 → papers/dreamerv3-2023.md
      (implementation-grade; symlog/two-hot/free-bits + critic-EMA-target noted for retention)
- [ ] arxiv:1803.10122 World Models (Ha & Schmidhuber) + https://worldmodels.github.io/
- [ ] arxiv:2509.24527 Dreamer 4 — read fully (id verified)
- [ ] arxiv:2511.08544 LeJEPA + code https://github.com/rbalestr-lab/lejepa (id verified)
- [ ] arxiv:2605.26379 When Does LeJEPA Learn a World Model? (id verified; conditions under which JEPA recovers true environment structure)
- [~] openreview:lecun-path — INGESTED at skim depth 2026-06-12 →
      papers/lecun-2022-path.md; full-PDF read still queued for promotion to current

## Rung-3 prep (Crafter/Craftax SOTA line; found + url-verified 2026-06-12)

- [ ] arxiv:2502.01591 Improving Transformer World Models for Data-Efficient RL —
      the early-2025 Craftax-classic SOTA jump (53.2%→67.4% reward); read before
      designing our rung-3 agent
- [ ] arxiv:2605.16457 Identifiable Token Correspondence for World Models (ITC,
      May 2026) — current Craftax SOTA (72.5% / 35.6%), beats token-based WMs on
      Atari100k; "identifiability" framing connects to the LeJEPA theorem
      (2605.26379) — the two threads are converging
- [ ] arxiv:2406.19320 Δ-IRIS context-aware tokenization · arxiv:2406.01361 DART
      (Atari in a world of tokens) — token-WM efficiency line, lower priority

## Imagination-module substrate (forward / modular-reuse thread; raised in conversation 2026-06-13)

OPEN QUESTION, NOT A DECISION. When a future modular design asks "what powers the
imagination module," read these two as the bracketing pair before forming any view —
they span the design space. Flagged explicitly because the question arose from an
off-the-cuff idea (could a Cosmos-scale video model BE the Dreamer imagination loop?);
do not let that framing prejudge the read. Both items already in the KB:
- diffusion-in-(latent/pixel): arxiv:2405.12399 DIAMOND (queued below under Medium) —
  small diffusion WM an RL agent learns inside; the existence proof that diffusion WMs
  work at Atari scale (contrast: Cosmos-scale video diffusion does NOT fit a tight
  per-grad-step imagination loop — latency/pixel/non-differentiable).
- frozen-encoder + small action-conditioned predictor: V-JEPA 2-AC, inside
  arxiv:2506.09985 (V-JEPA 2; already a verified source) — reuse a big pretrained
  encoder, learn a small dynamics model on top. The "reuse the representation, not the
  simulator" path; pairs with DINO-WM (2411.04983) and the DINOv3 frozen-encoder
  retention hypothesis (above).
Read both neutrally; the decision (if any) belongs to a later ADR, not this note.

## Medium (context and alternatives)

- [ ] arxiv:2508.10104 DINOv3 (Meta, verified 2026-06-12) — 7B SSL vision backbone,
      Gram anchoring for stable dense features + distilled small variants. Rung-3
      hypothesis: frozen DINOv3-small encoder + learned belief/dynamics (à la
      DINO-WM) — a frozen encoder is immune to primacy bias by construction
      (connects to exp 0009–0011 retention thread).

- [ ] arxiv:1811.04551 PlaNet (RSSM origin) / arxiv:2010.02193 DreamerV2
- [ ] arxiv:2209.00588 IRIS (token-based WM) · arxiv:2405.12399 DIAMOND (diffusion WM)
- [ ] arxiv:1911.08265 MuZero · arxiv:2310.16828 TD-MPC2 (value-equivalent line)
- [ ] arxiv:2506.01622 "General agents need world models" (theory; id moderately confident)
- [ ] arxiv:2402.15391 Genie + Genie 2/3 blogs (deepmind.google)
- [ ] arxiv:2109.06780 Crafter · Craftax (github.com/MichaelTMatthews/Craftax)
- [ ] arxiv:2411.14499 world-model survey · arxiv:2405.03520 Sora-as-world-simulator survey
- [ ] arxiv:2605.25874 WBench (confirmed 2026-06-12 as the late-May AMI-circle
      companion to 2605.26379): multi-turn interactive video world-model benchmark,
      5 dimensions, 289 cases; headline: current models collapse under minor visual
      shifts. Read with the theorem paper as a pair.
- [ ] arxiv:2602.11389 Causal-JEPA (Feb 2026) — world models via object-level
      latent interventions; the causality-meets-JEPA line, relevant to whether our
      latents capture manipulable structure (keys, doors)
- [ ] arxiv:2512.10942 VL-JEPA (Dec 2025) — vision-language JEPA; lower priority
      (language enters our roadmap only with mission-conditioned MiniGrid tasks)

## Teams to watch (world-model lines worth tracking; added 2026-06-13)

Not papers — research groups Eric Xing flagged in the DataCamp podcast "Will World
Models Bring us AGI?" (youtube VNyLNZunv9E) as holding a view of world models close
to his own (simulator of actionable possibilities, not video generation). Track for
releases; ingest specific outputs as they land.

- [ ] **Demis Hassabis / Google DeepMind** — Xing reports "almost perfect alignment"
      with Hassabis (conv. ~Mar 2026) on what a world model and a virtual cell are,
      and a "grounded but sophisticated" view of how to build AND test them. Xing
      expects "something fancier and disruptive in the next few months"; says
      DeepMind's public releases trail their internal work by months-to-years.
      Watch: Genie line (2402.15391 + Genie 2/3 blogs, already queued under Medium),
      virtual-cell work. The Hassabis-alignment quote also worth a line on
      concepts/ or design/ once a concrete artifact exists.
- [ ] **Fan-Yun Sun / Moonlake AI** — ex-Stanford SAIL (Holodeck, LayoutVLM); now
      co-founder/CEO of Moonlake, a stealth-ish world-model startup (Chris Manning,
      Ian Goodfellow in the orbit). Thesis: causal world models should be
      multimodal, interactive, and EFFICIENT — structure + causality over blind
      scaling; flags physical/spatial glitches (floating solids, interpenetration)
      as the failure mode. Aligns with our generative-vs-predictive +
      Causal-JEPA (2602.11389) threads. Source to ingest first:
      Latent Space "Moonlake" episode (latent.space/p/moonlake). Watch for first
      model release.

## Value calibration / model exploitation (exp 0019–0021 thread; scouted 2026-06-13, ids from search results — verify at ingest)

The published toolkit for the inflated-imagined-value failure mode (our 0019/0020
finding: RSSM fixed policy quality, not value calibration). Cheapest-first; b1 already
implemented (exp 0021) reusing DreamerV3's critic-on-replay rather than reinventing.

- [ ] arxiv:1906.08253 MBPO "When to Trust Your Model" (Janner 2019) — short rollouts
      BRANCHED from real replay states; decouples model horizon from task horizon,
      limits compounding error. Our fork-b2 (shorten horizon 15→~5 + branch). Has a
      monotonic-improvement bound. HIGH priority — the canonical short-rollout result.
- [ ] arxiv:2005.13239 MOPO: Model-based Offline Policy Optimization — penalize reward
      by model uncertainty (lower-bound the true return). Fork-b4 pessimism lever.
- [ ] MOReL (Kidambi et al. 2020; id to verify) — pessimistic MDP, companion to MOPO.
- [ ] CBOP: Conservative Bayesian Model-Based Value Expansion (ICLR 2023; ssanner.github.io
      /papers/iclr23_cbop.pdf; arxiv id to verify) — adaptively weights model rollouts
      by posterior uncertainty; the calibrated version of value expansion.
- [ ] (already queued) arxiv:2412.14312 "Stealing That Free Lunch" — Dyna-style limits;
      read alongside this thread (the skeptic's case for why imagination value misleads).
- NOTE DreamerV3 (papers/dreamerv3-2023.md, INGESTED) is the primary source for b1
      (critic-on-replay β_repval 0.3) + b3 (percentile return-norm, two-hot critic) —
      page corrected 2026-06-13 to capture the replay-critic detail it had missed.

## Background / lineage (no rush; see concepts/intellectual-lineage.md; ids unverified)

- [ ] Sutton 1991, "Dyna, an Integrated Architecture for Learning, Planning, and
      Reacting" (SIGART) — our flywheel's true name; read before exp ~0012
- [ ] LeCun et al. 2006, "A Tutorial on Energy-Based Learning" — the formal frame
      under JEPA
- [ ] Hadsell, Chopra & LeCun 2006, "Dimensionality Reduction by Learning an
      Invariant Mapping" (CVPR) — contrastive/joint-embedding origin
- [ ] Kaelbling, Littman & Cassandra 1998, "Planning and Acting in Partially
      Observable Stochastic Domains" — belief states, formally
- [ ] Rao & Ballard 1999, "Predictive Coding in the Visual Cortex" (Nat. Neurosci.)
- [ ] Craik 1943, *The Nature of Explanation* (book; excerpt suffices) — the origin
      sentence of the whole field

## Low (breadth)

- [ ] arxiv:2412.03572 Navigation World Models · arxiv:2411.04983 DINO-WM (id
      corrected 2026-06-12; ICML 2025; frozen DINOv2 features + planning — pairs
      with the DINOv3 frozen-encoder retention hypothesis)
- [ ] arxiv:2403.00504 Image World Models · arxiv:2502.11831 intuitive physics from video
- [ ] arxiv:2309.17080 GAIA-1 · arxiv:2503.20523 GAIA-2 (Wayve, driving)
- [ ] arxiv:2501.03575 NVIDIA Cosmos · arxiv:2206.14176 DayDreamer (robots)
- [ ] arxiv:2509.14252 LLM-JEPA (id unverified) · arxiv:2307.12698 MC-JEPA · A-JEPA (audio, id unverified)
- [ ] awesome-list: https://github.com/LMD0311/Awesome-World-Model
- [ ] World Labs "Marble" (worldlabs.ai) · Oasis (oasis.decart.ai) — neural game engines

### Found via Nikishin citation walk (2026-06-12, ids url-verified)

- [x] arxiv:2310.15017 Mind the Model, Not the Agent — INGESTED 2026-06-12 →
      papers/qiao-model-primacy-2023.md (redesigned exp 0012)
- [ ] arxiv:2412.14312 Stealing That Free Lunch: Exposing the Limits of
      Dyna-Style RL — direct Dyna-limits analysis; pairs with the lineage thread
- [ ] arxiv:2501.16918 On Rollouts in Model-Based RL
- [ ] arxiv:2502.00802 Fisher-Guided Selective Forgetting (primacy-bias mitigation)
- [ ] Revisiting Plasticity in Visual RL: Data, Modules and Training Stages
      (2023; id to verify) — module-level plasticity localization, directly our Q
- [ ] EvoAgent: continual world model for long-horizon tasks (2025; id to verify)

## Minecraft milestone (ADR 0005; ids UNVERIFIED — batch-verify before ingest)

- [ ] VPT: Learning to Act by Watching Unlabeled Online Videos (Baker et al. 2022,
      OpenAI) — the IDM pseudo-labeling recipe; the action-labeled-video signal
- [ ] MineDojo (Fan et al. 2022) — task suite + YouTube/wiki/Reddit corpus; the
      text-signal data source
- [x] Dynalang: Learning to Model the World with Language (Lin et al.) — INGESTED
      2026-06-14 → papers/dynalang-2023.md (see Rung-4 section above)
- [ ] Voyager (Wang et al. 2023) — LLM-as-planner over Minecraft skills; the
      pragmatic fallback architecture
- [ ] STEVE-1 — instruction-following Minecraft agent (text->behavior bridging)
- [x] RTFM (arxiv:1910.08210) + Messenger/EMMA (arxiv:2101.07393) — VERIFIED + QUEUED
      in Rung-4 reading-ignition section above; ingest from there
- [ ] Plan2Explore (Sekar et al. 2020) — world-model-uncertainty-driven exploration;
      ancestor of the self-generated-hypotheses horizon + candidate ignition fix
      (id to verify)

## Language/imagination design space (scouted 2026-06-12 after the grounding discussion; ids url-verified)

### Binding/installation (text -> world model)
- [x] arxiv:2511.22904 LED-WM: Language-conditioned WM improves policy generalization by
      reading ENVIRONMENTAL DESCRIPTIONS (Nov 2025) — VERIFIED + QUEUED in Rung-4
      reading-ignition section above
- [x] arxiv:2308.01399 Dynalang — INGESTED 2026-06-14 → papers/dynalang-2023.md
- [ ] arxiv:2407.13466 LIMT · arxiv:2509.21797 MoWM · arxiv:2604.02097 LatentUM —
      conditioning variants

### Grounded-language GAME ENVIRONMENTS (the "does our testbed already exist?" check, 2026-06-14)
KEY DISTINCTION: instruction-following (text = a GOAL/command) vs reading-to-learn-DYNAMICS
(text = the rules; our "installation" interest). Read this cluster BEFORE building any custom
grounding env — they give task designs + baselines, and decide reuse-vs-build. ids from search,
verify at ingest.
- [ ] arxiv:2505.11962 **CrafText** (AIRI, ACL 2025; github AIRI-Institute/CrafText) — goal-
      conditioned **Craftax** instruction-following (3924 instructions). "Crafter+language"
      ALREADY EXISTS — but instruction-following, not read-to-learn-dynamics, AND it's JAX/Craftax
      (ADR-0006 interop wall). Read to confirm it is NOT our installation testbed.
- [x] **RTFM** (arxiv:1910.08210, ICLR 2020; VERIFIED) — QUEUED in Rung-4 reading-ignition section.
- [x] **Messenger / EMMA** (arxiv:2101.07393, ICML 2021; VERIFIED) — QUEUED in Rung-4 reading-ignition section.
- [ ] arxiv:2110.10661 **SILG** (NeurIPS 2021) — unified benchmark wrapping RTFM/Messenger/NetHack/
      ALFWorld under one interface; the map of the grounded-language-game space.
- [ ] arxiv:2210.00066 **Language Dynamics Distillation** — pretrain to predict dynamics from
      language-annotated demos, then RL; a METHOD directly on the read-to-learn-dynamics target.
- LIKELY GAP (UNVERIFIED — needs the deep wall protocol, not just 2 searches): a RICH (Crafter-
      class) reading-to-learn-DYNAMICS env. CrafText covers rich+instructions; RTFM covers
      dynamics+toy. The intersection (rich + dynamics) is the candidate gap → "Crafter-with-manuals"
      DEVELOPMENT testbed (referent-swap + tutorial-reveals-hidden-recipe), real Minecraft for DEPLOY.
- SPEC (maintainer, 2026-06-14): = "CrafText's rich world + RTFM's MANDATORY reading + PyTorch",
      built as a manual-layer ON our existing CrafterEnv (not a new engine). The HARD part is the env
      design, not the plumbing: make reading mandatory by per-episode randomizing the crafting graph /
      referent map (RL can't discover, text can reveal) WITHOUT making it unsolvable. Verification =
      ablate {with-manual, no-manual, SWAPPED-manual}: grounded iff with >> without AND behaviour
      follows the swapped manual (the referent-swap gold standard). Gate before building: deep-verify
      the gap + ingest CrafText/RTFM/SILG/LDD for task designs + baselines. Downstream of the Crafter
      foundation; not now.
### Generated video as experience (architecture 2b)
- [ ] arxiv:2603.28489 Video Generation Models as World Models survey (2026) —
      map of the area; + awesome-list github.com/ziqihuangg/Awesome-From-Video-Generation-to-World-Model
- [ ] arxiv:2509.23958 RL with Inverse Rewards for WM post-training

### LLM-as-latent-backbone (PAN / GLP — Eric Xing; queued 2026-06-13 after the PAN discussion)
- [ ] arxiv:2511.09057 PAN: A World Model for General, Interactable, Long-Horizon
      Simulation — Generative Latent Prediction (GLP): vision encoder → latents,
      LLM backbone (Qwen2.5-VL-7B) as the latent DYNAMICS model conditioned on
      language actions, multi-granular diffusion decoder → video. The published
      answer to "how do you wire an LLM into a latent world model" — further than
      our sketch. INGEST QUESTION to settle: does PAN show (a) only static
      pretrained knowledge baked into the backbone, or (b) TEST-TIME acquisition —
      read a novel tutorial, form new latents, act differently? (b) is our edge
      case (text->new-representations); if PAN demonstrates it, roadmap changes.
      Also: text-through-the-visual-channel (read signs/menus as pixels) as input
      unification — relevant to our language-grounding staircase. Pairs with
      concepts/generative-vs-predictive.md (PAN sits BETWEEN pure-JEPA non-generative
      latent and pixel-generation; names the midpoint "GLP") and design/architecture-strategy.md.
- [ ] arxiv:2507.05169 Critiques of World Models (Xing et al., Jul 2025) — the
      theoretical position paper behind PAN; argues FOR a generative decoder
      (validate predictions against real observations) vs JEPA's non-generative
      stance. Read as the pair to PAN; the explicit counter-argument to LeCun on
      our generative-vs-predictive axis.
### Trust gate (synthetic-experience weighting)
- [ ] arxiv:2104.04174 Learning to Reweight Imaginary Transitions — the ancestor
- [ ] arxiv:2506.09270 Uncertainty Prioritized Experience Replay (RLC 2025)
- [ ] arxiv:2410.18082 Prioritized Generative Replay · arxiv:2602.14351 WIMLE
      — NOTE the gap that keeps 2b novel: all weight by the GENERATOR's
      self-confidence; reality-corroborated cross-source priors (maintainer's gate)
      appear uncovered
### Self-generated hypotheses / directed exploration
- [ ] arxiv:2005.05960 Plan2Explore (id confirmed) — ensemble-disagreement
      information gain; candidate ignition fix at rung-2 scale
- [ ] arxiv:2510.21418 DreamerV3-XP — uncertainty-driven exploration + prioritized
      replay in current Dreamer; nearest modern baseline

## Temporal abstraction (events-not-ticks thread; ids unverified)

- [ ] Sutton, Precup & Singh 1999, "Between MDPs and Semi-MDPs: Options" — the
      extended-action formalism
- [ ] Hafner et al. 2022, Director: Deep Hierarchical Planning from Pixels —
      hierarchy inside a Dreamer world model
- [ ] Zacks & Tversky — event segmentation theory (cog-sci; chunking at
      prediction-error boundaries — implementable in our stack)
- [ ] Zhao et al. 2023, ACT action chunking (robotics) — extended actions as units
- [ ] (context) Stephen E. Robbins' Bergson-based critique of computational world
      models — the philosophical counterpoint; treat as framing, not spec
- [ ] arxiv:2310.07418 Revisiting Plasticity in Visual RL (ICLR 2024) — localizes
      plasticity loss in the CRITIC (model-free + augmentation); TENSION with our
      encoder-localization (exp 0013) — ingest to reconcile; their Adaptive RR +
      data-augmentation-as-plasticity-preserver are untested levers for us
- [ ] arxiv:2504.17490 Plasticine benchmark · arxiv:2410.07994 Neuroplastic
      Expansion — plasticity tooling/methods (lower priority)

## Crafter DEPTH / achievement-hierarchy fork (exp 0027 thread; found 2026-06-14 via lit-gate, ids from search — verify at ingest)

The published toolkit for exactly our remaining ceiling: the agent reaches ~3–6
shallow/mid achievements but does NOT climb the wood→table→pickaxe→stone→iron
tech-tree (0025 result; 0026 showed replay-ratio buys breadth not depth). These are
the candidate anchors for 0028, branched on the 0027 outcome (data-bound → scale;
exploration/hierarchy-bound → these). INGEST the matching one(s) once 0027 picks the fork.

- [x] arxiv:2307.03486 **Discovering Hierarchical Achievements via Contrastive Learning**
      (Achievement Distillation, NeurIPS 2023) — INGESTED 2026-06-14 →
      papers/achievement-distillation-2023.md (method depth; segmentation rule,
      intra/cross InfoNCE losses, Crafter per-achievement results, RSSM portability).
- [ ] arxiv:2305.00508 **Learning Achievement Structure for Structured Exploration in
      Domains with Sparse Reward** — learns the achievement dependency graph to drive
      structured exploration; the explicit "explore TOWARD the deep sequence" method.
- [x] arxiv:2306.15934 **Curious Replay for Model-based Adaptation** — INGESTED
      2026-06-14 → papers/curious-replay-2023.md (method depth; priority formula,
      update rule, Crafter results, frozen-encoder applicability).
      KEY CONTRAST with 0026: uniform 2× replay failed (breadth not depth), but
      *curiosity-prioritized* replay reweights toward novelty — a different lever on
      the same axis. Cheapest of the three to graft onto our flywheel; read FIRST if
      0027 is null (exploration-bound).
- NOTE DreamerV3-XP (2510.21418, queued below under directed exploration) is the
      uncertainty-driven-exploration sibling; read as the pair to Curious Replay.

LEADERBOARD SYNTHESIS (2026-06-14, verified via search; resolves the 0028 tension). Crafter
Score = geometric mean of 22 achievement rates. Human 50.5%. **Achievement Distillation
(2307.03486) 21.8% / reward 12.6 — best FROM-SCRATCH**; Curious Replay 19.4; PPO-ResNet 15.6;
DreamerV3 14.5 / reward 11.7; LSTM-SPCNN 12.1. **The deep tree is near-universally UNSOLVED:**
even AD (the champion) collects IRON only ~3%, DreamerV3 ~0.15% (AD = "20× DreamerV3"); diamond
≈0% for all from-scratch. So the ~14–22% scores are dominated by SHALLOW/MID breadth (the
geom-mean rewards any nonzero rate). Implication: our depth plateau is MOSTLY the universal
Crafter ceiling at the iron/diamond tier; the real headroom is the STONE tier (collect_stone /
stone_pickaxe / furnace), where SOTA reaches moderate rates and we get ~0. The method credited
with the depth gain is **Achievement Distillation — an ACTOR-SIDE contrastive self-imitation on
the achievement hierarchy, NOT a WM trick** — which vindicates our 0029/0028 actor-side
diagnosis. New sources to queue:
- [ ] arxiv:2507.04075 Accurate & Efficient World Modeling with Masked Latent Transformers
      (MLT, 2025) — recent Crafter WM entrant; check its per-achievement depth.
- [ ] arxiv:2406.07381 World Models with Hints of LLMs for Goal Achieving — LLM-prior
      exploration on Crafter; the "cheap external prior" lever for the deep tree.
- (already queued) 2307.03486 Achievement Distillation — PROMOTE to ingest if the next fork is
  the actor-side capstone; 2305.00508 structured-exploration is its sibling.

## Hierarchy / subgoal emergence (maintainer's credit-assignment brainstorm 2026-06-13; ids to verify)
- [ ] Hafner et al. 2022, Director: Deep Hierarchical Planning from Pixels (already
      noted under temporal-abstraction) — manager-worker subgoals INSIDE a Dreamer
      world model; the most direct fit for a hierarchical actor
- [ ] Vezhnevets et al. 2017, FeUdal Networks (FuN) — the feudal manager-worker
      origin for deep RL
- [ ] Bacon et al. 2017, The Option-Critic Architecture — end-to-end option discovery
- [ ] empowerment / intrinsic motivation: Klyubin 2005 (empowerment), Mohamed &
      Rezende 2015 (variational empowerment) — key-possession as objectively valuable
- [ ] Eysenbach et al. 2019, DIAYN (diversity is all you need) — unsupervised skill
      discovery, skills as latent-conditioned policies
- [ ] Ross & Bagnell DAgger (AISTATS 2011) — BC compounding error O(eT^2) + on-policy
      fix; background for why exp 0016 failed (cite, likely no full ingest needed)
- [ ] arxiv:2007.09560 Hoel, "The Overfitted Brain: Dreams evolved to assist
      generalization" (Patterns 2021; verified 2026-06-13) — dreams as
      anti-overfitting data augmentation; ties dreaming to our primacy-bias/retention
      thread; backs concepts/dreaming.md
- [ ] Levy & Goldberg 2014, "Neural Word Embedding as Implicit Matrix Factorization"
      (NeurIPS) — word2vec ≈ factorizing a shifted-PMI co-occurrence matrix; backs
      "association matrix = embedding" (language-grounding §associations); id to verify
