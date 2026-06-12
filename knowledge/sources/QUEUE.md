# Intake Queue

Unread material waiting for proper ingestion (`/ingest-source`). Items here must
NOT be cited by wiki pages. **All arXiv ids batch-verified 2026-06-12 via the arXiv
API (id→title match)**, except the Background/lineage section (pre-arXiv classics).
Corrections found in that pass: DINO-WM is 2411.04983 (2411.04958 is an astronomy
paper); "General agents CONTAIN world models" (not "need"); Sutton-lab plasticity
paper is "MAINTAINING Plasticity in Deep Continual Learning".

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
