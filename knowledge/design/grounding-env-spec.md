---
status: draft
owner: world-model
scope: local
sources: []
verified: false
last_reviewed: 2026-06-14
---

# Spec: **crafter-rtfm** — a read-to-learn-dynamics grounding benchmark

Origin: world-model project (rung-3 / language-grounding thread). This document is the
seed spec for a **separate, standalone, pip-installable benchmark repository** named
**`crafter-rtfm`** (Crafter + the RTFM "read the manual to learn the dynamics" paradigm).
The world-model project is one *consumer*; the env must be framework-agnostic.

## 0. GATE — do this before writing any env code

1. **Verify the gap (deep literature protocol, not 2 searches).** Read CrafText
   (arxiv:2505.11962), RTFM, Messenger/EMMA, SILG (arxiv:2110.10661), Language Dynamics
   Distillation (arxiv:2210.00066). Confirm no existing env already provides *rich-world +
   mandatory-read-to-learn-DYNAMICS + framework-agnostic*. If one does → reuse/port it, stop.
2. **Steal designs + baselines** from the above (task taxonomy, referent-swap protocol,
   EMMA text-attention baseline, LDD method). Do not reinvent what they solved.

## 1. Purpose

A benchmark where an agent **must read a per-episode text manual to learn that episode's
DYNAMICS** (recipes / referents / hidden rules), in a **rich, pixel-based, Crafter-class**
world, with a **built-in harness that proves whether grounding is real** (not spurious).
Distinct from instruction-following (text = a goal, e.g. CrafText): here text = *the rules*.

## 2. Core design principles

- **Mandatory reading**: the optimal policy is *unreachable* by exploration alone — the
  manual carries information RL cannot otherwise obtain — but *easily actable* once read.
- **Verifiable grounding (first-class, not an afterthought)**: the env ships the ablation
  harness that distinguishes real grounding from "text-presence correlates with success."
- **Rich but cheap**: reuse Crafter's engine/rendering/dynamics (PyTorch-friendly, no JAX).
- **Framework-agnostic**: gymnasium interface; no torch/jax in the env core.

## 3. Functional requirements

### 3.1 Base environment
- Built on the original **Crafter** (Hafner 2021; Python/numpy). Reuse its world gen,
  rendering, 17 actions, achievement/health systems. Pixel obs (64×64) by default.

### 3.2 Per-episode rule randomization (the mandatory-reading mechanisms — independently toggleable)
- **R1 Randomized recipes**: scramble the crafting graph each episode (which inputs yield
  which tool/placeable; e.g. "table needs 2 stone, not wood"). The manual states the recipes.
  RL cannot assume defaults. *Easiest to make genuinely mandatory; build first.*
- **R2 Referent swap**: randomize sprite↔material identity each episode (the thing that
  *looks* like wood is iron). The manual provides the legend. Enables the gold-standard test.
- **R3 Hidden / counterintuitive dynamics**: non-obvious rules stated *only* in text
  ("lava is safe with an iron pickaxe", "the red plant heals, the green poisons"). Richest,
  hardest to design learnably.
- Mechanisms compose; each has an on/off + difficulty parameter.

### 3.3 Manual generation
- Each episode emits a manual that **faithfully** describes that episode's randomized rules
  (so a *swapped* manual is factually wrong for this episode — required for the harness).
- Start **templated** (structured facts → text) with **paraphrase variety** (multiple
  surface forms per fact → tests wording robustness / natural-language generalization).
- Keep the underlying **structured ground-truth** alongside the text (for the harness and
  for symbolic baselines). Optional later: LLM paraphrasing for naturalness.

### 3.4 Observation / action / interface
- **Gymnasium** env. Observation = `Dict{ "image": Box(64,64,3 uint8), "manual": <text> }`
  where `<text>` is raw string + optional pre-tokenized ids (configurable). Action = Discrete
  (Crafter's 17; extend only if a mechanism needs new actions). Deterministic `seed` →
  reproducible (world, rules, manual). No deep-learning deps in the env core.

### 3.5 Verification harness (THE differentiator — must be a first-class API)
- Built-in eval modes selectable per episode/run:
  - `correct` — the true manual for this episode.
  - `none` — no manual (empty text).
  - `swapped` — a *correct-format* manual from a *different* episode (wrong facts).
  - `shuffled` — this episode's manual with its facts permuted (wrong mapping).
- Reported metrics:
  - **reading-necessity** = score(correct) − score(none): is reading needed at all?
  - **grounding** = score(correct) − score(swapped): does behaviour *follow the manual's
    content*, not just benefit from text being present? (the killer test)
- An agent is "grounded" iff reading-necessity > 0 **and** grounding > 0 (ideally
  score(swapped) ≤ score(none): it's actively *misled* by a wrong manual → proof it reads).

## 4. Configuration knobs

Active mechanisms (R1/R2/R3); difficulty per mechanism; manual paraphrase diversity;
vocabulary size + compositional depth; curriculum (how much reading helps, for shaping);
obs mode (pixel default; optional symbolic for cheap ablations).

## 5. Baselines to ship (so new methods have comparisons)

- **No-text RL** (lower bound — *must fail* if mandatory-reading is correctly designed).
- **Text-conditioned RL** (EMMA-style entity↔text attention; the standard grounding baseline).
- **LDD** (language-dynamics distillation) as a stronger reference.
- An **oracle** (env exposes the structured rules directly) — upper bound + sanity that the
  task is solvable *with* the info.

## 6. Non-goals

- Not a general instruction-follower (that is CrafText). Not a Minecraft replacement (that's
  deployment). Not photorealism. Not a new game engine — a *manual layer* over Crafter.

## 7. Build phases (each ends in a validation gate)

- **P0** Wrap Crafter as a gym `Dict` env with a dummy manual; runs end-to-end.
- **P1** R1 randomized recipes + templated manual + the harness (`correct/none/swapped`).
  **Critical gate:** no-text RL *fails*, oracle *solves*, and a manual-conditioned agent
  beats no-text → the mandatory-reading property is real. *If no-text still solves it, the
  env design failed — fix before continuing.*
- **P2** R2 referent-swap + the `grounding` (swapped) metric working as a clean signal.
- **P3** R3 hidden dynamics + paraphrase variety + curriculum + vocabulary scaling.
- **P4** Ship baselines (EMMA, LDD) + a small leaderboard-style eval suite + docs.

## 8. Success criteria for the ENV itself (how we know it's good)

- Mandatory-reading holds: no-text/​swapped ≈ chance; correct ≫ both; oracle ≈ correct.
- Cheap: an episode + manual generation adds negligible overhead vs vanilla Crafter.
- Reusable: `pip install`, gym API, zero DL-framework deps in core, reproducible seeds.
- Comparable: ships ≥2 published-style baselines and an oracle.

## 9. Packaging

Standalone repo `crafter-rtfm`; `pip`-installable; MIT/Apache; gymnasium-registered ids
(`CrafterRTFM-R1-v0`, …); CI with the P1 validation gate as a test; README with the
harness usage and the grounding metric front-and-centre. Depersonalized / public-ready.

## Links (this project)

[[language-grounding]] (binding/installation + the falsifiability battery) ·
[[crafter]] · [[environment-ladder]] · QUEUE → "Grounded-language GAME ENVIRONMENTS".
