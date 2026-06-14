---
status: draft
owner: agent
scope: local
sources: []
verified: false   # design-space page; literature ids in QUEUE, verify at ingest
last_reviewed: 2026-06-12
---

# Language → World Model: the Binding-and-Installation Problem

## What it is

The text-signal goal (ADR [[0005-minecraft-milestone]]) requires two capabilities
our agent lacks entirely (formulated by the maintainer, 2026-06-12):

1. **Binding**: map text entities ("furnace") onto visual/latent concepts learned
   from play — needs a shared or aligned embedding geometry (VL-JEPA/CLIP
   question).
2. **Installation**: convert a *declarative* statement ("smelting ore yields
   ingots") into a *dynamics belief* — a new edge in the transition model,
   composed from existing latent nodes, WITHOUT having experienced the transition.
   Today all knowledge enters via gradients on experienced transitions; text
   demands knowledge that arrives before experience.

## Language ≠ the subgoal gap; language ≠ a plugged-in LLM (maintainer, 2026-06-13)

Two refinements that set the order of operations:

- **Language is not why the agent lacks "get the key" as a subgoal** — that is
  [[hierarchy-and-credit]] (temporal abstraction). A rat/pre-verbal infant forms
  "get the key" as a nameless behavioral unit; options/Director form subgoals as
  nameless latent vectors. Language is the *separate* layer that LABELS those
  abstractions so they can be received, composed, transferred. ⇒ **Order**:
  grounded concepts + nameless events/subgoals form FIRST (in sensorimotor
  experience); words ATTACH to that pre-existing structure. Language grounding
  rides on top of the hierarchical actor — its subgoals/events are the attachment
  points.
- **Real language learning = learned binding to OUR latent geometry, not an LLM
  bolted on.** An LLM's "key" is grounded in text statistics (neighbors lock/door
  because of co-occurrence) → words connected to other words. Grounding we need =
  "key" connected to THIS agent's latent of seeing/grabbing a key, learned in its
  own belief space. Cannot be imported; specific to our model.

## Verifying grounding is REAL (not a shortcut) — the falsifiability battery

The hard part (maintainer): how to check the word is connected to the internal model, not
a spurious correlation. Tests, weakest → strongest:

1. **Imagination match**: text "get the key" → latent rollout ends in a
   key-possession belief that matches real experience (the "can imagine it" test).
2. **Cross-modal retrieval/probe**: latent of the *word* "key" ≈ latent of *seeing*
   a key in the shared geometry.
3. **Referent swap (GOLD STANDARD)**: change the manual so the word points at a
   different object ("get the GEM", gem now in key's role) — does behavior follow
   the word's NEW referent? Rules out text-ignoring and memorization. This is
   exactly why Messenger/RTFM shuffle word↔object assignments per episode (makes
   grounding necessary AND testable). First grounding experiment should be built
   around this.
4. **Compositional zero-shot**: novel instruction combining known grounded words
   executed without having seen the combination.
5. **Modality transfer**: learn "key" from vision+action, then present it ONLY in
   text — agent still acts correctly ⇒ word bound to concept, not to co-occurring
   pixels.

Where to test: develop + VERIFY grounding on cheap isolated envs (Messenger/RTFM,
or a custom DoorKey-with-manual) — Crafter is where grounded language PAYS OFF
(real tutorials, deep tree), not where it is first established. Same tier logic as
[[hierarchy-and-credit]] and the compute-efficiency principle ([[compute-strategy]]).

## The three installation architectures (testable spectrum)

| # | mechanism | "new model" lives in | nearest literature | cost |
|---|---|---|---|---|
| 1 | **text-as-context** — tutorial embedding carried in the belief state; dynamics head conditions on it | in-context (no weight change) — world-model ICL | Dynalang ([[dynalang-2023]]); EMMA's entity attention (Messenger) | low — first Messenger experiment |
| 2a | **text-as-data, own dreams** — world model imagines rollouts from text; trains on them | gradients, via synthetic experience | (apparently unexplored in latent WMs — original-idea candidate, run wall protocol before claiming) | medium — but circular: requires text understanding first |
| 2b | **text-as-data, external imagination engine** (maintainer, 2026-06-12) — text → domain video generator (Oasis / Dreamer-4-WM / Genie-3 class) → IDM action labels (VPT) → replay as trust-weighted synthetic experience → **corroboration gate**: real play validates/reinforces or discounts ("verstärkt oder nicht") | gradients, via externally generated experience | Oasis, Cosmos (synthetic-data platform), VPT IDM; gate ≈ MBPO's model-trust lessons | medium; breaks 2a's circularity — the external model carries the text grounding; unifies ADR-0005 signals 2+3 (text rides the video path) |
| 3 | **text-as-weights** — hypernetwork: tutorial → Δθ of dynamics head | explicit fast weights | Schmidhuber 1991 fast-weight programmers ([[intellectual-lineage]]); knowledge-editing (ROME/MEMIT) | high |

Prediction worth pre-registering when the thread opens: (1) suffices for
Messenger-scale (manual fits in context); (2)/(3) become necessary when tutorials
exceed context or must persist across episodes (Minecraft wiki scale).

The cognitive analogy (maintainer): humans convert text into mental imagery/simulation
([[intellectual-lineage]]: Craik's "try it in the head"), which becomes belief only
through corroboration against experience. Human imagination is not exact simulation
either — it works *because of* constant verification and reinforcement, not despite
inexactness. So hallucination is not disqualifying; the gate IS the mechanism.

**Unified trust-weighted replay (maintainer, 2026-06-12):** every transition carries a
trust weight w. The SOURCE sets the prior (own validated play: high; others'
IDM-labeled video: medium; generated video: low; own dreams: lowest); CORROBORATION
updates it (consistency with verified experience raises w, contradiction lowers it);
training loss scales with w. "Real" vs "imagined" stops being categorical — just
different priors on one scale; imagination must earn the weight real experience
gets at birth. Engineering neighbors: prioritized replay (different objective),
MBPO's model-trust lessons, Bayesian source priors. Note the symmetry with this
KB's own epistemics (sources enter unverified, trust is earned by verification).
Cheap prototype (pre-Minecraft): inject deliberately corrupted synthetic
transitions into Crafter replay with low priors; verify the gate discounts them
and clean synthetic data earns weight. Gate failure modes are
[[retention]]-adjacent (synthetic data = another distribution shift).

## Horizon: self-generated hypotheses (maintainer, 2026-06-12)

Once binding + installation + the trust gate exist, the source of new beliefs can
shift from external text to the agent itself: compose novel candidate dynamics in
latent space ("ideas"), enter them as lowest-prior beliefs, then **act deliberately
to validate them** — the agent internalizes the research loop (PROCESS.md) as
cognition. Ancestry: Schmidhuber's artificial curiosity ([[intellectual-lineage]]);
modern form: Plan2Explore-style world-model-uncertainty exploration (queue).
Practical near-term echo: hypothesis-driven exploration is the principled fix for
the ignition problem (exp 0008) — this horizon idea has a rung-2-scale prototype.

## Why it matters here

This is the staircase's scientific payload — (i) Messenger/RTFM, (ii) text-Crafter,
(iii) Minecraft — and the place where our JEPA thread (aligned latent spaces) meets
the agent thread. Also interacts with [[retention]]: installation method 2 trains
on synthetic data — another distribution-shift source the flywheel lessons apply to.

## Links

[[0005-minecraft-milestone]] · [[intellectual-lineage]] · [[retention]] ·
[[jepa]] · [[agent-architecture]] · [[hierarchy-and-credit]] · [[temporal-abstraction]]
(grounding rides on top of the hierarchical actor — its subgoals/events are the
attachment points words bind to)

## Word associations: an "association matrix" IS a word embedding (maintainer, 2026-06-13)

The maintainer: humans have associations with words; the word itself "lands in latent space".
Proposed associated-words-as-metadata, or an association matrix — "not sure it
scales." Sharpening:

- Two kinds of meaning: **grounded** (word↔world: "key"↔percept — capability #4) and
  **associative** (word↔word: "key"↔"lock"↔"open"). Humans have both.
- **An association matrix IS a word embedding.** Distributional semantics: word2vec/
  GloVe implicitly factorize a (shifted-PMI) co-occurrence matrix (Levy & Goldberg
  2014). So don't rebuild it as metadata (the maintainer's "doesn't scale" instinct is right —
  the scalable form is the dense embedding VECTOR = compressed associations). Import
  it free from a pretrained text embedding/LLM.
- Architectural move: take the associative structure for free; BIND it to OUR grounded
  latent (contrastive/CLIP/VL-JEPA) — because an LLM's associations are word↔word
  (ungrounded), and we need word↔our-experience.
- **Why associations are central to tutorials — transitive grounding:** they let a
  NEVER-grounded word reach experience via a grounded neighbor. Agent grounded
  "furnace" from play; tutorial says "smelt"; the association smelt↔furnace bridges
  the ungrounded word to grounded experience. Associations are the bridge that makes
  tutorial words connect to the agent's world even for unfamiliar words. ⇒ grounding
  is two-hop: word ↔ (embedding/associations) ↔ grounded concepts.

## Ideas to look into later: symbols-as-thought, not pixels (maintainer, 2026-06-13)

Forward ideas, not decisions — flagged to revisit when the language rung opens. They
sharpen HOW language should enter, and double as the reason the [[frozen-encoder-lean]]
(reuse frozen DINO/JEPA vision) survives even at the language rung.

- **Pixels-for-percept / symbols-for-thought boundary.** Two different jobs, opposite
  right answers. (a) *Text-as-percept* — a sign/menu/tutorial screen the agent SEES in
  the world — genuinely arrives through the eyes; render→frozen vision encoder, fine.
  (b) *Text-as-thought* — language as a medium of abstraction/planning — must NOT go
  text→pixels→vision encoder: the symbolic structure is already in the tokens, and
  rendering throws it away to make a model re-derive a lossy shadow. So the
  "translate text to vision first, keep one encoder" path is right for (a), wrong for
  (b). (Distinct from text→video→latent, which stays legit — its job is *generating
  experience to train on* (install-arch 2b), not *binding meaning to belief*. Same
  operation, different job.)

- **The deep question is symbol→belief/memory binding, and it's more than image-gen.**
  A word should be a *pointer into the region of belief space* the agent has
  experienced as that thing (a "door" = the cluster of latent states/affordances of
  doors), not a pixel pattern. Meaning = learned alignment between symbol and belief
  region — the [[#What it is]] binding/installation problem at its deepest. This is
  also where language becomes a CAPABILITY jump, not an I/O channel (the ape↔human gap:
  language as the big cognitive difference): a compositional, recombinable abstraction
  layer over latent state lets the agent **imagine in abstractions, not pixels** —
  "what if I had a pickaxe" is trivial symbolically, near-impossible pixel-by-pixel;
  counterfactuals, negation, the-not-yet-seen become representable ⇒ systematic
  generalization (the thing flat sensory latents are worst at). Binding language in
  adds a NEW KIND OF LATENT the imagination can run on.

- **LLM-as-aligned-source-into-belief = PAN's leverage, done experience-primary.**
  PAN gets its power by making the LLM the *substrate* (capped at the text prior); the
  clean alternative gets the SAME pretrained symbolic structure as an *overridable
  source aligned into our belief space*. Concrete sketch: keep the frozen vision
  encoder for perception; add a separate *language pathway* mapping tokens (seeded by
  LLM embeddings, for the free associative/compositional structure — see association-
  matrix section) into the SAME belief space the world model uses, with the alignment
  LEARNED BY GROUNDING (text bound to the experiences it describes). Then imagination
  runs over a latent that is simultaneously experiential and symbolic. Closest existing
  work: **[[dynalang-2023]]** (language as a predicted modality inside Dreamer, token-by-token
  streaming — see that page for the static-manual mismatch with our rung-4 design); this
  thesis goes past it on two axes — deep symbol↔belief *binding* (not just predicting tokens)
  and *reusing the LLM prior* rather than learning symbolic structure from scratch. See
  PAN/GLP + Critiques-of-World-Models in [[QUEUE]] (the substrate-vs-source axis) and
  [[generative-vs-predictive]].
