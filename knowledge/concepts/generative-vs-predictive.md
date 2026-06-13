---
status: draft
owner: agent
scope: local
sources: []
verified: false   # design brainstorm (Dave, from a LeCun vs Eric Xing debate)
last_reviewed: 2026-06-13
---

# Generative vs Predictive (and why hallucination is a third axis)

## The debate (Dave, 2026-06-13, LeCun vs Eric Xing)

Xing: the encoder's latent is partial/lossy; some tasks need to reconstruct full
reality → you need a generative decoder. LeCun: the partiality is a FEATURE —
predict only the predictable/relevant abstraction; reconstructing pixels wastes
capacity on irrelevant detail. Both are right, conditional on the goal.

## Two orthogonal axes (the sharpening)

1. **Representation: latent-abstract (LeCun) vs generative-full (Xing).** A question
   about what you must PRODUCE, not which is "more intelligent". Xing owns
   *reconstruction* (rendering, communicating, generating training data, binding to
   imagery). LeCun owns *action* (deciding — pixel detail is wasted capacity + a
   liability).
2. **Hallucination / model-exploitation — INTRINSIC to optimizing against any
   learned model, latent OR generative.** Empirical proof, exp 0017: we imagined
   purely in latent space (no decoder, pure LeCun) and the actor STILL hallucinated
   — imagined return 2-3× the real max, chasing reward that doesn't exist. So
   "no decoder → no hallucination" is FALSE. LeCun's approach buys representation
   efficiency, NOT immunity. Verification is mandatory either way. ([[retention]]
   for the training-dynamics cousin; continue-predictor exp 0018 = our first patch.)

## Language binding: discriminative suffices to recognize; generative to EXPERIENCE

- Ground "key" ↔ key-concept: **contrastive/discriminative** (CLIP, VL-JEPA — no
  decoder, LeCun-friendly). Gets recognition.
- Turn a *declarative* sentence ("smelt ore → ingots") into *rehearsable
  experience*: needs **generation** — this is [[language-grounding]] arch 2b. So
  generation is not needed to ground words; it is needed to IMAGINE NOVEL EXPERIENCE
  from words.

## The proposed loop and its honest justification

Dave's loop: latent → decode (gen-AI, language-conditioned via an LLM) → re-encode →
latent, governed by trust-weighting. Cost: lossy, expensive, two models that can
disagree. The ONLY honest justification for round-tripping through pixels (vs just
predicting in latent space, which LeCun would demand) is **knowledge import** —
inject content from a generator that learned the world from data we don't have
(text→image/video, Oasis, VPT corpus; see ASSETS.md). You borrow its grounding and
launder it into your latent space. Value = importing external experience, NOT better
prediction. Where you only need prediction, render nothing.

## The unifying principle and the human analogy

"Real experiences must outweigh imaginations" ([[language-grounding]] trust-weighted
replay; exp 0018 continue predictor; the corroboration gate) is not a design
preference — it is the only thing between useful imagination and delusion, in brains
(confabulation, false memory, vivid dreams, bad plans from wrong imagined chains)
and in our agent (exp 0017 reward-farming). Humans run BOTH: fast latent prediction
(System-1, LeCun) + vivid generative simulation (episodic imagination, Xing), kept
sane by constant reality-testing.

## Map to our roadmap

Rungs 1–4: LeCun-pure (latent, no decoder — cheaper, no reconstruction needed to
act). Generative/Xing element enters at rung 5b (Minecraft, language→experience),
gated behind a verification mechanism we are learning to build now. Exps 0017→0018
are the first, smallest instance of the exact problem the generative loop lives or
dies on.

## Links

[[language-grounding]] · [[capability-map]] · [[temporal-abstraction]] ·
[[world-models]] · [[jepa]] · [[0017-imagination-actor-critic]] ·
[[0018-continue-predictor]] · [[lecun-2022-path]]

## Imagination as the source of novelty (Dave, 2026-06-13)

The reframe: an original idea IS a hallucination that survived verification.
Imagination = blind variation; reality-testing = selection; creativity = the loop
(Popper conjecture/refutation; Campbell blind-variation-selective-retention; Dennett;
Schmidhuber curiosity — [[intellectual-lineage]]). Consequences:

- **Value is in what survives, not in the generation.** Most imagination is
  degenerate (exp 0017 reward-farming = a *bad idea*, killed by the continue
  predictor 0018 = its *refutation*). Human "creativity" is survivorship — we forget
  the thousands of stupid imagined chains. So "hallucination → ideas" holds ONLY
  because a brutal selection filter runs behind it.
- **Creativity–safety tradeoff:** a perfectly-calibrated never-hallucinating agent
  is perfectly UNCREATIVE (can only reproduce experience). Full suppression is the
  wrong target. Right target = REGULATED imagination: channel toward
  plausible-unexplored (entropy bonus — we have one in the actor; curiosity;
  Plan2Explore) + verify by acting. LeCun's exact-prediction pole is safe;
  free-association is delusional; creativity is the verified middle.
- **The closure:** the agent's creativity loop ≡ our research loop (conjecture =
  imagined claim, refutation = experiment). The lab and the mind run the same
  variation-under-selection algorithm. Capability #7 (self-generated hypotheses,
  [[capability-map]]) is this, internalized — and is what "autonomous" finally means.

The machinery that SUPPRESSES bad imagination (continue predictor, trust-weighting,
corroboration) is the SAME machinery that, pointed at plausible-unexplored regions,
CULTIVATES good imagination. Suppression and creativity are one mechanism, two aims.
