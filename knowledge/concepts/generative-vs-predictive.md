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
