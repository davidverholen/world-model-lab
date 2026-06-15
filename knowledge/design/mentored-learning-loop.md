---
status: draft
owner: world-model
scope: local
verified: false
last_reviewed: 2026-06-16
---

# Mentored learning loop — the north-star use case

**Status: draft, north-star vision (design discussion 2026-06-15).** A long-horizon target the rung-4
grounding work + the Dyna flywheel + [[validated-reading-reward]] are aiming at. Not a near-term
experiment — the frame that says *why* the current rung matters.

## The loop

A human mentors an agent through a game, live, and the agent genuinely gets better at what it is
taught:

1. **Instruct** — the mentor tells it (in natural language) how the game works / what to do.
2. **Instant in-context memory** — it conditions on the text *immediately*, no weight change (working
   memory). The rung-4 cross-attention conditioning, generalized from a static per-episode manual to
   *streaming* text ([[dynalang-2023]] does streaming language).
3. **Act + validate against reality** — it tries the instruction in the world and checks whether it
   holds. This is [[validated-reading-reward]] / [[0048-rtfm-validated-reading]]: read → predict →
   act → reward iff reality confirms (reality is the un-fakeable judge).
4. **Consolidate by dreaming and/or practice** — it trains the actor in *imagined* rollouts of the
   world model ("dreaming" = [[imagination-training]], literally how Dreamer-class agents improve) and
   /or repeats the validated behaviour in real play (the Dyna flywheel), turning the instruction into
   skill held in weights.
5. → **retained skill**, and the mentor moves on to the next lesson.

Maps onto how humans learn: working memory → sensorimotor test → sleep/replay consolidation
([[dreaming]]: Hoel, dreams as consolidation/anti-overfitting) → procedural skill.

## The load-bearing insight: validation GATES dreaming

The ordering "validate **before** you dream" is not incidental — it is the safety property. Training
in imagination on an *unvalidated* instruction just reinforces the world model's own hallucinations —
exactly the exp 0017–0025 imagination-exploitation wall. So step 3 (validated-reading) is not only
"learn from the mentor"; it is the **gate that keeps step 4's dreams grounded**. The agent dreams to
improve *only* about dynamics reality has confirmed. The loop is self-correcting by construction, and
robust to a wrong/​adversarial mentor (bad advice fails validation → never consolidated).

This is also why the reward must be un-wireheadable (reality as judge, not self-assertion): a mentored
agent that could grant itself the reward would dream-consolidate fictions.

## The fast→slow handoff (the hard frontier)

"Remembers the text *instantly* AND keeps it *forever* without breaking the rest" is the deep open
problem: the context→weights installation axis ([[language-grounding]]: binding vs installation), run
*online* during play. Online consolidation collides with continual-learning / plasticity-loss and
catastrophic forgetting (the [[retention]] thread). So the instant-and-permanent version is a frontier;
the *session-scale* version (mentored episodes feed the flywheel, the agent improves over the session)
is buildable from parts that exist.

## Why it matters now

It is the use case that makes read-to-learn-dynamics worth climbing: an agent you can *teach by talking
to it*, that tests what you say and gets better at what works. Every rung-4 result (does it read? does
it act on what it read? does validated-reading lift obedience?) is a brick in this. See
[[rung4-manual-conditioned-agent]], [[hierarchical-imagination-agent]], [[validated-reading-reward]].
