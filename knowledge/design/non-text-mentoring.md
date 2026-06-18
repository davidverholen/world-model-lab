---
status: draft
owner: world-model
scope: local
verified: false
last_reviewed: 2026-06-18
---

# Non-text mentoring — teach grounded goal-reaching by SHOWING, not telling

**Status: draft direction (2026-06-18), from a strategic conversation.** The maintainer's challenge: an
LLM only models the distribution of human *descriptions* of reality — a simulation of understanding, not
grounded understanding. For an agent that must *act* in a world, understanding has to come from its own
loop of predict → act → check reality, and the [[mentored-learning-loop]] north star should be mentor-able
*without* text. This page is the **non-text branch** of that north star: mentor the world-model +
goal-reaching substrate directly; treat language as an optional naming layer attached *after* a grounded
skill exists (the inversion of LLM-first framing).

Text/reading (rungs 0044–0068) is hereby reframed as the **interface layer** — useful, but the least
fundamental part — and **shelved** ("revisit later"). exp0067/0068 established it reads *meaning* but
generalises only as far as its language diversity; that's an interface-robustness problem, not the core
understanding problem.

## The channels (more primary than language)

1. **Demonstration / imitation** — the mentor *shows* the trajectory; the agent reproduces it, grounded in
   its own WM. (How every skill is taught before it can be named.)
2. **Goal-state** — the mentor *points at a desired state* (a target latent/image); the agent plans to
   reach it (the LEXA achiever pattern). Instruction = a place in the world model, not a sentence.
3. **Evaluative feedback** — warmer/colder, approval/correction shaping behaviour toward goals.
4. **Shared interaction / scaffolding** — the mentor acts in the same world / shapes the curriculum.

Unifying principle: **mentor the grounded substrate (WM + goal-reaching), with reality as judge** (the
same "validation gates dreaming" rule as [[mentored-learning-loop]] and [[validated-reading-reward]]).

## Env feasibility finding (2026-06-18) — demonstration, NOT goal-state, fits crafter-rtfm

Empirical check (scripted-gesture rollout on `crafter_rtfm`):
- **A scripted oracle that emits the DISPLAYED gesture (the known action sequence) completes the recipe**
  (triggers `tutorial_newly`) — so mentoring-by-showing needs no materials/navigation oracle. ✓
- **But goal-STATES are barely distinctive**: start→end pixel change ~2–3, and *cross-recipe* end-state
  differences only ~4.5/255 mean. crafter-rtfm's task is "emit this action sequence" (reading-grounded),
  not "reach this world-state" — the goal *image* carries almost no recipe information. So pure
  goal-state conditioning is **ill-posed here** (the goal latent can't reliably separate recipes).

⇒ **For crafter-rtfm the right non-text channel is DEMONSTRATION** (the gesture trajectory carries the
full recipe), not goal-state. Goal-state mentoring belongs in a *state-goal* env (base Crafter
"reach a state with item X", MiniGrid) — a separate rung if/when we want it.

## Proposed rung — demonstration mentoring + a grounding test

**Mentor:** a scripted oracle performs the episode's (possibly SWAPPED) displayed gesture → a demonstration
trajectory of states (DINO latents) + actions. **Agent:** conditioned on the demonstration (NOT the text
manual), learns to reproduce the gesture in its own rollout. **No text, no LLM, no reading.**

**The grounding test (the swap-following analog) = DEMONSTRATION-FOLLOWING:** show a demonstration of a
*swapped* gesture (≠ the habitual/default recipe); does the agent perform the *demonstrated* gesture
rather than its habit? A memoriser/habit-learner fails; a grounded imitator follows what it was shown.
This is the honest test that the demonstration is *grounded*, not copied — especially when the demo comes
from a *different* world instance than the agent's episode (so it can't pixel-copy).

**Comparison:** demonstration-conditioned vs the rung-4 text-conditioned agent on the same swap test — is
showing a workable (better? cheaper? more grounded?) mentoring channel than telling?

## Open design questions (resolve at build)

- **Demo representation the agent conditions on:** the demonstrated *state* latents (watch-and-infer-actions
  — the interesting, grounded version) vs the demonstrated *action* sequence (near-copy — a weaker test).
  Lean state-latents, with the demo from a *different* instance to force generalisation over copying.
- **Same-seed leakage:** if the demo and the agent's episode share a world, reproducing is trivial copying.
  Use a *different* instance for the demo (same recipe, different world) so success requires grounding.
- **Architecture:** condition the actor/WM on a demo embedding (pooled demo-trajectory latent) — reuses the
  conditioning machinery, swapping text tokens for a demo latent. Reviewer-gated build.
- **Reward:** the existing achievement + path reward (reach/perform the demonstrated gesture).

## Staged plan (bottom-up, each rung a falsifiable test)

1. **exp0069 — length-1 demonstration-following (minimal):** mentor shows a 1-action gesture (from a
   different instance); agent must perform it; **swapped-demo test** = does it follow the shown action vs
   habit? Kill signal: demonstration-following ≈ chance ⇒ the demo isn't grounding.
2. Length-2 demonstration-following (the composition the text agent plateaued at ~0.20 — does *showing*
   beat *telling* on the 2-step wall?).
3. (Optional, separate) goal-state mentoring in a state-goal env (base Crafter / MiniGrid).
4. Attach language *last*: name the grounded skills (the read→LLM seam becomes a convenience over a
   grounded substrate, not its foundation).

## Links

[[mentored-learning-loop]] · [[validated-reading-reward]] · [[hierarchical-imagination-agent]] · [[rung4-manual-conditioned-agent]] · [[0067-rtfm-reading-generalization-probe]] · [[lexa-2021]] · [[language-grounding]]
