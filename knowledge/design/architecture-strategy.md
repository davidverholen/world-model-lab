---
status: draft
owner: human
scope: local
sources: [openreview:lecun-path]
verified: false   # design strategy (maintainer, 2026-06-13); resolve empirically at rung 3/5b
last_reviewed: 2026-06-13
---

# Architecture Strategy: modes, interfaces, reusable components

How the SYSTEM is composed (vs [[agent-architecture]] = what the current system IS,
vs [[capability-map]] = what capabilities it needs). Design brainstorm with the maintainer;
the calls here are resolved empirically at rung 3/5b, not committed abstractly.

## Epistemic modes (not the default loop)

The generative / external-learning / creative machinery is NOT the default training
loop. Default = learn from OWN sensory experience (immediately grounded, JEPA-pure,
safe). Special modes (reading a tutorial; learning from any external source) differ
in exactly two things: the SOURCE of the signal, and the VERIFICATION REGIME applied
to it. Own experience earns trust on arrival; external/generated input must be
decoded → imagined → verified before earning trust. So a "mode" is a trust+
verification setting over (largely) the same components — i.e. LeCun's
**configurator** orchestrating modules per task. (Trust-weighted replay,
[[language-grounding]]; verification = [[generative-vs-predictive]].)

## The real decision is the INTERFACE, not the model count

Monolithic-vs-modular is the wrong axis (genuinely unresolved in the field, and
scale/task dependent). With a STABLE interface (for us: the belief-state vector
every consumer reads), "one model or many" becomes reversible and per-component:
swap a frozen pretrained encoder (DINOv3), reuse a pretrained world model, retrain
just the actor — OR train end-to-end when joint optimization helps. Modern systems
= differentiable modules with stable interfaces, jointly-trainable or staged as
needed. Our agent already is this: `encoder → belief → {reward, value, actor,
continue}` on one 256-d interface. ⇒ **Invest in keeping the interface clean and
documented**; that makes the monolithic/modular call cheap, reversible, per-component
— not a one-way bet.

## Reusable pretrained components (the Minecraft WM)

Two independent arguments to reuse a pretrained Minecraft world model as a frozen
module:
1. **Knowledge import + compute**: we cannot reproduce that pretraining; reuse is
   the compute-efficiency moat ([[compute-strategy]]; cf. retained VPT weights,
   ASSETS.md).
2. **Components want different lifecycles** — our encoder-freeze finding ([[retention]]
   exp 0013): the world model/encoder wants to become a STABLE FROZEN reusable
   component once good; actor/heads stay PLASTIC. We discovered staged modularity by
   debugging retention. A pretrained Minecraft WM is that frozen component handed to
   us pre-trained (à la DINO-WM: frozen features + learned dynamics).

## Synthesis + how we decide

Shape: **reusable components + stable interfaces + a per-mode orchestrator**
(= the modularity answer, the mode-switching answer, and the reusable-WM answer at
once; = LeCun's Path-paper architecture). Decide joint-vs-staged and
frozen-pretrained-vs-from-scratch EMPIRICALLY at rung 3/5b via the tier ladder
([[hierarchy-and-credit]]), not abstractly. Literature-first next move: FULL read of
[[lecun-2022-path]] (the reference for modular cognitive architecture — the maintainer has now
re-derived its configurator/modules from first principles 3×).

## Links

[[agent-architecture]] · [[capability-map]] · [[lecun-2022-path]] · [[retention]] ·
[[compute-strategy]] · [[generative-vs-predictive]] · [[language-grounding]] ·
[[0005-minecraft-milestone]]
