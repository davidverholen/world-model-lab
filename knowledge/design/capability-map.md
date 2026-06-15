---
status: current
owner: world-model
scope: local
sources: []
verified: true
last_reviewed: 2026-06-13
---

# Capability Map: distinct capabilities, distinct problems

Orientation index (2026-06-13). The agent needs several SEPARATE
capabilities, each solving a DIFFERENT problem. They interact and stack, but
conflating them causes design errors (e.g. "the agent can't form 'get the key'
because it lacks language" — wrong: it lacks temporal abstraction; language is a
different capability). Keep them separate; build in dependency order.

| # | capability | problem it solves | depends on | status |
|---|---|---|---|---|
| 1 | **world modeling** (latent prediction) | predict consequences of actions in latent space | — | ✅ working (rungs 1–2; SIGReg anti-collapse) |
| 2 | **flat credit assignment** (value head) | learn what leads to sparse reward; carry credit beyond planning horizon | 1 | ✅ working (MC value, 63% > PPO) |
| 3 | **temporal abstraction / hierarchy** | reusable, context-conditional SUBGOALS; long-horizon composition | 1, 2 | ▶ next (rung 3; hierarchical actor) — [[hierarchy-and-credit]], [[temporal-abstraction]] |
| 4 | **language grounding** | bind words to THIS agent's internal latent concepts (learned, not an LLM) | 1 (+ 3 for event-level attachment) | queued (Messenger/RTFM verify; rung 3+/5b) — [[language-grounding]] |
| 5 | **language installation** (learning from tutorials) | turn declarative text into a dynamics belief WITHOUT experiencing it | 4 | rung 5b — [[language-grounding]] |
| 6 | **trust-weighted imagination** | learn safely from synthetic/generated experience (corroboration gate) | 4 (+ trust gate) | rung 5b (arch 2b) — [[language-grounding]] |
| 7 | **self-generated hypotheses** (creativity) | generate + validate own ideas; directed exploration. An idea = a hallucination that survives verification (variation+selection); the agent loop ≡ the research loop | most of the above + regulated imagination | horizon — [[generative-vs-predictive]] §novelty, [[language-grounding]] §horizon |

## The separation that keeps getting blurred

- **#3 (subgoals) ≠ #4 (language).** A rat gets the key (capability 3) with no
  words (capability 4). Temporal abstraction produces NAMELESS subgoals; language
  later LABELS them. Order: 3 before/under 4.
- **#4 (grounding) ≠ plugging in an LLM.** An LLM has words-bound-to-words;
  grounding needs words-bound-to-our-latents, learned in our geometry.
- **#5 (installation) ≠ #4 (grounding).** Grounding connects "key"↔key-concept;
  installation connects "smelting yields ingots"↔a new dynamics edge. You can
  ground every word and still not be able to install a new rule from a sentence.

## How we test each (compute-efficient, per [[compute-strategy]])

Develop+verify each capability on the cheapest env that can reveal it; deploy on
Crafter/Minecraft where it pays off. Verification must be falsifiable
(e.g. grounding → referent-swap test; hierarchy → flat-vs-hier A/B on
KeyCorridor). Never first-establish a capability on Crafter.

## Links

[[agent-architecture]] (the CURRENT system's 5 optimization layers — this page is
the FUTURE capability set) · [[hierarchy-and-credit]] · [[language-grounding]] ·
[[temporal-abstraction]] · [[environment-ladder]] · [[0005-minecraft-milestone]]
