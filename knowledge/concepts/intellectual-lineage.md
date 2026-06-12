---
status: draft
owner: agent
scope: local
sources: []
verified: false   # canonical classics from training knowledge; verify ids on ingest
last_reviewed: 2026-06-12
---

# Intellectual Lineage of World Models

## What it is

The decades-deep background behind our project, organized as four threads. Items
marked (queued) have QUEUE entries for proper ingestion; the rest are background
context — cite only after verification.

## Thread 1 — Mental models in minds (psychology → the core idea)

- **Craik 1943**, *The Nature of Explanation*: the founding articulation — organisms
  carry "small-scale models of reality" in their heads and try out actions in the
  model before acting. Our L1 planning is this sentence, implemented. (queued)
- **Tolman 1948**, "Cognitive Maps in Rats and Men": rats learn maps, not
  stimulus–response chains — internal models demonstrated empirically, against the
  behaviorists. The original "model-based vs model-free" debate.
- **Kahneman 2011**, *Thinking, Fast and Slow*: System 1 / System 2 — explicitly the
  source of LeCun's Mode-1/Mode-2 ([[lecun-2022-path]]); our CEM planner is a
  System-2; the future distilled policy is its System-1.
- **Spelke** (core knowledge program): infants have priors for objects, continuity,
  contact causality — the empirical basis for "intuitive physics" as a learnable
  target (cf. Meta's intuitive-physics-from-video work, QUEUE 2502.11831).

## Thread 2 — The predictive brain (neuroscience)

- **Rao & Ballard 1999**, predictive coding in visual cortex: cortex as a hierarchy
  of predictors exchanging prediction *errors* — the neuroscience twin of latent
  predictive learning. (queued)
- **Friston 2010**, the free-energy principle: perception and action both minimize
  surprise under a generative model. Philosophically adjacent to the energy-based
  framing of JEPA; mathematically contentious; read as inspiration, not spec.

## Thread 3 — LeCun's own arc (1989 → 2022, one consistent bet)

- **LeCun 1989/1998** (LeNet, "Gradient-Based Learning Applied to Document
  Recognition"): convolutional networks — our ConvEncoder's direct ancestor.
- **Chopra/Hadsell & LeCun 2005/2006**: siamese networks + contrastive loss
  ("Dimensionality Reduction by Learning an Invariant Mapping") — learning
  representations by comparing embeddings, the methodological seed of joint
  embedding architectures. (queued)
- **LeCun 2006**, "A Tutorial on Energy-Based Learning": the EBM framework — score
  compatibility instead of normalized probabilities; the formal language the 2022
  path paper and JEPA are written in. (queued)
- **2016 "cake" talk**: self-supervised learning as the bulk of intelligence
  (the cake), supervised the icing, RL the cherry — the strategic claim our
  data-hungry-RL experience (exp 0007/0008!) keeps confirming.
- **[[lecun-2022-path]]** (2022): all of the above composed into one architecture.

## Thread 4 — Model-based RL (the algorithmic lineage)

- **Sutton 1991, Dyna**: integrate learning, planning, and reacting — act, learn a
  model, plan with the model, repeat. **Our collect→train→plan flywheel IS Dyna**,
  with deep networks; recognizing this names our exp-0009/0010 problems as "Dyna
  with function approximation under non-stationarity". (queued)
- **Schmidhuber 1990–91**: "making the world differentiable", artificial curiosity —
  learned neural world models + intrinsic motivation, 30 years early.
- **Kaelbling, Littman & Cassandra 1998**, POMDPs: belief states as sufficient
  statistics of history — our GRU belief is a learned approximate belief state.
  (queued)
- **Ha & Schmidhuber 2018** → PlaNet → Dreamer → today: see [[world-models]] /
  [[imagination-training]].

## Why it matters here

Almost every component we built has a 30–80-year-old ancestor, and the ancestors
predict our failures: Dyna's literature knew iterated model retraining is fragile;
Tolman's critics demanded the model demonstrably help action (our PPO baselines);
Craik's "try it in the head first" is the whole bet. The lineage keeps us honest
about what is genuinely new (training *stability* of deep latent world models under
continual collection — the gap our experiments live in).

## Links

[[lecun-2022-path]] · [[world-models]] · [[agent-architecture]] · [[retention]] ·
[[jepa]] · [[imagination-training]]
