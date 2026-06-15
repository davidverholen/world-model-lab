---
status: draft
owner: world-model
scope: shared
sources: [arxiv:2006.15762]
verified: true
last_reviewed: 2026-06-15
---

# Empirically Verifying Hypotheses Using Reinforcement Learning (Marino et al., 2020)

**Authors:** Joseph Marino, Rob Fergus, Arthur Szlam, Abhinav Gupta · **Lab:** Facebook AI Research (FAIR) · **Venue:** arXiv preprint (2020) · **arXiv:** 2006.15762

## What it is

This paper frames exploration as **hypothesis verification**: rather than exploring to maximize prediction novelty or coverage, the agent is given a specific hypothesis about world dynamics and must perform a sequence of actions that generates observations sufficient to confirm or refute it. A hypothesis is a structured triplet — pre-condition / action-sequence / post-condition — and the agent receives a reward when the hypothesis's truth value is determined by **real-environment interaction** (not by a model). The reward is gated on the outcome of actual experience, not on the agent's internal predictions or declarations.

## Why it matters here

This is the **closest published analog to our validated-reading reward** (exp0048). The structural similarity is precise: in both cases, the agent must act to generate observations that test a claim about dynamics, and the reward fires when the real environment delivers a verdict. The anti-wireheading property is identical — the reward is gated on reality, which the agent cannot fabricate. However, Marino et al. is missing the language-reading component, the marginal-value framing, and the explicit grounding-of-a-manual objective. Understanding what is and is not shared between this paper and our mechanism is the primary purpose of this page.

## Details (method depth)

### Hypothesis structure

A hypothesis H is factored as a **pre-condition / action-sequence / post-condition** triplet:

- **Pre-condition** π_H: a predicate on the current state (e.g. "door is closed and agent is adjacent").
- **Action-sequence** α_H: a sequence of actions to execute (e.g. [pick-up key, move to door, open door]).
- **Post-condition** ρ_H: a predicate on the state after executing α_H (e.g. "door is open").

A hypothesis is **confirmed** if the pre-condition holds before execution, all actions are executed, and the post-condition holds after. It is **refuted** if the pre-condition holds, the actions are executed, and the post-condition does not hold. The experiment is **inconclusive** if the pre-condition did not hold (the agent couldn't set up the test).

### The experiment-design / reward structure

The agent's task is to reach a state where the pre-condition holds, execute the action-sequence, and then observe whether the post-condition is satisfied. The reward signal is:

```
r = +1  if H is confirmed  (pre held, actions taken, post holds)
r = -1  if H is refuted    (pre held, actions taken, post fails)
r =  0  if inconclusive    (pre-condition not met, no valid test)
```

This is a **resolving reward** — the agent earns reward for *determining the truth value* of the hypothesis, regardless of whether it is true or false. Confirmation and refutation are both informative outcomes; failure to set up the experiment earns nothing.

Critically: the reward is computed by comparing the real environment's post-condition observation against the hypothesis's predicate, NOT by querying the agent's model. The environment is the judge.

### Policy structure

The agent runs a two-phase policy:

1. **Setup phase**: navigate the environment to satisfy the pre-condition (an undirected exploration problem, potentially with its own intrinsic reward or subgoal guidance).
2. **Execution phase**: execute the action-sequence α_H and observe the outcome.

The paper experiments with both policy-gradient (REINFORCE) and Q-learning variants for the setup phase. The execution phase is deterministic (the action-sequence is given); only the setup is learned.

### Hypothesis generation

In the paper, hypotheses are provided externally or sampled from a prior over (pre-condition, action-sequence, post-condition) templates. There is no mechanism by which the agent reads a text description and derives a hypothesis from it. Hypothesis generation is treated as a separate, unsolved problem and is not the paper's contribution.

### Anti-wireheading property

Because the reward is computed by the environment (real post-condition observation), not by the agent's model, the agent cannot engineer the reward without actually performing the action and receiving the real-world outcome. This is the same structure as the RND fixed-target anti-wireheading pattern ([[rnd-2018]]): there is an incorruptible external judge. In Marino et al., the judge is the environment's state transition; in RND, it is the frozen random network; in our mechanism ([[validated-reading-reward]]), it is the real embedding of the next state against which the manual-conditioned WM prediction is scored.

### Key experimental finding

The paper demonstrates that hypothesis-verification reward effectively guides exploration in environments where hypotheses capture the relevant structure (grid-world, simulated manipulation). The reward is sparse per hypothesis but more informative than undirected exploration because it targets specific dynamics claims. The paper also shows the agent learns to set up valid experiments (reach the pre-condition) as a prerequisite skill.

## Open questions

- **Hypothesis generation at scale**: the paper leaves hypothesis generation entirely to an external oracle or hand-designed prior. For the mechanism to be autonomous, a module that proposes verifiable hypotheses is required — analogous to our manual-reading step.
- **What if many hypotheses are available?** The paper does not address prioritization among competing hypotheses (which to test first), a problem our marginal-IG framing naturally handles (test the manual whose marginal gain is highest).
- **Generalization of post-condition predicates**: the triplet structure assumes predicates can be evaluated on real observations. For continuous or high-dimensional observations, predicate evaluation requires a learned classifier, adding another inductive bias.

## Relevance to our work

### Structural parallels to exp0048

The validated-reading reward ([[validated-reading-reward]] / [[0048-rtfm-validated-reading]]) shares the core structure of Marino et al.:

| Dimension | Marino et al. | Our exp0048 mechanism |
|---|---|---|
| Claim about dynamics | Structured triplet (pre/action/post) | WM prediction conditioned on manual |
| How claim is generated | External oracle or template prior | Agent reads the episode manual |
| How claim is tested | Agent acts to set up pre-condition, executes action-sequence | Agent follows predicted action-sequence |
| Judge | Real environment post-condition | Real next-state embedding vs WM prediction |
| Reward sign | +1 confirmed / -1 refuted (resolving) | clip≥0 (marginal gain of correct manual) |
| Anti-wirehead | Yes — environment is the judge | Yes — real embedding is the judge |

### What Marino et al. is missing vs our mechanism

1. **No language / reading step**: hypotheses are given as structured triplets, not derived by reading a text manual. The grounding problem — converting natural language to verifiable predictions — is entirely absent. Our mechanism begins where Marino's ends: we must first extract a hypothesis *from the manual*, then design the act-to-verify loop.

2. **No marginal-value framing**: Marino et al. rewards the agent for resolving a hypothesis (true or false). Our mechanism rewards the **marginal contribution of the manual** — the correct manual must improve the WM's prediction over a null-manual baseline. Without the marginal framing, the agent could earn reward from dynamics it already knows without the manual's help. The marginal framing is what forces obedience to the *specific* manual content.

3. **No contrastive dual pass**: our computation requires two forward passes (with-manual vs null-manual) and takes their prediction-error difference. Marino et al. is a single-pass execution.

4. **Resolving vs confirming**: Marino's reward fires for *both* confirmation (+1) and refutation (-1). Our reward fires only for confirmation (clip≥0). For grounding the manual, refutation is less useful: we need the agent to learn that correct manuals predict reality, not just that they make falsifiable predictions. The asymmetric clip is a design choice that may need revisiting if the dark-room case is too strong.

### Relationship to the foil and anti-wirehead pattern

- **[[icm-2017]]** (foil): ICM rewards prediction ERROR — the wrong sign for "this manual predicted reality correctly." Both Marino et al. and our mechanism are inverted in spirit from ICM.
- **[[rnd-2018]]** (anti-wirehead pattern): RND's un-fakeability comes from a frozen random target. Marino et al.'s un-fakeability comes from the real environment; our mechanism's comes from the real next-state embedding. All three instances of this pattern use an external judge the agent cannot manipulate — the core anti-wireheading principle.
- **[[vime-2016]]**: VIME provides the information-gain framing that underlies our prediction-error difference (marginal IG). Marino et al. does not use information gain; it uses a binary resolving reward. Our mechanism combines Marino's act-to-verify structure with VIME's IG signal, applied contrastively.

## Links

[[validated-reading-reward]] · [[0048-rtfm-validated-reading]] · [[vime-2016]] · [[icm-2017]] · [[rnd-2018]] · [[language-grounding]] · [[0040-rtfm-actor-conditioning]] · [[hierarchical-imagination-agent]]
