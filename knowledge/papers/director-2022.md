---
status: draft
owner: agent
scope: local
sources: [arxiv:2206.04114]
verified: true
last_reviewed: 2026-06-15
---

# Director: Deep Hierarchical Planning from Pixels (Hafner, Lee, Fischer, Abbeel — NeurIPS 2022)

**Lab:** Google Research / UC Berkeley
**Project:** https://danijar.com/project/director/
**arXiv:** 2206.04114 · **Submitted:** 2022-06-08 · **Venue:** NeurIPS 2022 (main conference)
**Read state:** method depth (primary: arXiv abstract + NeurIPS page + Google Research blog +
secondary technical summaries; PDF binary-inaccessible — key mechanism claims cross-checked
across three independent sources)

## What it is

Director is a hierarchical world-model agent that trains both a high-level **manager** and a
low-level **worker** inside the latent imagination of a DreamerV2-style RSSM. The manager
proposes subgoals in a learned discrete goal space every K=8 steps; the worker executes
low-level actions to reach each subgoal. Neither level directly sees pixels during policy
training — both learn from imagined rollouts in the world model's latent space. Subgoals are
interpretable because the world model can decode them back to images via the goal autoencoder.

## Why it matters here

**Direct answer to our localized bottleneck.** Experiments 0041–0043 closed the diagnosis with
four converging results: reading is solved (inv\_ratio>1 at length-2), the flat actor cannot
execute multi-step sequences (correct≈0). Director is the published, Dreamer-native solution to
exactly that class of failure: a manager that breaks a long goal into a sequence of reachable
subgoals, each executed by a specialized worker in imagination. It pays off on both our open
fronts at once:

- **rtfm length-2 wall** ([[0043-rtfm-execution-wall]]): the 2-step gesture is a manager-worker
  pair — manager schedules step-1 subgoal, worker achieves it, manager schedules step-2.
- **rung-3 Crafter depth plateau** ([[0007-crafter-mastery-milestone]]): wood→table→pickaxe→stone
  is the canonical long-horizon subgoal chain that flat Dreamer cannot climb.

The "reading-grounded subgoal" twist (our novel contribution) extends Director's learned goal
space: instead of the manager sampling a latent code purely from exploration, the manual
provides the subgoal sequence directly. This is the cleaner, non-incremental architecture move
that ADR-0007 (proposed) points toward.

## Details

### 1. World model substrate (DreamerV2 RSSM)

Director builds on DreamerV2 (arxiv:2010.02193): a convolutional encoder maps images to
posterior stochastic states; a GRU deterministic path carries sequence context; the combined
RSSM state `s_t = (h_t, z_t)` is the shared latent. Imagined rollouts from this RSSM are the
only training environment for both manager and worker.

### 2. Goal autoencoder: the discrete goal space

Raw RSSM latents are high-dimensional and continuous — giving them directly to the manager
creates an intractably large goal-search problem and leads to instability on many environments
(ablation result confirmed in the Google Research blog: removing the goal autoencoder causes
failure in most tested environments). Director instead learns a **goal autoencoder** that
compresses RSSM states into smaller discrete codes `z ∼ enc(z | s)`.

The architecture is a **VQ-VAE** (vector-quantized variational autoencoder). The manager's
effective action space is the codebook — it selects a code index, the decoder maps it back to a
feature-space goal vector `g = dec(z)`, and this vector is what the worker receives as its
conditioning signal. This gives:

- **Tractable search:** discrete codebook rather than a continuous manifold.
- **Reachability:** codes are learned from real RSSM states, so every code corresponds to a
  plausible model state (unlike sampling arbitrary continuous vectors).
- **Interpretability:** the decoder reconstructs an image, so the human (and debugging tools)
  can see what subgoal the manager proposed.

### 3. Temporal abstraction: K=8

The manager selects a new goal every **K=8 timesteps**. Between selections the worker takes
low-level primitive actions (one per tick). The manager operates at an 8× slower clock:
effective horizon is multiplied by ~8 relative to a flat agent. During manager imagination, the
RSSM rolls forward K steps under the worker's policy to produce the next manager state.

### 4. Worker conditioning and reward

The worker policy `π_w(a_t | s_t, g)` is conditioned on the current RSSM state AND the current
goal vector `g`. The worker receives no task reward. Its sole training signal is a **goal
similarity reward**:

```
r_worker(s_t, g) = sim(s_t, g)   [cosine/feature similarity in RSSM space]
```

Goal-reaching is self-supervised from the goal autoencoder's feature space. The worker never
sees the extrinsic reward signal — it only learns to navigate the RSSM latent space toward the
target feature vector.

### 5. Manager reward: task + exploration bonus

The manager policy `π_m(z | s)` is trained in imagination at the K-step level and receives:

```
r_manager = r_task  +  α · exploration_bonus(s)
```

- `r_task`: the true extrinsic reward from the environment (sparse; accumulated over K steps).
- `exploration_bonus`: based on the **goal autoencoder's reconstruction error** — states where
  the autoencoder incurs high prediction error are novel, so the manager is pushed to propose
  subgoals in under-explored regions of state space. This is a simple, temporally-extended
  version of curiosity that requires no separate ensemble or prediction model.

### 6. Imagination training: the two-level loop

All policy learning happens inside imagined rollouts from the RSSM:

1. **Real data collection:** run both manager + worker in the real env; store transitions in
   replay. Update the RSSM world model from replay (reconstruction + dynamics losses, as in
   DreamerV2).
2. **Worker imagination:** branch imagined rollouts of length K from sampled RSSM states;
   train the worker actor-critic on similarity reward.
3. **Manager imagination:** abstract the K-step imagined worker trajectory into a single
   manager transition; train the manager actor-critic on task + exploration reward.

Credit flows via the manager's actor-critic loss across the K-step abstraction: the manager
learns which goal sequences lead to high downstream task reward over many K-step chunks. The
worker's gradient does NOT directly optimize task reward; it only sees the similarity signal.
This clean separation prevents the worker from gaming the task reward directly and keeps the
two optimization problems disentangled.

### 7. Benchmarks and results (arxiv:2206.04114)

Director was evaluated across four domains (all from pixels):

| Domain | Director result |
|---|---|
| Control Suite (continuous, dense) | Fully matches or exceeds DreamerV2 |
| Atari (discrete, varied) | Competitive with DreamerV2 |
| DMLab levels (3D, partial observability) | Competitive |
| Ant Maze XL (3D, sparse, egocentric camera) | Director reliably reaches goal; DreamerV2 fails to explore; Plan2Explore robot flips |
| Visual Pin Pad (sparse, sequential button presses) | Director "outperforms previous methods by a large margin" |

Key finding: on dense-reward, short-horizon tasks Director is a non-regression; on **sparse-reward,
long-horizon tasks** it is a qualitative gain. The Ant Maze result is the headline: a 3D maze with
egocentric camera and no global position input, which flat exploration-based methods (Plan2Explore)
and flat reward-maximizers (DreamerV2) both fail to solve at the larger scale.

**Crafter:** The paper reports evaluation across Crafter; specific per-achievement scores are not
extractable from web sources at this read state (PDF required for the table). The stated conclusion
is competitive-to-superior performance vs the flat DreamerV2 baseline.

### 8. Limitations (arxiv:2206.04114)

- The K=8 fixed abstraction is a hyperparameter; no learned termination (unlike the Options
  framework). Mismatched natural event lengths reduce efficiency.
- Goal autoencoder quality bounds goal reachability — if the codebook doesn't cover a needed
  subgoal, the manager cannot propose it.
- Scale: Director inherits DreamerV2's RSSM, not the larger DreamerV3/4. At the time of
  publication, the RSSM was the standard; Dreamer 4 ([[dreamer4-2025]]) later moved to a 2B
  transformer, which Director's architecture could in principle be layered on.
- Two-level only: deeper hierarchies (3+ levels) are not explored.

## Our scale feasibility assessment

**Our stack (16 GB desktop, RSSM):** Director is built on DreamerV2, which is RSSM-based and
runs on a single GPU. The added cost of Director over a flat DreamerV2 is:

- Goal autoencoder (VQ-VAE on RSSM states — small, the state dim is not large).
- Two actor-critics (worker + manager) instead of one — roughly doubles the actor-critic compute.
- Two-level imagination loop — worker rolls K=8 steps per manager step, so imagination is ~8×
  longer per manager update cycle. However, the RSSM step is cheap, so this is not a memory
  bottleneck.

**Verdict: feasible on our 16 GB machine.** The RSSM backbone is already in our stack (we run
DreamerV3 RSSM on 8 GB local). The additional VQ-VAE + second actor-critic does not require
large parameter growth. The 8× imagination horizon increases wall-clock training time but not
memory. A careful implementation should fit comfortably on 16 GB and train overnight on the
desktop GPU. The local 8 GB machine could run a small-scale diagnostic (tier-1 MiniGrid) but
might be tight for a full Crafter run.

## Design section: reading-grounded Director for our rtfm wall

### (a) Where does reading fit in Director's architecture?

Director's manager proposes a subgoal by sampling from the VQ-VAE codebook via learned policy
`π_m(z | s)`. In the original, the only conditioning signal is the current RSSM state `s` and
the exploration bonus. Our proposed twist: **condition the manager on a manual-derived subgoal
sequence** extracted from the read recipe.

The manual (for a length-2 recipe) encodes: "do action A, then action B." This maps to a
2-element subgoal sequence. Instead of the manager learning which goal to propose via RL, the
manual supplies the sequence directly. The worker still trains on similarity reward inside
imagination — the worker-level mechanism is unchanged.

Two implementation strategies:

1. **Manual → goal codes (hard substitution):** tokenize the recipe text with the RSSM's
   language conditioning (our existing Dynalang-style pathway, [[dynalang-2023]]) and MAP each
   recipe step to a goal code (or to a goal feature vector in the decoder's space). Replace the
   manager's sampled code with the manual-derived code. The manager's RL is disabled during
   the recipe; execution is pure worker. The worker still needs to be trained to follow goal codes.

2. **Manual → manager conditioning (soft):** keep the manager as an RL policy but condition it
   on the manual embedding. The manager LEARNS which codebook entry best instantiates the
   manual's description of each step. This is cleaner architecturally (no hard-switching) but
   requires the manager to ground language→goal-code, which is a second learning problem.

For the minimal first experiment (see below), strategy 1 is simpler: skip the manager RL
entirely and use the manual as a sequencer.

### (b) Can a manual-derived goal slot into the VQ-VAE codebook?

The goal autoencoder codes represent RSSM states. A manual-derived goal ("do swap action") is
not an RSSM state — it is a semantic description of a transition target. The grounding problem
is: find the codebook entry whose decoded RSSM state most resembles the state we'd be in after
correctly executing the gesture. This is a retrieval problem, not a generation problem — and it
has a cheap approximation: run one episode WITH the correct gesture, record the RSSM state at
the target timestep, and find its nearest codebook neighbor. In training the codebook is
updated to represent visited states, so the target state will be representable once the worker
explores enough to visit it. This is the key fragility: if the correct goal state is
off-distribution (never reached during exploration), it won't be in the codebook. The fix is
the exploration bonus — the manager is incentivized to push into novel states, which should
eventually include the target.

For our rtfm setting (WM already reads the gesture, worker can't execute), the goal state IS
reachable (the environment is deterministic and the WM knows the dynamics). The bottleneck is
assigning credit across the 2-step sequence, which Director directly solves by giving the
manager a 2-step abstract credit channel.

### (c) Minimal first experiment: hierarchy on the rtfm length-2 wall

**Hypothesis:** A Director-style manager-worker loop (without reading conditioning; just
hierarchy) will crack the rtfm length-2 correct-execution wall, because the wall is
temporal-credit, not perception.

**Minimal implementation (3 components, in order):**

1. **Worker:** a flat actor conditioned on `(RSSM state, goal feature vector)` trained in
   imagination with a goal-similarity reward. This is a standard goal-conditioned actor — no
   new architecture beyond adding the goal vector as an input.

2. **Manager:** a small policy (MLP or thin linear head) trained in imagination at K=8
   abstraction with the existing extrinsic sparse reward. Proposes goal indices from a tiny
   codebook (can start with K-means over a buffer of RSSM states — no full VQ-VAE needed for
   the first probe).

3. **No reading conditioning initially:** the reading grounding is a separate capability
   already demonstrated at length-1. The hierarchy experiment tests EXECUTION / CREDIT-ASSIGNMENT
   in isolation. If Director-style hierarchy cracks length-2 WITHOUT reading conditioning (using
   the env's extrinsic sparse reward), the experiment confirms that hierarchy is the missing
   component. Then a follow-up adds reading conditioning to the manager, testing the
   "manual-as-sequencer" variant.

**Environment:** rtfm length-2, same setup as exp 0043. Four seeds, overnight on desktop (16 GB).
The signal is the same as 0043: inv\_ratio (WM reads, expected to stay >1), correct (expected to
lift from ≈0.05 toward ≈0.35+ if hierarchy helps), and swap\_follow (goal reachability under
swapped manual — the definitive grounding test).

**Kill signal:** if a Director-style hierarchical agent still gets correct≈0 on length-2 (with
the manual supplying the recipe and hierarchy supplying the credit channel), the diagnosis shifts
to a representation / policy-gradient problem, not a temporal-abstraction problem.

**Pre-registration note:** this experiment should be registered under [[0044]] once the
maintainer confirms the direction and architecture choices are settled in ADR-0007.

## Open questions

1. What is the exact VQ-VAE codebook size and state-compression ratio used in Director?
   (PDF required for the hyperparameter table; load-bearing for our implementation.)
2. Are the worker's imagined rollouts branched from the replay, or from on-policy RSSM states?
   (Affects how the worker's goal-conditioned policy interacts with our flywheel.)
3. Does Director's exploration bonus remain useful when the manager is conditioned on the manual
   (since the manual removes the manager's need to explore for subgoals)?
4. Can a single VQ-VAE codebook simultaneously represent both task-relevant and
   exploration-novel states? Or does the fixed codebook create a coverage gap for sparse-reward
   tasks?

## Links

[[imagination-training]] · [[hierarchy-and-credit]] · [[temporal-abstraction]] ·
[[dreamer4-2025]] · [[dreamerv3-2023]] · [[dynalang-2023]] ·
[[0043-rtfm-execution-wall]] · [[0007-crafter-mastery-milestone]] ·
[[language-grounding]] · [[rung4-manual-conditioned-agent]]
