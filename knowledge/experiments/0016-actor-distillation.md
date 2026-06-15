---
status: draft
owner: world-model
scope: local
sources: [arxiv:2301.04104]
verified: true
last_reviewed: 2026-06-13
---

# 0016: Planner distillation fails (BC compounding error) — go on-policy (run 2026-06-13)

## Hypothesis

Tier-1 mechanism check for the actor thread ([[hierarchy-and-credit]],
[[capability-map]] #3 substrate). A flat reactive policy π(a | belief), distilled by
behavioral cloning from the CEM planner of the trained 80% world model
(`dk6_utd_s0.pt`), reaches **≥ 0.7× the planner's own eval success** on DoorKey-6x6,
at ~1000× lower inference cost (one MLP forward vs 512×20×2 imagined rollouts).
If so: we have a Mode-1 policy (the Crafter-scale-collection prerequisite) and a base
for the hierarchical actor. Counter-outcome: actor underperforms badly → the
planner's competence is not representable from the belief state reactively (the
belief under-encodes, or the policy needs more than the GRU's recurrence) → informs
whether the imagination actor-critic (exp 0017) needs a richer policy/belief.

Lineage: planner→policy distillation = AlphaZero/expert-iteration pattern; the
imagination actor-critic target is DreamerV3 (already ingested: horizon 15, λ 0.95,
entropy 3e-4). This experiment is the supervised, low-risk precursor.

## Setup

Load `runs/remote/dk6_utd_s0.pt` (teacher = RecurrentMPCAgent, H20/512/iter2).
Collect ~8k (belief_state, planner_action) pairs by rolling the teacher in
DoorKey-6x6 (partial obs). Train Actor MLP (state_dim 256 → 7) by cross-entropy.
Eval: ActorAgent alone (argmax) vs teacher, 20 episodes each, seeds 0–19. Local,
~minutes. Seeds 0/1/2 for the actor train+eval.

## Result

3 seeds, local (~5 min total; laptop ran hot — the TEACHER's CEM planning is the
cost, not the actor):

| seed | train_acc | actor success | teacher success | ratio |
|---|---|---|---|---|
| 0 | 0.54 | 0.00 | 0.25 | 0.00 |
| 1 | 0.54 | 0.00 | 0.25 | 0.00 |
| 2 | 0.52 | 0.05 | 0.30 | 0.17 |

**Refuted** (bar ratio ≥ 0.70). Two findings:

1. **BC distillation of a planner is the wrong tool here.** Train accuracy caps at
   ~54% (the CEM teacher is *stochastic* — same belief → different argmax across
   calls → noisy imitation target) AND eval collapses to ~0% even where per-step
   accuracy is decent: classic behavioral-cloning compounding error, O(ε·T²) in
   horizon (literature gate: DAgger / Ross-Bagnell). Our ~25-step key→door→goal
   chain is maximally brittle — one wrong action and the episode is unrecoverable,
   in states the off-policy actor never saw.
2. **Eval-rigor surprise (important, separate):** the same `dk6_utd_s0.pt` teacher
   scores 25–30% on 20 fresh seeds (0–19) vs the 80% it logged during training (10
   episodes, seeds 10000–10009). Our headline success numbers are **high-variance
   10-episode point estimates**; absolute values across the retention arc should be
   read as soft (relative/within-protocol comparisons still hold). ⇒ standardize on
   ≥20 eval episodes over a fixed wider seed set going forward.

## Lesson

The fix the literature mandates is **on-policy** training (the actor must train on
its OWN state distribution). Two routes: DAgger (keeps the slow planner in the
loop, more orchestration) vs **imagination actor-critic** (Dreamer-style: the actor
trains by policy gradient on the world model's imagined rollouts — on-policy by
construction, NO teacher, so it fixes the stochastic-target, the compounding-error,
AND the slow-planner-in-loop problems at once). The BC negative pushed us toward the
approach we needed for Crafter anyway. Exp 0017 = imagination actor-critic.

## Links

## Links

[[hierarchy-and-credit]] · [[capability-map]] · [[imagination-training]] ·
[[dreamerv3-2023]] · [[0012-utd-and-targeted-resets]]
