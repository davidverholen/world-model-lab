---
status: draft
owner: human
scope: local
sources: [arxiv:2301.04104]
verified: true
last_reviewed: 2026-06-13
---

# 0016: Planner distillation — can a reactive actor match the planner? (planned)

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

_pending_

## Lesson

_pending_

## Links

[[hierarchy-and-credit]] · [[capability-map]] · [[imagination-training]] ·
[[dreamerv3-2023]] · [[0012-utd-and-targeted-resets]]
