---
status: draft
owner: world-model
scope: local
sources: [arxiv:1811.04551, arxiv:2301.04104]
verified: true
last_reviewed: 2026-06-12
---

# 0005: Memory under partial observability — belief-state MPC on DoorKey-5x5 (run 2026-06-12, negative)

## Hypothesis

A GRU belief-state world model (deterministic RSSM-lite: closed-loop belief update on
real latents, open-loop on imagined latents; reward from belief state) trained with
two collection rounds (60k random steps → 20k ε=0.3-MPC steps) lets belief-state CEM
planning solve MiniGrid-DoorKey-5x5 from egocentric 7×7-tile RGB at ≥40% success over
20 episodes — ≥4× the measured random baseline of 8.0% (300 episodes, seed 0–299).
Secondary: the iterative round measurably increases reward-event density vs round 0.

## Setup

`uv run python -m world_model.train_recurrent --env-id MiniGrid-DoorKey-5x5-v0
--rounds 2 --save runs/doorkey5x5_rwm.pt --seed 0`
(window 24, burn-in 8, SIGReg λ=0.05, pos_weight 100, 3000 updates/round)
Eval: `uv run python -m world_model.play --env-id MiniGrid-DoorKey-5x5-v0
--checkpoint runs/doorkey5x5_rwm.pt --episodes 20`. Baseline measured empirically:
24/300 = 8.0%, avg episode 244 steps.

## Result

Run 2026-06-12, CUDA/laptop GPU. **Hypothesis refuted: eval 0/20** (both ε=0 and ε=0.25).

But the run decomposes informatively:

- Round 0 (random, 60k steps): 246 episodes, 21 reward events (8.5% ≈ baseline).
  Training converged: pred_loss 0.45 → 0.39, latent_std ~0.73–0.82, no collapse.
- Round 1 collection **with the round-0 model** (ε=0.3 MPC, 20k steps): 84 episodes,
  12 reward events = **14.3% — 1.8× the random baseline.** The learned model + cheap
  planning already adds real value mid-pipeline.
- Round 1 training **degraded the model**: pred_loss rose monotonically 0.46 → 0.59,
  latent_std drifted 0.87 → 0.96. The post-round-1 checkpoint evaluates at 0/20,
  i.e. worse than the round-0 behavior that produced 14.3%.

## Lesson

1. **Pure receding-horizon MPC hits its structural limit here**: the DoorKey chain
   (find key → pickup → door → toggle → goal) is ~25+ actions with reward only at the
   end — beyond horizon 20, imagined returns are ~0 everywhere at episode start.
   This is precisely why the literature adds a **value function** (TD-MPC) or an
   **actor-critic trained in imagination** (Dreamer). Exp 0006: add a value head
   bootstrapped on imagined rollouts; planner score = Σ rewards + γ^H·V(s_H).
2. **Naive continued training across collection rounds destabilizes**: same optimizer
   state + shifted data distribution made prediction loss climb. For exp 0006:
   monitor per-round losses separately, consider lower lr after round 0 or
   fresh-optimizer restarts, and evaluate after *every* round (we only evaluated at
   the end and lost the better round-0 agent — checkpoint each round).
3. The 8.5% → 14.3% collection improvement is the first evidence of the
   collect→train→collect flywheel working; it failed at the *training stability*
   step, not the concept.
4. Infrastructure validated and kept: RecurrentDynamics (GRU belief), burn-in +
   open-loop sequence training, RecurrentMPCAgent, iterative collection rounds,
   play --checkpoint auto-detects recurrent models, --epsilon eval knob.

## Links

## Links

[[0004-mpc-agent-empty8x8]] · [[imagination-training]] · [[environment-ladder]] ·
[[minigrid]]
