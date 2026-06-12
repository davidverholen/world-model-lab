---
status: draft
owner: human
scope: local
sources: [arxiv:2511.08544, arxiv:1811.04551]
verified: true
last_reviewed: 2026-06-12
---

# 0004: First acting agent — latent MPC plays Empty-8x8 (run 2026-06-12, 19/20)

## Hypothesis

A planner needs nothing but our learned world model: with a reward head trained on
sparse random-play rewards (upweighted 10×), MPC by random shooting in latent space
(512 candidates, horizon 12, γ=0.98, re-planned every step) reaches the goal in
MiniGrid-Empty-8x8 in ≥80% of 20 evaluation episodes, versus a random-agent baseline
well below 50%. All planning happens in imagination — pixels are only seen once per
real step.

## Setup

Train: `uv run python -m world_model.collect --env-id MiniGrid-Empty-8x8-v0
--sigreg-weight 0.05 --steps 20000 --updates 3000 --seed 0 --save runs/empty8x8_wm.pt`
(encoder + predictor + reward head jointly, SIGReg λ=0.05, batch 128).
Eval: `uv run python -m world_model.play --checkpoint runs/empty8x8_wm.pt
--episodes 20 --record /tmp/eval.gif` vs the same command without checkpoint.
Episode seeds 0–19 (fixed layout env; seeds vary agent start).

## Result

Three iterations were needed (all 2026-06-12, CUDA/RTX 4070; random baseline: 3/20):

| version | training | planner | success |
|---|---|---|---|
| v1 | 1-step pred, pos_weight 10, 20k steps | random shooting 512×H12 | 6/20 |
| v2 | + 8-step rollout training | CEM 3×512×H16 | **0/20** |
| v3 | + 12-step rollouts, pos_weight 100, 30k steps | CEM 3×512×H16 | 12/20 |
| v3 | same checkpoint | CEM 3×**1024**×**H20** | **19/20** |

Diagnostics that drove each fix (details in LOG and the play.py trace):

- v1 failure: reward head ranked correctly near the goal (+0.128 forward vs +0.005
  turns) but the *known-optimal* 11-step sequence scored **negative** in imagination
  (−0.023, below random max +0.07) → compounding open-loop rollout error; predictor
  had only ever been trained on 1-step transitions.
- v2 failure: drift fixed enough to make returns monotone in distance-to-goal
  (−0.001@11 → +0.027@2), but absolute differences (~0.003) sat below planner sampling
  noise at spawn distance — the agent oscillated left/right/toggle at (1,1) forever,
  0/20. A *systematically* wrong-confident planner is worse than a noisy one (v1's
  6/20 was luck-adjacent wandering).
- v3 fix: reward head magnitude was the bottleneck (predicted 0.018 for a true 0.96
  goal reward; pos_weight 10 couldn't counter 99.7% zero-reward transitions, doubled
  by drifted-latent reward training). pos_weight 100 + more goal events (30k steps,
  ~115 episodes) restored signal; stronger planning at eval (free) did the rest.

Final model: dynamics ratio 0.172, latent_std 0.83, no collapse. Mean successful
episode ≈ 50–65 steps (optimal ~11; navigation works, efficiency has headroom).

**Correction (2026-06-12, from exp 0006's eval-hygiene incident):** all eval rows
above were measured with a 2× observation-scale mismatch (play.py tile 16 vs
training tile 8). Re-eval under matched observations: **20/20**. The recorded
numbers understate matched performance; relative conclusions unaffected.

## Lesson

1. **The world model plays its first game**: encoder + latent predictor + reward
   head + CEM planner, all learned from random play, no policy network, no
   environment access at planning time. Roadmap "first acting agent" done.
2. **Multi-step rollout training is non-optional** for planning (compounding error
   makes 1-step models unusable beyond a few imagined steps) — matches why
   PlaNet/Dreamer/TD-MPC all do it.
3. **Sparse-reward magnitude calibration matters as much as ranking**: planner signal
   must beat sampling noise at the *distances where decisions happen*, not just
   adjacent to reward. Watch predicted-vs-true reward magnitude as a standard
   diagnostic.
4. Planner compute at eval is the cheapest win available (12/20 → 19/20 for free).
5. Negative probe R² (−0.27) on a model that demonstrably navigates confirms exp
   0003's demotion of the linear probe.

## Links

[[0003-probe-protocol-8x8]] · [[value-equivalent-planning]] · [[environment-ladder]] ·
[[latent-collapse]]
