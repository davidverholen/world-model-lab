---
status: draft
owner: world-model
scope: local
sources: [arxiv:2301.04104]
verified: true
last_reviewed: 2026-06-12
---

# DreamerV3: Mastering Diverse Domains through World Models (Hafner et al., 2023)

**Lab:** DeepMind / U Toronto · **Code:** https://github.com/danijar/dreamerv3 · **Read state:** skimmed (method sections via full text, 2026-06-12)

## One-paragraph summary

RSSM world model + actor-critic trained on imagined rollouts, with robustness tricks
(symlog predictions, two-hot return targets, KL balancing, free bits) that let a single
configuration work across 150+ tasks — first agent to collect Minecraft diamonds from
scratch without human data. Later published in Nature (2025). The reference
general-purpose world-model RL algorithm and our designated baseline.

## Key ideas (implementation-grade, fetched 2026-06-12)

- **RSSM**: 32 categorical latents × 32 classes + GRU deterministic state (256–4096
  units by size); model state = concat(h, z). Latents are *stochastic discrete* —
  vs our deterministic GRU belief.
- **Symlog**: `sign(x)·ln(|x|+1)` on encoder inputs, reconstruction, reward, and
  value targets — scale robustness across domains without per-env tuning.
- **Two-hot distributional critic**: K=255 buckets over symlog[-20,20]; soft labels.
- **KL balancing + free bits**: β_pred 1, β_dyn 0.5, β_rep 0.1; KL clipped below
  1 nat (free bits) — prevents both collapse and over-regularization. Unimix 1%.
- **Imagination actor-critic**: horizon 15, λ=0.95, γ=0.997, entropy 3e-4 fixed;
  return normalization by inter-percentile range Per(95)−Per(5), only scale-down.
- **Critic-on-replay (anti-overoptimism)**: the *actor* trains on imagined rollouts
  only, but the *critic* trains on BOTH imagined (β_val 1) AND real replay trajectories
  (β_repval 0.3) — grounding the value estimate in real returns so the policy cannot
  chase a purely hallucinated value. This is DreamerV3's direct answer to model
  exploitation; reused in our exp 0021 (the cheap "lite DAgger"). [[0021-critic-on-replay]]
- **Stability kit relevant to our [[retention]] thread**: critic EMA *target*
  (decay 0.98, regularize critic toward it — NOT acting-EMA like our failed 0010
  arm!), return-scale EMA 0.99, grad clip, no schedules/decay/dropout anywhere.
- Sizes XS(8M)→XL(200M); training ratio 0.5–16 replayed steps per env step.

## Relevance to our experiments

Ladder rungs 2–4 baseline and a parts bin: symlog + two-hot for our reward/value
heads (would replace pos_weight hacks more principledly), free-bits for SIGReg
balance, and — directly on-thread — their critic-EMA-as-*target* (vs our
EMA-as-actor that failed in 0010) is another untested retention mechanism: it
regularizes the critic toward its own slow copy instead of replacing the actor.
Note their training ratio (up to 16) vs ours (~0.1) — consistent with the Nikishin
replay-ratio finding; our exp-0011 arm (c) probes this.

## Links

[[imagination-training]] · [[dreamer4-2025]] · [[environment-ladder]]
