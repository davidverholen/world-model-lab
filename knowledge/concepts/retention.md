---
status: draft
owner: agent
scope: local
sources: [arxiv:2205.07802, arxiv:2411.04832]
verified: true
last_reviewed: 2026-06-12
---

# Retention (Plasticity Loss / Primacy Bias / Interference)

## What it is

The cluster of failure modes where a network in a *continual* training loop loses
previously acquired competence or the ability to acquire new competence:

- **catastrophic interference**: new updates overwrite old function;
- **primacy bias**: overfitting to early data blocks exploitation of later data;
- **plasticity loss**: the network gradually loses trainability altogether
  (NTK-rank collapse / "churn" view; dormant units).

## Why it matters here

The flywheel retrains on a shifting data distribution every round — continual
learning by construction. Our cleanest evidence (exps 0009/0010): 6 success examples
→ 60% greedy after round 0, then continued training *deterministically* destroys it
(all four 0010 arms, same seed, same crash). Verified NOT fixable by update
smoothing: lr decay ×0.3 and EMA acting weights both failed (0010, 12 runs).

## Fix families (literature status → our status)

| family | literature | ours |
|---|---|---|
| smoothing (lr decay, EMA) | not the recommended fix | **ruled out (0010)** |
| **resets** (reinit late layers, keep replay) | Nikishin 2022, robust across SAC/SPR/DrQ; scales with replay ratio | **DEAD in our regime** (5 variants, exps 0011/0012, all ≤ ctrl): both recipes assume high-UTD overfitting; our failure needed more fitting, not forgetting |
| **UTD scaling** (more gradient steps, same env budget) | Qiao Fig-3 (agent-UTD helps MBPO) | **BIG WIN (0012): ×4 → 63% mean, beats PPO; 80% peak.** Crashes persist at higher amplitude → not the full fix |
| churn reduction / NTK regularization | ICML 2025 (queued: 2506.00592) | untested |
| continual backprop (selective reinit of dormant units) | Sutton lab (queued: 2306.13812) | untested |
| frozen pretrained trunk (no plasticity needed) | DINO-WM / DINOv3 line | rung-3 candidate (immune by construction) |

## Open questions

- Does the reset recipe transfer from model-free TD agents to our model-based
  MC-value setup? (exp 0011 answers)
- Is the encoder or the heads the locus of interference here? (0011's heads-vs-deep
  arms answer indirectly; a frozen-encoder arm would answer directly)

## Links

[[nikishin-primacy-2022]] · [[0009-ignition-mechanics]] · [[0010-retention-mechanics]] ·
[[0011-nikishin-resets]] · [[agent-architecture]] · [[latent-collapse]] (the *other*
representation pathology — collapse is too little change, interference is too much)
