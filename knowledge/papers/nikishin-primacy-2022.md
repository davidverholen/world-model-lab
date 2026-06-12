---
status: draft
owner: agent
scope: local
sources: [arxiv:2205.07802]
verified: true
last_reviewed: 2026-06-12
---

# The Primacy Bias in Deep Reinforcement Learning (Nikishin et al., 2022)

**Lab:** Mila / Université de Montréal · **Code:** github.com/evgenii-nikishin/rl_with_resets
· **Read state:** skimmed (method + protocol sections fetched 2026-06-12)

## One-paragraph summary

Deep RL agents overfit their earliest experience ("primacy bias"), which then
*prevents* them from exploiting later, better data — heavier early training makes it
worse, and the damage persists even as fresh data arrives. The fix is brutally
simple: **periodically re-initialize the late layers of the networks while keeping
the replay buffer**, then let the agent relearn from replay. Recovery is fast
because the buffer acts as a non-parametric world model.

## Key protocol details (implementation-grade)

- Reset depth: last 1–3 layers normally; deeper for pixel inputs; full network for
  dense-state SAC. Early conv/representation layers usually kept.
- Frequency: 3–10 resets over training; even one helps.
- Optimizer moments reset alongside (ablation: barely matters — recovers in ~10–1000
  steps). Replay buffer NEVER reset — "keeping the buffer is essential".
- **Benefit scales with replay ratio** (updates per env step): at ratio 32, SAC
  +100%; at extreme ratios resets enable training where baselines fail entirely.
  At low ratios, little or no benefit.
- Brief performance dip right after each reset; recovery within thousands of steps.
- L2/dropout do NOT substitute — the issue is early-data overfitting specifically.

## Relevance to our experiments

Direct blueprint for exp 0011 (reset arms on DoorKey-6x6). Our observed failure
(60% after round 0 → destroyed by training on later rounds, identical across all
exp-0010 arms) matches their description of primacy bias almost verbatim — except
inverted in time: *our* networks overfit round-0 random data and can't digest the
better flywheel data. Caveats for us: our replay ratio (~0.1) is far below the
regime where their gains are largest → exp 0011 arm (c) doubles updates; and our
MC-return value targets differ from their TD setups.

## Limitations / critiques

Results are model-free (SAC/SPR/DrQ); transfer to model-based latent world models
(our case) is the open question exp 0011 tests. Reset frequency is a tunable.

## Links

[[0011-nikishin-resets]] · [[0010-retention-mechanics]] · [[retention]] ·
[[agent-architecture]]
