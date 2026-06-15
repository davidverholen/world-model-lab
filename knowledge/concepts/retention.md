---
status: draft
owner: world-model
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
| **frozen encoder** (earn it, then pin it) | DINO-WM / DINOv3 line (pretrained variant); Ma 2024 (frozen pretrained encoders suffice) | **STABILITY SOLVED (0013): encoder freeze @r2 stops all crashes, 6/6 seeds; GRU must stay plastic.** Interference = encoder drift. 0014 mapped the freeze-timing frontier (40/47/53/60/63% across ftrunk/fenc@2/@3/@4/free); 0015: also freezing the GRU caps at the trunk ceiling (30-37%) — **stability↔performance is a genuine frontier at this budget**, not out-tunable |

## Open questions

- ~~Does the reset recipe transfer from model-free TD agents to our model-based
  MC-value setup?~~ → **ANSWERED (0011/0012):** no — both Nikishin and Qiao recipes
  assume high-UTD *overfitting*; our regime was *under*-trained (UTD ×4, not resets,
  was the win). See [[qiao-model-primacy-2023]].
- ~~Is the encoder or the heads the locus of interference here?~~ → **ANSWERED (0013):**
  the encoder. Freezing it @r2 stops all crashes; the GRU must stay plastic.
- **Still open:** the stability↔performance ceiling — crash-free freeze configs cap
  below the UTD-×4 peak (rung-2b was MET at 0012 anyway: 63% vs PPO 37%). Untested
  levers: churn/NTK regularization, continual backprop, and [[dreaming]]'s
  latent-dream-augmentation (a cheap candidate retention regularizer).

## Links

[[nikishin-primacy-2022]] · [[qiao-model-primacy-2023]] · [[ma-plasticity-2024]] ·
[[0009-ignition-mechanics]] · [[0010-retention-mechanics]] · [[0011-nikishin-resets]] ·
[[0012-utd-and-targeted-resets]] · [[0013-trunk-freeze]] · [[0014-freeze-round-sweep]] ·
[[0015-staged-freeze]] · [[agent-architecture]] · [[latent-collapse]] (the *other*
representation pathology — collapse is too little change, interference is too much)
