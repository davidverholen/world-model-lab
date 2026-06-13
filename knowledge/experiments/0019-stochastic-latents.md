---
status: draft
owner: human
scope: local
sources: [arxiv:2301.04104, arxiv:1811.04551]
verified: true
last_reviewed: 2026-06-13
---

# 0019: Stochastic latents (RSSM) — the exploitation fix [Dave: invest now]

## Hypothesis

Exp 0017/0018 showed a DETERMINISTIC world model is maximally exploitable: the
imagination actor-critic drives the model to a single fake-reward state (imagined
return 2-5 vs real max 1; eval 0). Making the latent STOCHASTIC (RSSM, PlaNet/Dreamer
lineage) fixes this: imagination samples z from a prior p(z|h) trained (via KL) to
match the posterior q(z|h,obs) over real data, so imagined rollouts stay near the
real distribution and there is no single exploitable trajectory. Expect:
imagined_return ≤ ~1.0 AND eval rises off 0, approaching the planner (~25-35%).
Dave chose this over the DAgger shortcut ("invest now") — RSSM is needed for Crafter
regardless. Build incrementally (module → train loop → agents → smoke → dispatch),
not big-bang, given it's the largest architecture change in the project.

## Setup

Gaussian RSSM (models/rssm.py): state=(h deterministic GRU, z stochastic);
belief=concat(h,z) for the heads. WM loss = recon(belief→embed) + KL(post||prior)
with free bits + KL balancing (DreamerV3 β_dyn 0.5 / β_rep 0.1); encoder+SIGReg kept
for embed anti-collapse. Imagination samples z~prior. Reuses continue predictor
(0018), actor-critic (0017). DoorKey-6x6, 3 seeds, desktop.

## Result

**Partial success — eval rose off zero for the first time on the imagination line.**
3 seeds, DoorKey-6x6, commit b8fdb62 (run 2026-06-13, ~3 h wall, 3 seeds sharing the
5070 Ti). Command per seed:
`OMP_NUM_THREADS=2 python -m world_model.train_rssm --env-id MiniGrid-DoorKey-6x6-v0
--rounds 7 --round0-steps 20000 --actor-steps 15000 --updates-per-round 4000
--ac-updates-per-round 4000 --success-frac 0.25 --ignition-events 5 --freeze-round 2
--eval-episodes 20 --save runs/dk6_rssm.pt --seed {0,1,2}`.

Per-round `eval_success` (20 eps, greedy):

| round | s0 | s1 | s2 |
|---|---|---|---|
| 0–2 | 0 | 0 | 0 |
| 3 | 0 | 0.15 | 0 |
| 4 | 0 | 0.20 | 0 |
| 5 | 0 | **0.55** | **0.35** |
| 6 | 0 | 0.00 | 0.20 |

best_eval: s0 **0.00**, s1 **0.55**, s2 **0.35** (both winners peak at round 5).
For comparison, exps 0017/0018 sat at ~0 eval here. The s1 checkpoint reproduces
**10/20** under `world_model.play` (reactive RSSM agent, added this exp) on independent
seeds — competence is real and watchable, not a logging artifact. Behaviour is
**bimodal**: wins are near-optimal (19–36 steps: key→door→goal executed cleanly),
losses are full timeouts (360 steps) — brittle to initial layout, not random flailing.

**Prediction partially refuted.** The hypothesis predicted `imagined_return ≤ ~1.0`.
It did NOT hold: imagined_return ran **0.7–4.3** across the full run — the *same*
inflated range as the exploited deterministic models (0017/0018: 2–5). The smoke test's
`imagined_return ~0.1` did not survive to the full-length run.

## Lesson

**RSSM fixed policy *quality*, not value *calibration* — and that dissociation is the
result.** In 0017/0018, imagined_return 2–5 came with eval 0 (pure hallucination). Here
the *same* inflation coexists with eval 0.55. Mechanism: stochastic latents stop the
actor collapsing onto one deterministic fake-reward trajectory — it must score well
across *sampled* futures, so the **relative ordering** of actions tracks reality (→ a
useful policy) even while the **absolute value scale** stays miscalibrated (→ inflated
imagined_return). Stochasticity declaws exploitation; it does not bound it. So the
"invest now" was justified — RSSM is what made the imagination line produce real
competence — but it is a *partial* fix, and value calibration is now the named open
problem (DAgger grounding / KL / horizon — deferred fork b).

**Two follow-on problems, both visible in the table:**
- **Peak-then-collapse** (round 5→6 on both winners: 0.55→0.00, 0.35→0.20) — the
  retention/plasticity signature from the 0009–0013 thread; `freeze-round 2` alone did
  not prevent it. This is the cheapest, most likely-to-convert lever → **fork (a)**.
- **Seed variance** (s0 never ignited) — 1/3 → 3/3 ignition is the bar.

**Process lesson:** a smoke test confirms *mechanism wiring*, NOT steady-state
behaviour. Claiming `imagined_return` was "fixed" from the smoke was premature — the
inflation re-emerges only after enough AC updates (4 k × 7) for the actor to re-find it.
Don't promote a smoke-test metric value to a finding.

## Links

[[0017-imagination-actor-critic]] · [[0018-continue-predictor]] · [[dreamerv3-2023]] ·
[[imagination-training]] · [[generative-vs-predictive]]
