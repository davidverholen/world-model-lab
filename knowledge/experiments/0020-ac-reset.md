---
status: draft
owner: human
scope: local
sources: [arxiv:2310.15017, arxiv:2205.07802]
verified: true
last_reviewed: 2026-06-13
---

# 0020: Actor-critic reset — attacking the round-6 collapse (pre-registered)

## Hypothesis

Exp 0019 (RSSM) produced real competence (s1 0.55, s2 0.35) but both winning seeds
**peaked at round 5 and collapsed at round 6** (0.55→0.00, 0.35→0.20), and s0 never
ignited. The encoder is already frozen (round 2, exp 0013 retention finding), so the
collapse originates **downstream of the encoder** — in the behaviour layer. The
actor-critic is the most primacy-prone module here: it eats 4 k updates/round on a
narrow, shifting *imagined* distribution (the Nikishin/Qiao setup for plasticity loss;
"Revisiting Plasticity" localizes loss in the critic). Periodically restoring its
plasticity should prevent the late-round collapse.

**Predict:** a per-round shrink-perturb reset of the actor+critic (α=0.5, rounds ≥1)
(i) prevents the round-6 regression — final-round eval ≈ peak eval rather than
collapsing — and (ii) lifts ignition toward 3/3 seeds. Null result (collapse persists)
would point at the value miscalibration instead (imagined_return still 0.7–4.3) → the
collapse is value-driven, escalate to fork (b) / the Qiao **world-model** reset.

## Setup

Single treatment arm vs the 0019 baseline as control (the `--ac-reset` flag defaults
off, so control ≡ 0019 exactly — no need to re-run it). At the start of each round ≥1:
`shrink_perturb(actor, 0.5)`, `shrink_perturb(critic, 0.5)`, resync `target_critic` to
the perturbed critic, rebuild `opt_ac` (fresh Adam moments). 3 seeds, DoorKey-6x6, all
other 0019 settings identical (rounds 7, 4 k WM + 4 k AC updates, freeze-round 2,
horizon 15, γ 0.98, λ 0.95). Held in reserve if null: Qiao α=0.8 reset on the RSSM
world model; halving AC updates (churn hypothesis).

Command per seed:
`python -m world_model.train_rssm --env-id MiniGrid-DoorKey-6x6-v0 --rounds 7
--round0-steps 20000 --actor-steps 15000 --updates-per-round 4000 --ac-updates-per-round
4000 --success-frac 0.25 --ignition-events 5 --freeze-round 2 --eval-episodes 20
--ac-reset --ac-reset-alpha 0.5 --save runs/dk6_acreset.pt --seed {0,1,2}`

## Result

_pending (dispatched)_

## Lesson

_pending_

## Links

[[0019-stochastic-latents]] · [[0013-encoder-freeze]] · [[qiao-model-primacy-2023]] ·
[[nikishin-primacy-2022]] · [[imagination-training]]
