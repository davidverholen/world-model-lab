---
status: draft
owner: human
scope: local
sources: [arxiv:2310.16828, arxiv:2301.04104]
verified: true
last_reviewed: 2026-06-12
---

# 0006: Value head extends the planning horizon — DoorKey-5x5 (planned)

## Hypothesis

Exp 0005 failed because receding-horizon MPC cannot see DoorKey's end-of-chain
reward (~25+ actions) within horizon 20. Adding a value head V(belief) trained on
Monte-Carlo return-to-go from real episodes, and scoring imagined futures as
Σ γᵗ·r̂(s,a) + γᴴ·V(s_H), makes beyond-horizon reward visible. With per-round
checkpoint+eval (0005 lesson: never lose the best intermediate agent), 3 collection
rounds (60k random + 2 × 20k ε=0.3-MPC) reach **≥40% greedy success on
DoorKey-5x5** (random baseline: 8.0%). Secondary: round-over-round eval rates are
non-decreasing for the *selected* checkpoint by construction; we observe whether
the flywheel (8.5%→14.3% in 0005) compounds.

## Setup

`scripts/remote.sh run python -m world_model.train_recurrent --rounds 3
--save runs/doorkey5x5_v.pt --seed 0` — first real training dispatched to the
remote 5070 Ti (ADR 0004 pipeline). Window 24, burn-in 8, SIGReg λ=0.05,
pos_weight 100 (reward) / 20 (value), γ=0.98, per-round eval 10 episodes
(seeds 10000+, greedy). Final eval locally: 20 episodes via play.py.

## Result

_pending_

## Lesson

_pending_

## Links

[[0005-doorkey-memory]] · [[value-equivalent-planning]] · [[imagination-training]] ·
[[0004-remote-dispatch]]
