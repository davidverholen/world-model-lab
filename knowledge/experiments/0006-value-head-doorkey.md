---
status: draft
owner: human
scope: local
sources: [arxiv:2310.16828, arxiv:2301.04104]
verified: true
last_reviewed: 2026-06-12
---

# 0006: Value head extends the planning horizon — DoorKey-5x5 solved (run 2026-06-12)

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

Trained on the remote 5070 Ti (first dispatched training, commit 54f29fe, ~25 min).

| round | collection (ε=0.3) | greedy eval (10 eps, seeds 10000+) |
|---|---|---|
| 0 random | 21/246 = 8.5% | 0% |
| 1 ε-MPC | 4/81 = 5% | **50%** |
| 2 ε-MPC | 35/97 = **36%** | **90%** |

Post-hoc local eval of the best checkpoint, corrected observations (see below),
20 episodes, seeds 0–19, planner 1024×3: **15/20 = 75%**. Combined greedy:
24/30 = 80% vs 8.0% random baseline (~10×). Hypothesis (≥40%) confirmed.

**Eval-hygiene incident (resolved):** the first local eval read 4/20 and triggered a
full investigation. Root cause: play.py passed `tile_size=16` to the env wrapper —
which scales the agent's *observations* (not the GIF) — while training used tile 8;
the encoder silently accepted 112×112 inputs it was never trained on. After the fix:
75%. Exp 0004 re-checked under matched observations: **20/20** (its recorded 19/20
was measured through the same mismatch and understated). Guard added: checkpoints
now store `obs_shape`; play.py refuses on mismatch.

## Lesson

1. **The value head closes DoorKey** — Σγᵗr̂ + γᴴ·V(s_H) makes the ~25-step reward
   chain visible to a 20-step planner; pure-MPC 0/20 (exp 0005) → 90%/75%.
2. **The flywheel compounds when the model is good enough to exploit**: collection
   success 8.5% → 5% → 36%; eval 0% → 50% → 90%. Per-round checkpoint+eval (0005
   lesson) made this visible and protected the best agent by construction.
3. **Observation preprocessing is part of the model contract.** Any knob that changes
   what the encoder sees (tile size, normalization, wrappers) must be pinned in the
   checkpoint and asserted at load. A 2× obs scale silently cost 60pp success.
4. First reviewer-agent pass (opus) on this milestone: verified the returns/value/
   eval-seed paths correct pre-hoc, caught the sweep-scrapability gap
   (`eval_success=` print fixed); the tile-size bug was outside its diff scope —
   found instead because two evals of the same checkpoint disagreed. Cross-checking
   the same artifact in two contexts remains the strongest bug detector we have.

## Links

[[0005-doorkey-memory]] · [[value-equivalent-planning]] · [[imagination-training]] ·
[[0004-remote-dispatch]]
