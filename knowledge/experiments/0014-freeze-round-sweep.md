---
status: draft
owner: human
scope: local
sources: []
verified: true
last_reviewed: 2026-06-13
---

# 0014: Freeze-round sweep — the stability-performance frontier (run 2026-06-13)

## Hypothesis

Exp 0013's fenc@2 met the stability bar but capped at 47% (unconverged) — the
encoder was pinned having seen only ignition-quality data. Freezing LATER (after
the flywheel improves the data) raises the ceiling without losing stability:
**fenc@3 or fenc@4 clears BOTH bars** (mean ≥55%, no >20pp post-freeze crash) at
the standard env budget. Counter-outcome: later freezes re-admit crashes before
the freeze lands (rounds 1–3 are where they historically begin) — then the answer
is freeze@2 + budget extension, a different trade to put to Dave.

Arms (3 seeds each, vs fenc@2 40/50/50 and utd 80/50/60):
(a) fenc@3; (b) fenc@4. UTD ×4 throughout, standard protocol.

## Setup

`--updates-per-round 6000 --freeze encoder --freeze-round {3|4}`, DoorKey-6x6,
7 rounds, standard collection. 6 runs, one 6-wide batch (~2h). [autonomous]

## Result

| arm | bests (s0/s1/s2) | mean | bar check |
|---|---|---|---|
| fenc@3 | 60/60/40 | 53% | s1 crashes 60→20 at r6 WITH encoder frozen → GRU drift implicated |
| fenc@4 | 80/50/50 | 60% | s0's 80% (r3, pre-freeze) decays to 50 around the freeze → bar (i) violated |

With 0012/0013 arms, a clean frontier emerges (stability ↔ performance):
ftrunk 40% (stable) → fenc@2 47% (stable, unconverged) → fenc@3 53% → fenc@4 60% →
free 63% (violent crashes). No single point yet clears BOTH bars.

## Lesson

1. **The trade is a frontier, not a binary**: each round of extra encoder
   plasticity buys ~5pp mean and costs stability. fenc@4 ≈ UTD's mean with far
   milder damage.
2. **Residual instability is GRU drift** (f3-s1 crashed with encoder frozen;
   ftrunk never crashed). Two plasticity taps, two pensions needed.
3. Literature gate (2026-06-13): progressive bottom-up freezing is established in
   supervised/SSL training (FreezeOut 1706.04983; LayerLock 2509.10156 — masked
   VIDEO models, JEPA-adjacent; "early layers converge first") — our staged
   variant in flywheel RL is uncovered; principle imported.
4. Exp 0015 (pre-registered): **staged pension** — encoder@2 + GRU@{4|5}, heads
   always free, UTD ×4. Both bars in one config is the explicit target.

## Links

## Links

[[0013-trunk-freeze]] · [[retention]] · [[0012-utd-and-targeted-resets]]
