---
status: draft
owner: human
scope: local
sources: [arxiv:2310.15017]
verified: true
last_reviewed: 2026-06-12
---

# 0013: Encoder freeze stops the crashes — interference localized (run 2026-06-12/13)

## Hypothesis

Exp 0012 left one suspect (the encoder/GRU trunk — never protected by any arm) and
one new baseline (UTD ×4: 80/50/60, mean 63%, crashes persist). If interference
lives in the representation, freezing it after it's competent should keep the
heads' gains: at UTD ×4 with the trunk frozen after round 2, **at least one freeze
arm clears BOTH bars** — mean best-eval ≥ 55% AND no eval >20pp below running best
after the freeze round. Counter-outcome that would also be decisive: freeze arms
crash anyway → interference lives in the heads' own function under shifting data,
and the remaining families are churn-reduction / continual backprop (queued).

Arms (3 seeds each, vs exp-0012 utd baseline 80/50/60):
- **fenc**: freeze encoder only @ round 2 (GRU keeps adapting);
- **ftrunk**: freeze encoder + GRU cell + action embed @ round 2 (only heads train);
- **warm**: UTD ×4 with round 0 at ×1500 (no freeze — dodges the measured round-0
  primacy cost; cheap, independent lever).

## Setup

Standard protocol (DoorKey-6x6, 7 rounds, 20k ignition, 6×15k ε-MPC,
success-frac 0.25) + `--updates-per-round 6000` everywhere;
`--freeze {encoder|trunk} --freeze-round 2` for arms a/b;
`--round0-updates 1500` for arm c. Batch 1: fenc+ftrunk 6-wide; batch 2: warm
3-wide. ~80–120 min per batch on the desktop.

## Result

(vs utd baseline 80/50/60 mean 63%, crashy; bar: mean ≥55% AND no >20pp crash)

| arm | bests (s0/s1/s2) | mean | stability |
|---|---|---|---|
| **fenc** (encoder frozen @r2) | 40/50/50 | 47% | **NO crashes, all seeds monotone-rising, ALL peak at final round — unconverged** |
| ftrunk (enc+GRU frozen @r2) | 30/50/40 | 40% | no crashes; capped (GRU plasticity was needed) |
| warm (UTD, round-0 @1500) | 30/60/40 | 43% | round-0 primacy cost dodged (s1: 60% vs 10%) but crashes return (encoder free) |

## Lesson

1. **Interference localized: the encoder.** Freeze it → the crash phenomenon
   (chased through exps 0009–0012) stops in 6/6 frozen-arm seeds; leave it free →
   crashes in every arm ever run. The GRU needs to keep learning (ftrunk < fenc);
   the heads were never the problem (exp 0011's resets were doubly misdirected).
2. **Stability bar: MET (fenc). Performance bar: missed (47% < 55%) but
   unconverged** — every fenc seed ends at its own maximum; the freeze@2 encoder
   only ever saw ignition-quality data. Hypothesis for 0014: freeze LATER
   (round 3/4, after flywheel data improves) → higher ceiling, same stability,
   same env budget.
3. **Literature tension recorded (gate run 2026-06-13):** arXiv:2310.07418
   (ICLR 2024) localizes plasticity loss in the CRITIC for model-free visual RL
   (with data augmentation, Adaptive RR) — our localization (encoder) differs;
   plausible reconciliation: our encoder is shared by the world model AND the
   value path, and we use no augmentation. Queued for ingestion with Plasticine
   (2504.17490) and Neuroplastic Expansion (2410.07994). Data augmentation as a
   plasticity preserver is an untested lever for us.
4. Exp 0014 (pre-registered, autonomous): **freeze-round sweep** — fenc@3 and
   fenc@4 vs fenc@2 (existing), same budget, 6 runs. Bar unchanged. If freeze@3/4
   clears both bars → retention SOLVED at rung 2 → propose 2b closure to Dave +
   pivot to the actor thread.

## Links

## Links

[[0012-utd-and-targeted-resets]] · [[retention]] · [[qiao-model-primacy-2023]] ·
[[agent-architecture]]
