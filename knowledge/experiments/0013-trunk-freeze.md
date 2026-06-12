---
status: draft
owner: human
scope: local
sources: [arxiv:2310.15017]
verified: true
last_reviewed: 2026-06-12
---

# 0013: Trunk freeze at high UTD — protect, don't forget (planned)

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

_pending_

## Lesson

_pending_

## Links

[[0012-utd-and-targeted-resets]] · [[retention]] · [[qiao-model-primacy-2023]] ·
[[agent-architecture]]
