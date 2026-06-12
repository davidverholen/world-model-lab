---
status: draft
owner: human
scope: local
sources: []
verified: true
last_reviewed: 2026-06-13
---

# 0014: Freeze-round sweep — when has the encoder earned its pension? (planned)

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

_pending_

## Lesson

_pending_

## Links

[[0013-trunk-freeze]] · [[retention]] · [[0012-utd-and-targeted-resets]]
