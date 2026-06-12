---
status: draft
owner: human
scope: local
sources: [arxiv:2205.07802]
verified: true
last_reviewed: 2026-06-12
---

# 0011: Primacy-bias resets on DoorKey-6x6 (planned)

## Hypothesis

Per Nikishin et al. (arXiv:2205.07802; protocol fetched 2026-06-12): periodically
re-initializing late layers while keeping the replay buffer defeats primacy bias,
with fast recovery because replay acts as a non-parametric world model. In our
round structure (reset at rounds 1–6 = 6 resets, within their 3–10 guidance,
optimizer rebuilt, replay + success-oversampling kept): at least one reset arm
shows monotone-ish improvement (no eval more than 20pp below running best after
round 1) and mean best-eval ≥ 55% across 3 seeds — where smoothing (exp 0010)
failed. Their caveat that benefits scale with replay ratio motivates arm (c).

Arms (vs exp 0010 ctrl: 20/60/40, mean 40%):
(a) reset-heads — reward, value, next-latent MLP;
(b) reset-deep — heads + GRU cell + action embed (encoder kept; deeper resets for
    pixel inputs per the paper);
(c) reset-heads-hr — heads + double replay ratio (updates 1500→3000/round; legal:
    budget axis is env steps).

## Setup

3 arms × 3 seeds, same protocol/budget as 0009/0010 (`--rounds 7 --round0-steps
20000 --mpc-steps 15000 --success-frac 0.25 --ignition-events 5 --reset
{heads|deep}`, arm c adds `--updates-per-round 3000`). uint8 replay storage shipped
first (lossless; exact-roundtrip test) — 6-wide batches now RAM-safe (~2.5 GB/run).
Batch 1: a+b (6-wide); batch 2: c (3-wide). Comparator ctrl reused from exp 0010
(same code path; uint8 storage is numerically lossless).

## Result

_pending_

## Lesson

_pending_

## Links

[[0010-retention-mechanics]] · [[0009-ignition-mechanics]] · [[agent-architecture]]
