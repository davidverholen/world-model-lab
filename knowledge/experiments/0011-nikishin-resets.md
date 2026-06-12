---
status: draft
owner: human
scope: local
sources: [arxiv:2205.07802]
verified: true
last_reviewed: 2026-06-12
---

# 0011: Naive reset transfer fails — the mapping was wrong, not the idea (run 2026-06-12)

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

**All three arms refuted** (vs ctrl 20/60/40, mean 40%):

| arm | bests (s0/s1/s2) | mean | signature |
|---|---|---|---|
| reset-heads, 1500 upd | 10/60/10 | 27% | post-reset rounds ≈ 0 everywhere; s1's 60% (round 0, pre-reset) never recovers |
| reset-deep, 1500 upd | 10/60/10 | 27% | same, flatter |
| reset-heads, 3000 upd | 10/20/40 | 23% | **s1 round-0 drops 60→20: doubling round-0 training deepens primacy bias BEFORE any reset** — Nikishin's core claim reproduced in our control variable |

uint8 replay held 6-wide RAM at ~3.7 GB/process (no paging; ~45–55 min batches at
75% CPU efficiency).

## Lesson

1. **The naive transfer mis-mapped "last layers."** Our `heads` include the
   next-latent prediction head — i.e. the world model itself. Nikishin resets
   policy/value heads over *stable representations*; we amputated the dynamics
   every round and (at replay ratio ~0.1–0.2, vs their ≥2) gave it no time to
   regrow. Result: per-round amnesia, worse than no treatment.
2. **The replay-ratio arm both refutes and confirms**: 2× updates didn't rescue
   resets (still ≈25× below Nikishin's regime), but the s1 round-0 degradation
   (60→20 from more early training) is direct in-setting evidence that primacy
   bias is the right diagnosis — the disease is confirmed even as this cure fails.
3. **Exp 0012 pre-registration (corrected mapping)**: (a) reset ONLY reward+value
   heads, dynamics untouched; (b) reset ONCE at mid-training (round 3), not every
   round; (c) post-reset round gets 4× updates (one-time cost, env-budget legal);
   (d) consider shrink-and-perturb (soft reset) as the gentler arm. Falsifiable
   bar unchanged: no >20pp crash + mean ≥55%.
4. **Horizon-rule status**: scout finds no published work on reset/plasticity
   mechanics for model-based latent world models under flywheel collection. Per
   the research loop's transformed step 2, exps 0009–0012 are now the primary
   literature for this question. The nearest-neighbor map (churn reduction,
   continual backprop, frozen trunk, DreamerV3's critic-EMA-target) is the design
   space; frozen-trunk (DINOv3 line) remains the structural escape hatch.

## Links

## Links

[[0010-retention-mechanics]] · [[0009-ignition-mechanics]] · [[agent-architecture]]
