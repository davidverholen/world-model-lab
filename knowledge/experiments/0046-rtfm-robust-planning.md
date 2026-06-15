---
status: planned
owner: world-model
scope: local
verified: false
last_reviewed: 2026-06-15
---

# 0046: robust-planning lever — sampled rollouts vs the reward head's OOD false positives

**Status: PRE-REGISTERED (not yet run).** The concrete next experiment after exp0045. Fresh-session
pick-up: read [[0044-rtfm-mpc-execution]] + [[0045-rtfm-oracle-probe]] + this page, then implement v1
below and dispatch.

## Hypothesis

exp0045 localized the length-2 execution wall: the WM reads AND values the true gesture (oracle
percentile ≈ 0.88, return 5–6× random) — but it is NOT the argmax, because the reward head OVERRATES
~12% of out-of-distribution action sequences, and naive MPC (which maximizes predicted reward over a
**prior-mean** rollout) chases those false positives. This is exp0017–0025 imagination-exploitation at
the planning layer. The prior-mean rollout gives a single OPTIMISTIC point estimate per candidate, so
OOD sequences whose mean is overrated win.

**v1 (do this first — minimal, directly targets the prior-mean optimism): SAMPLED-rollout MPC.**
Replace the prior-mean rollout in `agents/rtfm_mpc.py::_rollout_returns` with **K sampled rollouts**
(z ~ prior `.rsample()`, K≈10) per candidate, and score by the **mean** return across samples.
Averaging penalizes optimistic OOD point-estimates (high-variance/unreliable predictions regress
toward the random baseline), so the true gesture — which the WM values *consistently* — should rise
toward the argmax. Predict: **oracle pct → ~1.0 AND MPC `correct` lifts off the ~0.05 floor**, while
`swapped ≪ correct` holds on held-out (anti-baking preserved).

**Counter-outcome / escalation.** If sampled-averaging does NOT raise the oracle pct, the
overestimation is not merely prior-mean optimism → escalate: (b) **pessimism** — penalize each step's
predicted reward by the reward head's two-hot distributional spread (uncertainty); (c) **continue-head
gating** — multiply predicted reward by predicted continue prob (down-weight off-distribution/terminal
states); (d) **reward-head OOD regularization** at WM-train time so the gesture becomes ~argmax. The
oracle probe is the readout for each: a variant works iff it raises oracle pct toward 1.

## Setup (implementation + dispatch)

- **Implement:** add a `n_rollout_samples` (K) arg to `_rollout_returns` / `plan_action` / the probe;
  sample z per step instead of `.mean`, average returns over K. Keep K=1 = current behavior (so the
  flag is additive). Re-run the SAME A/B harness: `--mpc-eval --oracle-probe`.
- **Dispatch** (curriculum+aux WM, as exp0044/0045):
  `scripts/dispatch_rtfm.sh exp0046 4 --rounds 30 --curriculum-rounds 10 --episodes-per-round 60
  --updates-per-round 400 --ac-updates-per-round 400 --seq-batch 16 --window 12 --burn-in 4
  --horizon 10 --n-train-seeds 400 --n-eval-seeds 20 --max-steps 48 --length 2 --one-shot
  --reading-shaping-coef 1.0 --manual-aux-coef 1.0 --mpc-eval --oracle-probe` (+ the new K flag).
- **Smoke + commit** the planner change (CPU test in `tests/test_rtfm_mpc.py`), then dispatch.

## Standing bar (success)

Oracle pct → ~1.0 AND MPC `correct` clearly above the reactive ~0.05 floor at length-2, with
`swapped ≪ correct` on held-out (the anti-baking gate). That validates the design's execution pillar
([[hierarchical-imagination-agent]] §5,§7) and unblocks the next rung (abstraction on a Crafter
state-chain). If only partial, compose v1 + (b)/(c) before concluding.

## Links

[[0045-rtfm-oracle-probe]] · [[0044-rtfm-mpc-execution]] · [[hierarchical-imagination-agent]] · [[imagination-training]] · [[0021-critic-on-replay]]
