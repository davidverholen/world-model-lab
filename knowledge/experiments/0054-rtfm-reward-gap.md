---
status: draft
owner: world-model
scope: local
verified: false
last_reviewed: 2026-06-16
---

# 0054: reward-gap probe — is the downstream wall the reward READOUT or credit-assignment?

**Status: PRE-REGISTERED + IN FLIGHT** (`runs/exp0054`, 2 seeds, training). Follows the
[[0053-rtfm-imagination-fidelity]] finding: the WM imagines *faithfully*, so the ~0.10 length-2
swap_follow wall is **downstream** of the world model — in the policy / reward-readout. This run
decides *which*.

## Why this run exists (and why it's a fresh train)

exp0053 measured open-loop **belief** fidelity (needs only rssm + encoder) and found it good. The
*direct* test of "the actor optimises a distorted objective" is the **imagined-vs-real reward gap** —
does the task reward head over-predict reward on the OOD states the actor reaches in imagination? That
needs the `rew` head in the checkpoint, which the old save format dropped. The save format now persists
`recon_head`/`rew`/`cont`/`vr_head`, so exp0054 simply **re-runs the working baseline to get a
checkpoint with the reward head saved** — then `scripts/imagine_report.py` reports the reward gap.

## Hypothesis / the fork it decides

Run the confirmed ~0.10 working baseline (folded validated-reading, coef 0.3). Then probe the reward
gap on length-2:

- **Reward head OVER-OPTIMISTIC** (imagined cumulative reward ≫ real): the actor chases phantom reward
  even atop a faithful world model → the wall is the **reward readout** (exp0045 over-rating / exp0052
  VR-head reward-hacking, on the *task* channel). Fix = conservatism on the reward head
  ([[0047-rtfm-conservative-reward]] CROP/CQL), now well-motivated rather than speculative.
- **Reward head FAITHFUL** (imagined ≈ real reward): the readout is fine too → the wall is **pure
  credit-assignment / execution** of the 2-step gesture ([[0043-rtfm-execution-wall]]) — a reactive
  actor can't sequence it regardless of a correct model+reward. Fix lives in planning / horizon /
  a different action-selection tool, not another reward term.

**Predict:** over-optimism (the exp0045 ghost recurs on the task channel). Either way the result picks
the next lever unambiguously — that is the point.

## Setup

- `scripts/dispatch_rtfm.sh exp0054 2 --rounds 30 --curriculum-rounds 10 --length 2
  --manual-aux-coef 1.0 --validated-reading-coef 0.3` (the confirmed ~0.10 baseline config; new save
  format persists the WM heads).
- Harvest: confirm it lands at ~0.10 length-2 swap_follow (sanity vs [[0051-rtfm-flat-vr-optimization]]),
  then `imagine_report.py <ckpt> --mode CORRECT/SWAPPED` → read the "imagined-minus-real reward (sum
  over H)" line + belief/h fidelity (should reproduce exp0053).

## Standing bar — diagnostic (tag: `EXTRA-RIGOR`)

This run SUCCEEDS by producing a clear reward-gap number that selects the branch above; it is a
localization probe, not a swap_follow push. (Reaching ~0.10 just confirms it's the same baseline.)

## Links

[[0053-rtfm-imagination-fidelity]] · [[0045-rtfm-oracle-probe]] · [[0047-rtfm-conservative-reward]] · [[0043-rtfm-execution-wall]] · [[0051-rtfm-flat-vr-optimization]] · [[validated-reading-reward]]
