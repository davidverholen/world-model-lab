---
status: draft
owner: world-model
scope: local
sources: [arxiv:2508.10104]
verified: true
last_reviewed: 2026-06-14
---

# 0029: Is the frozen DINO encoder blind to the agent's own stats? (pre-registered)

## Hypothesis

Watching the 0027 agent (crafter_steps s1) play, we observed the **drink-at-cap
spam** survives: at water it keeps issuing `do` (=collect_drink) even at the drink cap (9),
a no-op. The two-hot reward head (0025) reduced but did not kill this. Proposed mechanism:
the agent is **perceptually blind to its own vitals** — Crafter renders health/food/drink/
energy as small HUD stat-bars in the frame, but a **frozen DINO encoder** (natural-image SSL,
then pooled) likely washes those icons out, so "thirsty at water" and "full at water" map to
near-identical latents and the actor can't learn to stop. This would also feed the depth wall
(farm legible shallow reward instead of exploring) and — if true — challenges the
frozen-encoder lean ([[frozen-encoder-lean]]): the encoder we assumed would *help* may be
*blocking* a class of state the agent needs.

Test: **linear-probe decodability**. If a linear map from the frozen embedding can recover the
true stats with high R², the info IS present (the actor's fault, fixable AC-side). If the
vitals are NOT linearly decodable (R² low, near the shuffled-control floor), the encoder drops
them → architectural signal (need a stat-aware perception path; do NOT hardcode the cap).

Predict: the **vitals** (esp. `drink`) probe **low** (HUD washed
out), while spatial/material structure the encoder was built for is fine. Patch pooling
(cls+patch) may recover more than cls alone if the HUD survives in patch tokens.

## Setup

Collect (frame → true `info['inventory']`) pairs from a mix of random episodes (vital
variance: food/drink/energy deplete over a rollout) + the trained crafter_steps_s1 agent
(material/tool variance). Encode each frame with `FrozenDinoEncoder` for pool ∈ {cls,
patch_mean, cls+patch}. Ridge-regression probe per target with a train/test split; report
test R² per stat. Control: a **shuffled-embedding** probe (break the frame↔stat
correspondence) to calibrate the chance floor. Targets: the 4 vitals + wood + stone.

Command: `python -m world_model.stat_probe --frames 5000 --agent runs/crafter_steps_s1_s1.pt`
(local — no desktop GPU; 0028 keeps running there). One variable of interest per stat: test R².

## Result

**Hypothesis REFUTED — the encoder is not blind, and the RSSM keeps drink. The drink-spam is
an actor/reward problem, not perception.** 4000 frames (random + crafter_steps_s1), ridge
probe, 80/20 split, shuffled control at the 0.03–0.09 chance floor.

| repr | health | food | drink | energy | wood | stone* |
|---|---|---|---|---|---|---|
| emb:cls | 0.66 | 0.91 | 0.93 | 0.76 | 0.88 | — |
| emb:patch_mean | 0.70 | 0.90 | 0.92 | 0.76 | 0.90 | — |
| **emb:cls+patch** (our pool) | **0.79** | **0.94** | **0.95** | **0.82** | **0.93** | — |
| **belief(rssm)** (what the actor sees) | 0.35 | 0.89 | **0.94** | 0.68 | 0.71 | — |

*stone has zero variance (random/early play never mines it) → R²=1.0 degenerate, ignored.

Two findings: (1) the frozen DINO embedding linearly carries the vitals — `drink` at **0.95**
(cls+patch); the HUD bars survive the encoder. (2) the trained RSSM **belief retains drink at
0.94** — near-zero loss from embedding to the state the actor consumes. So the agent has the
drink level available end-to-end and still spams collect_drink at the cap. Side finding:
**health** drops hard in the belief (0.79→0.35) — the RSSM compresses away the slow-moving
health stat (a real WM blind spot, but not the drink-spam cause).

## Lesson

**The drink-spam (and by extension the "farm legible shallow reward" tendency) is NOT a
perception failure — the frozen encoder AND the RSSM both preserve the drink level the actor
needs. It is an actor/reward/credit-assignment problem:** drinking at the cap is a zero-reward
*no-op with no penalty*, so nothing pushes the policy to stop. This **vindicates the frozen
encoder** ([[frozen-encoder-lean]] holds — it is not the blocker the watching-session feared)
and converges with the depth thread: 0026/0027 ruled out compute → actor/exploration-bound;
this rules out perception → actor/reward-bound. **Two independent threads now point at the same
locus: the policy/credit side, not the world model or perception.** Strategic implication:
keep the frozen encoder, stop spending on perception/WM capacity, invest the next effort
actor-side (structured exploration / hierarchy / incentive shaping). Also sharpens the 0028
prediction: Curious Replay is a WM-side lever, so it is likely NULL for *depth* if this holds.
Per [[no-hardcoded-env]] the fix is never "tell the model the cap is 9" — the cap is already
perceivable; the missing thing is an incentive/credit path that uses it.

## Links

[[0027-step-scaling]] · [[0025-reward-head-patch]] · [[frozen-encoder-lean]] · [[crafter]] · [[no-hardcoded-env]]
