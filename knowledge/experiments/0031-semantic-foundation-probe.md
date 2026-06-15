---
status: draft
owner: world-model
scope: local
sources: [arxiv:2508.10104]
verified: true
last_reviewed: 2026-06-14
---

# 0031: Is the frozen-encoder foundation sound for the tech tree? (semantic probe)

## Hypothesis

Before pivoting to rung-4 (reading-to-learn), validate the load-bearing assumption: does the
frozen DINO encoder — and the RSSM belief the actor consumes — carry the **achievement-critical
visual structure** (stone/coal/iron/tree/water/table tiles)? 0029 proved DINO encodes the stat
HUD (drink 0.94) but ALSO that the RSSM selectively drops things (health 0.79→0.35 in the
belief). So embedding-decodability is necessary but not sufficient: the decisive test is whether
**materials survive to the belief**. If yes → foundation sound, the ~2.6%/wood plateau is
exploration/scale, build rung-4 on the frozen encoder. If no → a learned **adapter** (not a
from-scratch encoder — that abandons the transfer thesis) is the fix first.

Per [[no-hardcoded-env]] this only MEASURES what the representation carries; it hardcodes nothing.

## Setup

`world_model.semantic_probe`: 8000 frames (random + crafter_steps_s1 agent). Ground truth =
Crafter `info['semantic']` (the 64×64 tile-id map) cropped to a 9×9 egocentric view around
`info['player_pos']`; binary target per material = "present in view" (ids: stone 3, tree 6,
water 1, coal 8, iron 9, table 11, …). Linear (ridge) probe → test **AUC** (0.5 = chance, 1.0 =
decodable) on the frozen embedding AND on the rolled RSSM **belief** (the actor's input), with a
shuffled-label control. Pool `cls+patch` (our trained recipe).

## Result

**Decisive GREEN — the foundation carries the full tech-tree structure, end-to-end.** Shuffled
control at chance (~0.49) throughout.

| material | support | emb AUC | **belief AUC** |
|---|---|---|---|
| stone | 3105 | 0.989 | **0.942** |
| iron | 327 | 1.000 | **0.979** |
| coal | 1510 | 0.993 | 0.952 |
| tree | 6953 | 0.995 | 0.944 |
| water | 3857 | 0.996 | 0.956 |
| sand | 3883 | 0.992 | 0.948 |
| lava | 354 | 1.000 | 0.962 |
| table | 141 | 1.000 | 0.999 |

(diamond/furnace low-support — the agent never reaches them, itself an *exploration* signal,
not a perception one.) Every achievement-critical material decodes from the frozen embedding at
**0.99–1.00**, and — the decisive part — **survives the RSSM into the belief at 0.94–0.98**.

## Lesson

**The frozen-encoder foundation is validated; the rung-3 plateau is NOT perception or
architecture — it is exploration/discovery.** The actor has stone, iron, coal, tables, etc.
cleanly available in its belief (AUC 0.94–0.98); it simply never navigates to mine them. This
closes the loop on the whole rung-3 investigation: 0026/0027 ruled out compute, 0028 ruled out
WM-side curiosity, 0030 ruled out self-imitation (can't imitate states never reached), 0029 +
0031 rule out perception. **Everything points to the long-horizon discovery problem** — the
agent doesn't *explore toward* the deep tree — which is the field's known-hard frontier (even
SOTA reaches iron ~3%).

Why materials survive the RSSM but health didn't (0029): materials are large, visually dominant,
reward-relevant features the recon objective preserves; health is a 3-pixel HUD bar it can drop.
The WM keeps what matters for prediction — and it keeps the tech-tree tiles.

**Strategic implications:** (1) **No learned adapter / custom encoder needed** — the frozen
encoder is not the bottleneck ([[frozen-encoder-lean]] holds, now stress-tested for Crafter's
worst-case synthetic sprites). (2) **The foundation is stable → safe to build rung-4
(reading-to-learn) on it.** (3) The 1M-step scaling run is **unnecessary** for the foundation
question and low-value for the plateau (the gap is exploration-diversity, which scaling the same
objective does not fix) — skip or background-only. → next: pivot to the reading-to-learn route
(crafter-rtfm), where reading the recipe *sidesteps* the discovery wall the agent can't explore
its way past.

## Links

[[0029-stat-perception-probe]] · [[0030-self-imitation]] · [[frozen-encoder-lean]] · [[crafter]] · [[no-hardcoded-env]]
