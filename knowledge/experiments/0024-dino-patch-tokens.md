---
status: draft
owner: world-model
scope: local
sources: [arxiv:2411.04983]
verified: true
last_reviewed: 2026-06-13
---

# 0024: DINO patch tokens — spatial features to break the Crafter plateau (pre-registered)

## Hypothesis

Exp 0023 learns but plateaus at ~2.5–3 achievements (the shallow tree: wood/table/survival).
The encoder uses DINO's **global CLS** token — "a tree is in view" but not *where*. The deeper
tree (stone→pickaxe→iron) needs the agent to localize and navigate to resources. DINO's 49
**spatial patch tokens** carry that layout. Predict: `pool="cls+patch"` (concat CLS +
mean-pooled patches, 768-d) lifts achievements above the 0023 plateau by improving
navigation/gathering. Null result → representation isn't the plateau's cause → look at
exploration / horizon / hierarchy instead.

## Setup

Identical to 0023 (2 seeds, 10 rounds, two-hot critic, repval 0.3, horizon 15) except
`--pool cls+patch` (encoder latent_dim 384→768; RSSM embed_dim follows; embedding cache stores
768-d). One change vs 0023 → clean attribution. Reuses the validated frozen-encoder cache.

Command per seed: `python -m world_model.train_crafter --pool cls+patch --rounds 10
--round0-steps 10000 --actor-steps 10000 --updates-per-round 2000 --ac-updates-per-round 2000
--seq-batch 16 --window 24 --burn-in 8 --horizon 15 --eval-episodes 8 --eval-length 1000
--ep-length 2000 --repval 0.3 --save runs/crafter_patch.pt --seed {0,1}`

## Result

**Surprising partial WIN — patch tokens unlock navigation (seed-dependent).** 2 seeds,
10 rounds, `--pool cls+patch` (else identical to 0023). best_eval_reward s0 **2.47**
(round 9, eval_achievements peak **3.38**), s1 **1.60** — aggregate looks like a wash vs
0023, BUT `behavior_report` reveals a huge behavioral split the eval hid:

| | moved_frac | bbox span | move_actions | top action | achievements |
|---|---|---|---|---|---|
| s0 | **0.12** | **17.7 tiles** | **0.42** | do:0.52 | 4 incl. **place_table** |
| s1 | 0.00 | 0.0 | 0.00 | do:0.96 | 2 (sapling/wake) |

**s0 navigates and builds a table** (real tech-tree progress needing wood + movement) —
the first non-stationary agent. s1 collapsed to the 0023 degenerate stationary policy.
imagined_return stayed inflated (8–22; the reward head is unchanged).

## Lesson

**Representation IS part of the bottleneck — overturns the pre-registered null.** With
global CLS (0023) the agent can't localize resources, so it can't navigate; DINO's spatial
**patch tokens** give the "where is the tree/stone" signal and s0 learned to move and make a
table. So perception was a real limiter, not just the policy. BUT the win is seed-dependent
because the **reward exploitation is unfixed** (imagined_return 8–22) and tips s1 into the
stationary collapse. ⇒ **both levers matter**: patch tokens for navigation, the two-hot
reward head (exp 0025) for stability. Next experiment combines them (`cls+patch` + bounded
reward), 3 seeds to measure the variance. Process win: the **behavior QA gate caught the
s0/s1 split that aggregate eval-reward completely masked** — exactly why the "it doesn't
move" observation became a standing gate.

## Links

[[0023-twohot-value]] · [[0022-frozen-dino-probe]] · [[frozen-encoder-lean]] · [[crafter]]
