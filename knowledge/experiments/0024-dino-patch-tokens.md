---
status: draft
owner: human
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

_pending (dispatched)_

## Lesson

_pending_

## Links

[[0023-twohot-value]] · [[0022-frozen-dino-probe]] · [[frozen-encoder-lean]] · [[crafter]]
