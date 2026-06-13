---
status: draft
owner: agent
scope: local
sources: [arxiv:2109.06780]
verified: false
last_reviewed: 2026-06-13
---

# Crafter (rung 3)

## What it is

2D open-world survival game (Hafner 2021). 64×64 RGB observations, **17 discrete actions**,
episode length 10000, death when health hits 0. The payload is a **22-achievement tech
tree** (wake_up → collect_wood → place_table → make_wood_pickaxe → collect_stone → … →
collect_diamond) with sparse, hierarchical, prerequisite-gated rewards. Standard metric:
**Crafter score = geometric mean of per-achievement unlock rates** across episodes (geometric
so broad shallow progress doesn't mask never reaching the deep achievements). Reference
budget: 1M env steps.

Why it's rung 3: the first env where **hierarchy and long-horizon credit assignment** are
required, not optional — a flat policy grabs the shallow achievements (wood, table) and
stalls before the deep ones (iron, diamond) that need a multi-step plan.

## Our wrapper (src/world_model/envs/crafter.py)

Crafter predates gymnasium (old-gym 4-tuple `step`, no-arg `reset`, own space classes).
`CrafterEnv` is a gymnasium adapter emitting **CHW float32 obs in [0,1]** exactly like
[[minigrid]], so the same encoder/agents/training consume it unchanged. `done` is split into
terminated (death; `info['discount']==0`) vs truncated (length limit). Seeded `reset(seed=s)`
rebuilds the underlying env (construct ~0.02s) for reproducible eval worlds; unseeded resets
vary (collection). `info['achievements']` (the 22-dict) is the score basis. Per
[[0006-crafter-vs-craftax]]: original PyTorch-native Crafter, **pixel** obs (the
[[frozen-encoder-lean]] thesis), not Craftax/JAX, not the symbolic view.

## First contact (2026-06-13)

Random-agent baseline: ~283 steps to death, reward ~0.1, **1/22 achievements** (wake_up) —
the expected random floor (random only trips trivial achievements). Confirms env + reward +
achievement + death/timeout signals all flow through our interface. 19/19 smoke tests pass.

## Role in the ladder — Crafter is a fast proxy for Minecraft

Crafter *exists because* real Minecraft is too slow to iterate on; it is a Minecraft-shaped
abstraction. So the rung-3 → rung-4 relationship is **proxy → real target**, and what crosses
the gap is specific:

- **Method / recipe transfers (re-instantiated, not weight-copied):** RSSM, the calibrated
  imagination loop ([[0021-critic-on-replay]]), hierarchy for the tech tree, language binding.
  Developed and *validated* on cheap Crafter, then rebuilt for Minecraft.
- **The frozen encoder transfers literally (weights and all):** DINO/V-JEPA is pretrained on
  real images, game-agnostic — it spans Crafter *and* Minecraft *and* the real world
  unchanged. The one component that physically carries across rungs ([[frozen-encoder-lean]]).
- **World-model weights do NOT transfer:** dynamics/reward/item knowledge are game-specific;
  retrained on Minecraft from scratch (warm-started by the frozen encoder + validated recipe).
- **New signals switch on at Minecraft:** human gameplay video (VPT action labeling) and the
  wiki/tutorial corpus (MineDojo) — capabilities proven cheaply (grounding on Messenger/
  text-Crafter, hierarchy on Crafter) get *deployed at real scale* there ([[0005-minecraft-milestone]]).
- **JAX corollary:** real envs can't be GPU-fused (Atari C++, Minecraft Java), so Crafter is
  the *last* rung where the Craftax/JAX speedup could even apply — another reason not to rewrite
  ([[compute-strategy]] JAX revisit triggers).

Compute-smart by design: debug expensive capabilities on the fast env, deploy on the slow
real one; the frozen encoder is the thread tying the rungs together.

## Pipeline status (2026-06-13)

Built and **learning**. Frozen DINOv2-S encoder (VALIDATED, [[0022-frozen-dino-probe]]) +
the calibrated RSSM imagination loop (RSSM + critic-on-replay) wired in `train_crafter.py`,
with an embedding cache (store DINO embeddings in replay, zero encoder forwards in WM
updates). First local-GPU run (rounds 0→1): eval_reward 0.10→**1.10**, achievements
1→**2** — above the random floor (1/22, reward ~0.1).

**Open problem — value inflation at Crafter scale:** `imagined_return` 9.95→21.67,
`critic_loss` 14→35 (vs ~1 calibrated on MiniGrid). Denser/larger-scale rewards overwhelm
plain-MSE critic + critic-on-replay (β_repval 0.3). This is model exploitation resurfacing
(cf. 0017/0018) — a long run now would just exploit the model.

## What's next

- **Harden the DreamerV3 recipe (the now-empirically-justified fix):** symlog + **two-hot
  distributional critic** (bounded — can't regress to 21 the way MSE does) + **percentile
  return-normalization** (advantage scale). Deferred from MiniGrid precisely because
  Crafter's dense rewards exercise it; the run above is the evidence it's needed.
- **Then** the first real (longer) desktop run, once the value scale is controlled.
- Later: spatial patch tokens (vs CLS) for finer detail; hierarchy for the deep tech tree.

## Links

[[environment-ladder]] · [[0006-crafter-vs-craftax]] · [[0005-minecraft-milestone]] ·
[[frozen-encoder-lean]] · [[minigrid]] · [[compute-strategy]] · [[hierarchy-and-credit]]
