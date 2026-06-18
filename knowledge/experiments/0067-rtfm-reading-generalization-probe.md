---
status: draft
owner: world-model
scope: local
verified: false
last_reviewed: 2026-06-18
---

# 0067: reading-generalization probe — does the agent read MEANING or surface tokens? (paraphrase invariance)

**Status: PRE-REGISTERED — the dead-end check.** Replaces the length-3 idea as the next experiment (the
maintainer's strategic question 2026-06-18): the rung-4 north star is reading **real** tutorials, but we
have only ever evaluated **held-out recipe configs within the templated vocabulary** — which can't tell
genuine reading from a sophisticated lookup over a closed token set. This probe tests the distinction
directly, cheaply (no training; the best exp0065 checkpoint), and **entirely on our side** (a manual-
rewording env wrapper — the crafter-rtfm env is untouched).

## The manual today (real example)

> "The smith's manual. Hold the listed inputs, then perform the steps in order.\nRecipe for offering:
> hold 1x \`sapling\`, 3x \`diamond\`, then perform \`place_furnace\`, \`make_iron_sword\`."

There is already *some* surface variation (preambles "The smith's manual…" / "Forge-lore of this land…",
connectives "then perform" / "and perform"), but the **content tokens are fixed env vocabulary in
backticks** (\`make_iron_sword\`, \`diamond\`). A real tutorial says "forge an iron sword", not
\`make_iron_sword\`. If the agent binds the *token*, it cannot read real tutorials — that's the dead end.

## Probe — swap_follow at three paraphrase levels (same recipe, reworded manual)

Reuse `harness.swap_follow_rate` (scores against the env recipe, independent of manual text) on a
`ManualRewordWrapper` that rewrites `obs["manual"]` before the agent reads it:

- **L0 — original** (baseline).
- **L1 — surface reword, KEEP tokens** (tests robustness to sentence structure; env already varies this
  a little). E.g. → "Here's the job. You'll need 1 \`sapling\` and 3 \`diamond\`. Carry out
  \`place_furnace\`, then \`make_iron_sword\`."
- **L2 — full NATURAL LANGUAGE** (the north-star test; backticks gone, tokens → NL phrases). E.g. → "For
  the offering you need a sapling and three diamonds. First build a furnace, then forge an iron sword."

Token → NL dictionary (the key artifact): `make_iron_sword`→"forge an iron sword",
`make_stone_pickaxe`→"craft a stone pickaxe", `place_furnace`→"build a furnace",
`place_table`→"set up a crafting table", `place_plant`→"plant a sapling", `place_stone`→"place a stone
block", `make_*_{sword,pickaxe}`→"craft/forge a {wood/stone/iron} {sword/pickaxe}"; items
`diamond`→"a diamond", `sapling`→"a sapling", etc. (Hand-built, local — no external LLM.)

## Hypothesis / fork

Best exp0065 checkpoints (s3=0.21, s2=0.21), swap_follow + swap_follow_s1 over the standard eval seeds at
each level. **Invariance ratio = swap_follow(Lk) / swap_follow(L0).**

- **Survives L1 AND L2** (ratio ≳ 0.8) → genuine **meaning-based** reading (MiniLM bridges
  token↔paraphrase and the binding is smooth) → on track for real tutorials → climb the ladder / test
  cross-domain next.
- **Survives L1, COLLAPSES on L2** (ratio ≲ 0.5) → the agent reads env **tokens, not meaning** →
  lookup-ish → **the dead-end signal**: redesign the task to force open-vocabulary reading (train on
  paraphrase-augmented / NL manuals, or an NL manual distribution) BEFORE investing more in execution.
- **Collapses even on L1** → brittle to surface form → worse than expected; same redirect, more urgent.

Watch step-1 vs full: step-1 may survive paraphrase while the chain breaks (binding degrades with depth).

## Setup

`scripts/paraphrase_probe.py <ckpt> --length 2` — loads the checkpoint (as train_rtfm builds it), runs
`swap_follow_rate` at L0/L1/L2 via the wrapper, prints swap_follow / swap_follow_s1 + invariance ratios.
Checkpoints: `runs/exp0065b/s3_s3.pt`, `runs/exp0065b/s2_s2.pt`. No training, no env changes.

## Standing bar — diagnostic (tag: `EXTRA-RIGOR`)

Does swap-following survive **L2 (natural language)**? This is the single most informative cheap test of
whether the whole rung-4 line generalizes toward real tutorials, or whether we're building a lookup table
and need to change the task. Either result redirects strategy, not just the next experiment.

## Links

[[0065-rtfm-per-step-plateau-large-budget]] · [[rung4-manual-conditioned-agent]] · [[language-grounding]] · [[mentored-learning-loop]] · [[hierarchical-imagination-agent]]
