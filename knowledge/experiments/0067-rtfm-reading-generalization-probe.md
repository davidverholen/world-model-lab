---
status: draft
owner: world-model
scope: local
verified: false
last_reviewed: 2026-06-18
---

# 0067: reading-generalization probe — does the agent read MEANING or surface tokens? (paraphrase invariance)

**Status: DONE (best 2 exp0065 checkpoints) — NOT a dead end. The agent reads MEANING, not surface
tokens: swap-following retains ~79–83% under FULL natural-language rewording** (env tokens →
"forge an iron sword"). A pure token-lookup would collapse toward 0; it doesn't. The "language as
commodity" design (frozen MiniLM bridges token↔paraphrase) holds empirically — the env vocabulary is a
*training distribution*, not a hard wall. Caveat: a consistent **~20% NL penalty** (reading isn't fully
abstract) and the **cross-domain-WM axis is still untested** (the bigger remaining gap).

## Result — swap-following survives natural-language rewording (60 SWAPPED seeds, len-2)

| level | s3 full | (ratio) | s2 full | (ratio) | s3 step-1 | s2 step-1 |
|---|---|---|---|---|---|---|
| L0 original | 0.200 | 1.00 | 0.233 | 1.00 | 0.500 | 0.600 |
| L1 surface (keep tokens) | 0.167 | 0.83 | 0.267 | 1.14 | 0.417 | 0.567 |
| **L2 natural language** | 0.167 | **0.83** | 0.183 | **0.79** | 0.383 (0.77) | 0.450 (0.75) |

Example L2 the agent read and still followed: *"…Your task: gather a sapling, stone, then carry out
**forge an iron sword, forge an iron sword**."* — i.e. the backtick env-tokens fully replaced by NL.

**Read:** under full NL, both checkpoints keep **~0.79–0.83** of their full swap-following and **~0.75–0.77**
of step-1. If the binding were a lookup on the `make_iron_sword`-style tokens, L2 (tokens gone) would
fall toward the no-manual floor — instead it barely moves beyond the surface-structure cost. So the
binding is **meaning-based**: MiniLM places the token near its paraphrase and the learned WM-conditioning
generalises across that gap. This directly answers the maintainer's dead-end worry on the **reading axis:
we are not building a lookup table.**

**Held-out-NL follow-up (60 seeds, s3) — strengthens it.** Added an L2b level with a *different* NL dict
+ framing ("create an iron blade", "construct a furnace") never used anywhere else: L0 0.200 → L1 0.183
(0.92) → L2 0.183 (0.92) → **L2b held-out 0.150 (0.75)**. A *graded* penalty — ~8% for near-paraphrases,
~25% for totally unseen phrasings — but **no collapse**. (An earlier 12-seed probe showed ~0 on L2b; that
was noise.) So the binding reads meaning even for phrasings it has never seen, just with a larger penalty
the further the surface form sits from the env tokens — exactly what [[0068-rtfm-paraphrase-augmented-reading]]
tries to close.

**Honest caveats (where the real work is):**
1. **A real ~20% NL penalty.** Both checkpoints L2 < L0, and in s2 it's specifically the token→NL swap
   (L1 0.267 → L2 0.183), so reading isn't *fully* abstract — there's residual surface/token dependence.
   Cheap fix to test next: **paraphrase-augment the training manual distribution** (the env already
   varies preambles; widen it) — likely closes the gap and is a direct robustness win for real tutorials.
2. **Small, noisy numbers** (0.18–0.27, 60 seeds, ~0.05 sd) — per-checkpoint significance is weak; the
   *consistent* L2<L0 direction across two checkpoints is the signal, not the exact ratio.
3. **This tests reading/binding ONLY.** "Read a real tutorial in a game it didn't train on" also needs
   **cross-domain world-model transfer**, which rung-4 does not build — that's the bigger, far gap
   ([[mentored-learning-loop]] / the env ladder), and the honest next strategic axis once reading is banked.

**Strategic takeaway:** not a dead end on reading → don't over-grind crafter *execution*; the higher-
leverage moves are (a) NL-augment the manuals to close the 20% gap, and (b) start testing **transfer** up
the ladder rather than perfecting one env.

## Original pre-registration

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
