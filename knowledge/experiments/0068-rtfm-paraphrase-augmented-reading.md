---
status: draft
owner: world-model
scope: local
verified: false
last_reviewed: 2026-06-18
---

# 0068: paraphrase-augmented training — does it make reading robust to UNSEEN natural language?

**Status: DONE (aug-ON vs aug-OFF, 60 rounds, 4+4 seeds) — augmentation TEACHES the trained NL forms but
does NOT generalise to held-out NL, at a base-task cost. "Memorisation-leaning."** A small fixed
paraphrase set is not enough for real-tutorial robustness → the path is language DIVERSITY at scale
(LLM-paraphrase / the read→LLM seam), not hand-written templates.

## Result (60 eval seeds × 4 model seeds per arm)

| level | aug-OFF | aug-ON | ON/OFF |
|---|---|---|---|
| L0 base | 0.233 | 0.192 | 0.82 |
| L1 surface | 0.188 | 0.221 | 1.18 |
| **L2 in-dist NL** (trained-on forms) | 0.121 | **0.204** | **1.69** |
| **L2b held-out NL** (never trained) | 0.025 | 0.050 | 2.01 |

- **In-distribution NL (L2): a real ~1.7× gain.** Training on the L2 reword forms made the agent read
  *those* forms — augmentation does what it claims on the trained distribution.
- **Held-out NL (L2b): does NOT robustly generalise.** The 2× is on tiny numbers (0.025→0.050, both
  within ~1 sd; per-seed ON [0.067, 0.05, 0.067, 0.017] vs OFF [0.0, 0.05, 0.017, 0.033] overlap
  heavily) and the absolute retention is ~0.26 of L0. So the binding mostly **memorised the specific
  paraphrase set**, with only weak transfer to genuinely unseen phrasings.
- **Base-task cost:** L0 0.233 → 0.192 (~18%, partly a noisy weak seed) — training on harder, varied
  manuals at fixed budget slightly hurts the core recipe.

**Caveats:** all numbers are small and the 60-round models are *under-trained* (exp0065 needed ~120
rounds for the 0.20 plateau) — held-out NL sits near the floor for both arms, so the L2b comparison is
weakly powered; a longer-trained aug model might transfer more. The robust signal is **L2 (in-dist):
augmentation clearly works for forms it sees.** The honest conclusion stands at the qualitative level: a
3-template paraphrase set generalises only as far as its own diversity.

**Strategic takeaway:** robust open-vocabulary reading needs **language diversity at scale** — many
diverse paraphrases (e.g. LLM-generated, the [[hierarchical-imagination-agent]] §4 read→LLM seam), not a
handful of rules. The reading axis is *fixable*, but with breadth of language, not template count. This
also re-surfaces the bigger untested axis: **cross-domain world-model transfer** (a game it didn't train
on), which no rung-4 experiment touches.

## Original pre-registration

## Why (exp0067 + a sharpening)

[[0067-rtfm-reading-generalization-probe]] found swap-following survives natural-language rewording with a
**graded** penalty (60-seed baseline, s3): in-dist NL retains ~0.92, **held-out NL (L2b, a different
dict/framing) retains ~0.75** — meaning-based, but the further the phrasing sits from the env tokens the
bigger the penalty. exp0068 asks whether **training** on surface-varied manuals closes that held-out gap.
(An earlier 12-seed probe showed L2b ~0 — that was noise; the 60-seed baseline is 0.75.)

## Probe levels (eval) — note L2b is HELD OUT of training

L0 original · L1 surface (keep tokens) · L2 natural-language (dict A) · **L2b held-out NL (dict B,
different phrasings + framing — never in training)**. Training augmentation samples only L0/L1/L2
(`manual_paraphrase.sample_paraphrase`, ~34/33/33); **L2b is the generalisation test.**

## Hypothesis / fork

Train aug-ON (`--paraphrase-aug`) vs aug-OFF at the proven config, length-2, 60 rounds, 4 seeds each;
then run the paraphrase probe (L0/L1/L2/L2b) on both. **Headline metric: L2b (held-out NL) swap_follow,
aug-ON vs aug-OFF.** Also watch L0 (aug must not wreck the base task).

- **aug-ON reads L2b ≈ its L0, aug-OFF collapses on L2b** → surface augmentation confers *genuine*
  open-vocabulary reading that generalises to unseen phrasings → cheap, real robustness toward real
  tutorials → this is the path; widen the augmentation, climb the ladder.
- **aug-ON helps L2 (in-dist) but NOT L2b** → it memorises trained paraphrase forms, doesn't generalise
  → augmentation alone is insufficient → need richer language grounding (more diverse augmentation, or a
  pretraining/installation lever).
- **aug-ON ≈ aug-OFF, both collapse on L2b** → the binding is fundamentally phrase-bound at this scale →
  a deeper grounding limit; reconsider the text→world-model architecture ([[language-grounding]]).
- **aug-ON L0 ≪ aug-OFF L0** → augmentation hurts the base task (too-hard manuals) → tune the mix.

## Setup

```
scripts/dispatch_rtfm.sh exp0068off 4 <proven> --rounds 60 --curriculum-rounds 10 --no-paraphrase-aug
scripts/dispatch_rtfm.sh exp0068on  4 <proven> --rounds 60 --curriculum-rounds 10 --paraphrase-aug
```
(proven = `--length 2 --manual-aux-coef 1.0 --validated-reading-coef 0.3 --n-train-seeds 400
--path-reward-coef 0.5 --path-reward-factor 1 --path-reward-decay 1.0`). 8 procs. After training:
`scripts/paraphrase_probe.py runs/exp0068{off,on}/s*.pt` → compare L2b.

## Standing bar — diagnostic (tag: `EXTRA-RIGOR`)

Does paraphrase-augmented training lift held-out-NL (L2b) swap-following clearly above the baseline? A
yes is the first evidence the rung-4 reader can be made robust to *arbitrary* tutorial phrasing — the
property the real-tutorial north star needs. A no localises the limit (memorisation vs architectural).

## Links

[[0067-rtfm-reading-generalization-probe]] · [[rung4-manual-conditioned-agent]] · [[language-grounding]] · [[mentored-learning-loop]] · [[0065-rtfm-per-step-plateau-large-budget]]
