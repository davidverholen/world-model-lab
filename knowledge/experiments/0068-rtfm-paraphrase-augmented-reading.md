---
status: draft
owner: world-model
scope: local
verified: false
last_reviewed: 2026-06-18
---

# 0068: paraphrase-augmented training — does it make reading robust to UNSEEN natural language?

**Status: PRE-REGISTERED — code committed, default-off `--paraphrase-aug`.** The cheap robustness fix
exp0067 pointed to: train the binding on surface-varied manuals so it reads MEANING, not the exact env
tokens. Crucially evaluated on a **held-out** NL style (never seen in training), so this tests
*generalisation*, not memorisation.

## Why (exp0067 + a sharpening)

[[0067-rtfm-reading-generalization-probe]] found swap-following survives natural-language rewording
(~0.79–0.83) — encouraging. **But a held-out-NL check sharpens it:** the baseline survives the L2 dict
(`make_iron_sword`→"forge an iron sword") yet an *early 12-seed probe collapsed (~0) on a DIFFERENT NL
dict* (`make_iron_sword`→"create an iron blade", `place_furnace`→"construct a furnace"). So the baseline
reads *some* paraphrases (those MiniLM places near the token) but maybe not arbitrary phrasings — i.e. it
is *partly* meaning-based, partly phrase-bound. (60-seed baseline number pending; if it confirms, exp0067's
"reads meaning" softens to "reads MiniLM-near paraphrases".) The fix to test: expose the binding to
surface variation in training.

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
