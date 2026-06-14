---
status: draft
owner: agent
scope: local
sources: [arxiv:2306.15934]
verified: true
last_reviewed: 2026-06-14
---

# Curious Replay for Model-based Adaptation (Kauvar et al., ICML 2023)

**Authors:** Isaac Kauvar, Chris Doyle, Linqi Zhou, Nick Haber
**Lab:** Stanford (Haber lab / AutonomousAgentsLab)
**Code:** github.com/AutonomousAgentsLab/curiousreplay
**Read state:** method-depth fetch 2026-06-14; verified against HTML full-text

## One-paragraph summary

Curious Replay is a prioritized experience replay scheme for model-based agents that
combines two curiosity signals: a **count-based novelty bonus** (unseen transitions
replayed more) and an **adversarial model-loss term** (transitions the world model
predicts poorly replayed more). Applied to DreamerV3 on Crafter it achieves 19.4%
mean score vs 14.5% baseline (+1.33×) and shows particular improvement on deep/rare
achievements in the tech tree.

## Priority formula (implementation-grade)

### Combined priority (Equation 1)

```
p_i = c · β^(v_i)  +  (|ℒ_i| + ε)^α
```

Where:
- `v_i` — visit count for transition i (number of times sampled into a training batch)
- `β ∈ [0,1]` — count-decay exponent; higher → slower decay (e.g. β=0.7)
- `c` — scale factor balancing the two terms (c=1e4 in all experiments)
- `ℒ_i` — cached world-model loss for transition i (scalar, see below)
- `ε` — numerical floor to ensure non-zero priority (ε=0.01)
- `α ∈ [0,1]` — loss exponent; <1 compresses dynamic range (α=0.7 in all experiments)

### Count-based term: `c · β^(v_i)`

- `v_i` is initialized to 0 for new transitions
- After each training step, `v_i ← v_i + 1` for every transition in the batch
- The term is monotone-decreasing in visit count: novelty decays with exposure
- Counter vector `v ∈ ℝ^|R|` for full buffer capacity |R|

### Model-loss term: `(|ℒ_i| + ε)^α`

For DreamerV3, the per-transition loss is the **sum of image, reward, and KL losses**:

```
ℒ_i = ℒ_image,i + ℒ_reward,i + ℒ_KL,i
```

The loss is cached per-transition during the training step that sampled it; it is
NOT recomputed for unsampled transitions. This means most priorities are stale between
training steps, but the authors report no observed deficiencies from staleness.

### Sampling distribution

**Proportional** sampling — no rank-based variant:

```
P(select i) = p_i / Σ_j p_j
```

Implemented via a SumTree for O(log n) sampling. **No importance-sampling (IS)
correction** (no β-IS weight) — unlike standard PER. This is a deliberate simplification;
the paper does not discuss bias from non-uniform sampling.

### Priority initialization for new transitions

New transitions enter the buffer with `p_i ← p_MAX` (set to 1e5), ensuring all
new experience is eligible for near-maximum sampling before any loss is observed.

## Hyperparameters (all environments, fixed after tuning on object-interaction assay)

| Symbol | Value | Role |
|--------|-------|------|
| β | 0.7 | Count-decay rate |
| α | 0.7 | Loss exponent |
| c | 1e4 | Count-term scale |
| ε | 0.01 | Loss floor |
| p_MAX | 1e5 | Initial priority for new transitions |

These were **tuned once on the object interaction benchmark and then frozen** across
all other environments including Crafter. No per-environment tuning.

## Crafter results

| Method | Score (mean ± std) |
|--------|---------------------|
| DreamerV3 baseline | 14.5 ± 1.6% |
| DreamerV3 + Curious Replay | 19.4 ± 1.6% |

**Does CR help deep/rare achievements specifically?** Yes, and this is the key finding.
Per Figure 4 analysis: "Curious Replay exhibiting higher success at the more challenging
achievements such as Collect Iron and Make Stone Pickaxe." On Make Iron Sword (the
deepest tech-tree node), CR succeeded on 2/10 seeds while the baseline succeeded on
zero. The improvement is concentrated in the RARE/DEEP achievement tier, not just a
uniform score lift. This directly addresses the depth problem (wood→table→pickaxe→stone→iron).

## Ablations

No direct isolation of count-only vs loss-only vs combined is reported in the main
results table. The paper compares:
- CR vs baseline DreamerV3
- CR vs DreamerV3 + TD-error prioritization (standard PER baseline)
- On object-interaction: CR is >6× faster than Plan2Explore w/ TD prioritization

TD-error prioritization alone does **not** achieve CR's gains, establishing that the
curiosity-specific signal (count + model-loss) is doing meaningful work beyond generic
PER. However, the paper does not provide a clean count-only or loss-only ablation on
Crafter, so the relative contribution of each term is not cleanly isolated.

## Update rule (algorithm summary)

```
Initialize: v[i] = 0 for all i; p[i] = p_MAX for all i
Each training step:
  1. Sample batch B using P(i) = p_i / Σ p_j
  2. Train world model + policy on batch B
  3. For each transition i in B:
       v[i] ← v[i] + 1
       ℒ[i] ← cached loss from step 2
       p[i] ← c·β^(v[i]) + (|ℒ[i]| + ε)^α
  (Priorities of un-sampled transitions are NOT updated — stale until next sampling)
```

## Applicability to frozen-encoder / cached-embedding setup

**Our setup:** WM trains on cached DINO embeddings. Per-transition learnable losses are:
- `ℒ_embedding_recon` — reconstruct embedding from belief state
- `ℒ_KL` — stochastic latent regularization (RSSM)
- `ℒ_reward` — reward prediction

**Available signals:**
- ✓ **KL term** — directly available; measures surprise in the stochastic latent (z)
- ✓ **Embedding-reconstruction loss** — directly available; replaces ℒ_image
- ✓ **Reward-prediction loss** — directly available
- ✓ **Visit count** — purely bookkeeping, always available
- ✓ **Combined priority** — all components available; our `ℒ_i = ℒ_recon + ℒ_KL + ℒ_reward`

**Unavailable / N/A:**
- ✗ **Pixel reconstruction loss** — not applicable (we never decode to pixels)
- ✗ **Image-space VQ/tokenization loss** (IRIS-style) — not our architecture

**Key caveat:** The model-loss term reflects how well the RSSM belief predicts cached
DINO embeddings, NOT raw pixel prediction error. This is a coarser signal than pixel
loss — DINO features are already semantically compressed, so reconstruction error may
be smaller in magnitude and less sensitive to low-level novelty. Whether the signal
remains discriminative enough to drive useful replay prioritization is an empirical
question for our stack.

**Implementation cost:** Low. Requires:
1. A SumTree over the replay buffer (standard PER data structure)
2. Per-transition loss caching during the training step
3. Visit counters (`v` vector, same size as buffer)
4. Priority update after each gradient step

No architectural changes required. Can be grafted onto the existing RSSM flywheel.

## Open questions

- Does embedding-recon loss (vs pixel loss) remain a discriminative curiosity signal?
  The compressed DINO space may have smaller variance than pixel space.
- No count-only vs loss-only ablation on Crafter — is the count term load-bearing or
  mostly the model-loss term?
- The p_MAX initialization assumes new transitions are maximally novel; does this create
  a fast-forgetting of the earliest buffer contents once they accumulate high visit counts?
- IS correction omitted — any bias concern under long training (buffer becomes very
  non-uniform)?

## Links

[[dreamerv3-2023]] · [[rssm]] · [[crafter]] · [[prioritized-replay]] ·
[[plan2explore]] · [[dreamerv3-xp-2510.21418]]
