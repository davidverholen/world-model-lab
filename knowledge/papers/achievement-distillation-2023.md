---
status: draft
owner: world-model
scope: local
sources: [arxiv:2307.03486]
verified: true
last_reviewed: 2026-06-14
---

# Achievement Distillation (Moon, Yeom, Park, Song — NeurIPS 2023)

**Authors:** Seungyong Moon, Junyoung Yeom, Bumsoo Park, Hyun Oh Song
**Lab:** Seoul National University (Song lab / mllab.snu.ac.kr) + KRAFTON
**Code:** https://github.com/snu-mllab/Achievement-Distillation
**Read state:** full-text read 2026-06-14; all formulas verified against PDF pages 5–9

## One-paragraph summary

Achievement Distillation (AD) is a self-supervised auxiliary learning method layered on
top of PPO that trains the encoder to predict the next achievement to be unlocked in a
trajectory. It does this via two contrastive losses: an **intra-trajectory** loss
(state-action pair predicts its future achievement) and a **cross-trajectory** loss
(achievements matched between different episodes via optimal transport are pulled
together). On Crafter it achieves 21.79% score at 1M environment steps — the best
from-scratch result at time of publication — with only 9M parameters (4% of DreamerV3's
201M). The method requires only the binary achievement-unlock reward signal, no explicit
knowledge of the achievement graph.

## Why it matters here

Our RSSM actor-critic stalls at shallow/mid achievements on Crafter (exp 0025–0026
finding). AD directly addresses the depth problem: it achieves 3% Collect Iron vs
DreamerV3's ~0.15% (20× improvement) and is the only from-scratch method to reach iron
tools. The contrastive self-imitation idea is paradigm-agnostic — only the PPO
integration is model-free specific; the achievement-segment representation mechanism is
grafable onto imagination actor-critics (see Portability note below).

## Details

### Base algorithm

PPO with a modified ResNet (IMPALA-style) encoder φ_θ : S → R^h:
- Channel sizes expanded from [16, 32, 32] to [64, 128, 128]
- Hidden size expanded from 256 to 1024
- Layer normalization before each dense/convolutional layer
- Value normalization (running mean/std on targets)

The policy π_θ and value V_θ share the encoder and are fed `concat(φ_θ(s_t), ν_θ(g_t⁻))`
where `ν_θ(g_t⁻)` is the **previous achievement representation** used as memory (Section 4.3).

### Achievement segmentation: how sub-trajectories are defined

Achievements in an episode are ordered by the timestep at which they are first unlocked.
Let `(g_i)_{i=1}^m` be the sequence of unlocked achievements in an episode, and
`(t_i)_{i=1}^m` be their corresponding unlock timesteps. For each timestep t:

- **Next achievement** `g_t⁺ = g_u` where `u = min{i | t ≤ t_i}` — the first achievement
  yet to be unlocked at time t (the target for intra-trajectory prediction).
- **Previous achievement** `g_t⁻ = g_l` where `l = max{i | t > t_i}` — the most recent
  achievement already unlocked at time t (used as memory).

**Segment boundary = an achievement unlock event.** A new segment begins each time a
novel achievement is collected (binary reward = 1 for a new achievement). The agent does
NOT need to know the achievement graph — segment boundaries are read directly from the
sparse reward signal. Achievements re-collected after the first unlock do NOT create new
boundaries (reward = 0 for repeated unlocks).

Achievement g_i itself is represented as:
```
ν_θ(g_i) = normalize( φ_θ(s_{t_i+1}) − φ_θ(s_{t_i}) )
```
i.e., the **residual of encoder outputs across the unlock transition**, normalized.
This is motivated by Nair et al. [38] — the latent difference captures what changed at
the unlock step. No separate achievement embedding table; the representation is derived
from the trajectory itself.

### Intra-trajectory contrastive loss (L_pred)

Given state-action pair (s_t, a_t) and its next achievement g_t⁺ within episode τ:

- **Anchor:** ν_θ(g_t⁺) — the achievement representation
- **Positive:** ψ_θ(s_t, a_t) — the state-action representation (FiLM-conditioned action
  into φ_θ(s_t), then MLP + normalize)
- **Negative:** ψ_θ(s_{t'}, a_{t'}) — a randomly sampled OTHER state-action pair from
  the SAME episode τ

InfoNCE-style loss (temperature λ):

```
L_pred(θ) = −E_{(s_t,a_t)~τ, (s_{t'},a_{t'})~τ} [
    log(
        exp(ψ_θ(s_t,a_t)ᵀ ν_θ(g_t⁺)/λ) /
        ( exp(ψ_θ(s_t,a_t)ᵀ ν_θ(g_t⁺)/λ) + exp(ψ_θ(s_{t'},a_{t'})ᵀ ν_θ(g_t⁺)/λ) )
    )
]
```

This is a **binary** InfoNCE (one positive, one negative per anchor). The positive is the
state-action pair pulled toward the achievement it precedes; the negative is any other
state-action in the same episode.

To prevent distortion of the policy and value heads, two regularizers are jointly
minimized:

```
R_π(θ) = E_{s~τ} [ D_KL(π_{θ_old}(·|s) ∥ π_θ(·|s)) ]
R_V(θ) = E_{s~τ} [ ½ (V_θ(s) − V_{θ_old}(s))² ]
```

where θ_old is the policy/value snapshot immediately before the auxiliary phase.

### Cross-trajectory contrastive loss (L_match)

Given achievement sequences g = (g_i)_{i=1}^m and g' = (g'_j)_{j=1}^n from two different
episodes, a soft matching T* ∈ R^{m×n} is computed via **partial optimal transport**:

Cost matrix:
```
M_{ij} = 1 − ν_θ(g_i)ᵀ ν_θ(g'_j)    (cosine distance)
```

Soft matching (entropic regularization parameter α):
```
T = argmin_{T≥0} <T, M> + α Σ_{i,j} T_{ij} log T_{ij}
    s.t.  T1 ≤ 1,  Tᵀ1 ≤ 1,  1ᵀTᵀ1 = min{m, n}
```

Hard matching T* = 1[T > 0.5] (thresholds away uncertain matches; each achievement
matched to at most one other). If g_i is matched to g'_k under T*:

- **Anchor:** ν_θ(g_i)
- **Positive:** ν_θ(g'_k) — matched achievement from the OTHER episode
- **Negative:** ν_θ(g'_j) — a randomly sampled UNMATCHED achievement from the same target sequence

InfoNCE-style loss (same temperature λ):
```
L_match(θ) = −E_{g_i~g, g'_k~g', g'_j~g'} [
    log(
        exp(ν_θ(g_i)ᵀ ν_θ(g'_k)/λ) /
        ( exp(ν_θ(g_i)ᵀ ν_θ(g'_k)/λ) + exp(ν_θ(g_i)ᵀ ν_θ(g'_j)/λ) )
    )
]
```

Same R_π, R_V regularizers are jointly minimized here too.

### Integration with RL training (Algorithm 1)

Two alternating phases per outer loop:

1. **Policy phase** (repeated N_π times): collect episodes T using π_θ, store in buffer B;
   run E_π PPO epochs on T to optimize J_π(θ) and J_V(θ).
2. **Auxiliary phase** (run once after the policy phase): update π_{θ_old} ← π_θ; run
   E_aux epochs on ALL of B optimizing:
   - L_pred(θ), R_π(θ), R_V(θ)  (intra-trajectory)
   - L_match(θ), R_π(θ), R_V(θ)  (cross-trajectory)

The auxiliary phase uses the full buffer B (all previously collected episodes), not just
the most recent rollout. There is **no separate loss weight** between L_pred and L_match —
both are optimized independently in the same auxiliary phase. No λ_aux coefficient mixing
them with J_π; the phases are strictly alternated, not summed.

## Crafter results

**Compute budget:** 1M environment steps (same as DreamerV3 standard eval).

**Overall score:** 21.79 ± 1.37% (geometric mean of 22 achievement success rates).

| Method | Score (%) | Reward | Params |
|--------|-----------|--------|--------|
| **AD (Ours)** | **21.79 ± 1.37** | **12.60 ± 0.31** | 9M |
| PPO (modified) | 15.60 ± 1.66 | 10.32 ± 0.53 | 4M |
| DreamerV3 | 14.77 ± 1.42 | 10.92 ± 0.53 | 201M |
| LSTM-SPCNN | 11.67 ± 0.80 | 9.34 ± 0.23 | 135M |
| MuZero+SPR (no pre-train) | 4.4 ± 0.4 | 8.5 ± 0.1 | 54M |
| Human Expert | 50.5 ± 6.8 | 14.3 ± 2.3 | — |

**Deep-tree achievement highlights (from Figure 6 / Section 5.2):**
- **Collect Iron:** AD ~3%, DreamerV3 ~0.15% → **20× improvement**
- **Make Iron Sword:** AD ~0.01% (achievable on individual runs); DreamerV3 ≈ 0%
- **Collect Diamond:** AD reaches it on individual runs; DreamerV3 ≈ 0%
- The improvement is concentrated in the mid-to-deep tier (iron+ level); shallow
  achievements (collect wood, place table, etc.) near-saturated for all methods

**Ablation (Table 2) — all 1M steps, Crafter:**
| I (intra) | C (cross) | M (memory) | Score (%) |
|-----------|-----------|------------|-----------|
| ✗ | ✗ | ✗ | 15.60 ± 1.66 (PPO baseline) |
| ✓ | ✗ | ✗ | 19.02 ± 1.65 |
| ✓ | ✓ | ✗ | 20.36 ± 1.79 |
| ✓ | ✓ | ✓ | **21.79 ± 1.37** |

Intra-trajectory prediction is the dominant contributor; cross-trajectory matching and
memory add incremental but meaningful gains.

**Representation quality:** AD encoder achieves 73.6% accuracy on next-achievement
classification (vs 44.9% for PPO), with median confidence 0.752 (vs 0.240 for PPO).

**Generalization:** AD also works on off-policy (QR-DQN, score 4.14→8.07), Procgen
Heist (score 29.6→71.0), and MiniGrid (3.33→8.04), confirming the method is not
PPO-specific in its mechanism.

## Portability note: grafting onto a DreamerV3-style imagination actor-critic

### What is paradigm-specific (PPO only)

- The **outer training loop** (alternate policy-phase / auxiliary-phase): PPO's on-policy
  rollout buffer is the natural substrate. An imagination AC can substitute real-rollout
  segments with imagined segments for auxiliary training, but note the achievement
  representation ν_θ(g_i) = normalize(φ_θ(s_{t_i+1}) − φ_θ(s_{t_i})) requires access
  to consecutive latent states at the unlock transition — available in an RSSM replay
  buffer but not in pure imagined trajectories (imagined states may not align with real
  achievement unlocks).
- The **R_π KL regularizer** is PPO-specific in form; an imagination AC would need an
  equivalent distillation regularizer (e.g., keep the actor distribution close to the
  pre-auxiliary snapshot, which is standard in any actor regularization scheme).

### What is paradigm-agnostic (transferable)

- **Achievement segmentation:** reading segment boundaries from the binary achievement
  reward signal. Works identically in any RL algorithm — the unlock event is
  environment-level information.
- **Achievement representation ν_θ(g_i):** residual latent across the unlock transition.
  In an RSSM, this is `normalize(h_{t+1} − h_t)` or `normalize(z_{t+1} − z_t)` at the
  step where reward = 1 (new achievement). Computable directly from the replay buffer.
- **Intra-trajectory InfoNCE loss:** pull state-action representations toward the latent
  of the achievement they precede; push away random same-episode pairs. This is a pure
  representation-learning signal that can be applied to the RSSM encoder (or the
  posterior encoder z_t) as an auxiliary objective during world-model training.
- **Cross-trajectory OT matching:** match achievement latents across episodes and apply
  another InfoNCE. This is encoder-agnostic — it operates on ν_θ(g_i) representations,
  which can be derived from RSSM belief states.

### Minimal graftable version (imagination AC)

The minimum viable port is **intra-trajectory achievement prediction only** (the dominant
contributor per ablation, adding ~3.4 pp alone):

1. From the replay buffer, extract timesteps where a new achievement was collected.
2. Compute achievement representation: `ν(g_i) = normalize(h_{t_i+1} − h_{t_i})` using
   RSSM deterministic states (or append stochastic z: `normalize([h;z]_{t+1} − [h;z]_t)`).
3. For each preceding timestep t < t_i, form the state-action rep `ψ(s_t, a_t)` via FiLM
   or concatenation of the RSSM state and action, then an MLP head.
4. Apply the binary InfoNCE loss L_pred on real-trajectory segments in the replay buffer,
   as an auxiliary objective run after the world-model / actor-critic update step.

No changes to the RSSM architecture, the imagination rollout, or the actor-critic loss
are required. The auxiliary loss touches the encoder (φ_θ or the posterior), which feeds
both the WM and the actor. The R_V regularizer equivalent is simply freezing the critic
target during the auxiliary step (or the existing EMA critic in DreamerV3).

**Key risk:** In RSSM-based agents, the encoder is the posterior over latents. Distorting
it toward achievement prediction may interfere with world-model fidelity (reconstruction /
KL balance). The R_π / R_V regularizers in AD exist precisely for this reason — an
analogous WM-reconstruction regularizer would be needed (e.g., keep ℒ_recon below a
threshold during the auxiliary phase, or apply a EWC-style penalty on the posterior
encoder parameters).

## Open questions

- Does the achievement residual `normalize(φ(s_{t+1}) − φ(s_t))` remain a useful
  signal when φ is the RSSM posterior encoder (which sees the full belief, not just
  the raw image)? The RSSM belief may already smooth out the unlock event.
- Would replacing the binary InfoNCE with a multi-negative version (all other achievements
  in the buffer) improve the cross-trajectory matching further?
- The authors note "we have not evaluated the transferability of our method to agent
  without any reward" — AD requires the sparse unlock signal. Could achievement boundaries
  be detected from belief-state anomaly detection instead (enabling the fully unsupervised
  version)?
- Interaction with Curious Replay: AD targets the actor/encoder; CR targets the replay
  buffer. They are orthogonal — can they be stacked on a single RSSM agent for additive
  gains?

## Links

[[curious-replay-2023]] · [[dreamerv3-2023]] · [[rssm]] · [[crafter]] ·
[[ppo]] · [[contrastive-learning]] · [[optimal-transport]]
