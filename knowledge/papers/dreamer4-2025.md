---
status: draft
owner: world-model
scope: local
sources: [arxiv:2509.24527]
verified: true
last_reviewed: 2026-06-15
---

# Dreamer 4: Training Agents Inside of Scalable World Models (Hafner, Yan, Lillicrap, 2025)

**Lab:** Google DeepMind · **Project:** https://danijar.com/project/dreamer4/ · **arXiv:** 2509.24527 · **Submitted:** 2025-09-29 · **Read state:** method depth (primary source: TalkRL transcript with Hafner + three secondary technical writeups; PDF compressed/inaccessible)

## What it is

Dreamer 4 is a 2-billion-parameter world-model agent that replaces DreamerV3's RSSM with a
scalable diffusion transformer and trains the policy by RL *entirely inside* the world model from
a fixed offline video dataset — no environment interaction during policy learning. It is the first
agent to obtain Minecraft diamonds from a purely offline dataset, using 100× less data than
OpenAI's VPT. The key architectural innovations are (1) a block-causal transformer world model
trained with a novel **shortcut forcing** objective and (2) an offline-imagination actor-critic
pipeline.

## Why it matters here

Three direct load-bearing intersections with our program:

1. **Our RSSM is the thing being replaced.** Dreamer 4 demonstrates what the transformer WM
   buys (scalability, long context, diffusion-quality stochasticity) and what it costs (2B params,
   256–1024 TPUs to train, H100 to run in real time). That is the boundary condition for our
   rung-3 decision: do we follow the literature upgrade or stay RSSM at home-lab scale?

2. **Shortcut forcing** is their answer to model-exploitation / imagination inflation — the
   exact family of failures we spent exps 0017–0025 debugging. The mechanism is different from
   our fixes (continue predictor + critic-on-replay + two-hot); knowing the relationship lets us
   decide if shortcut forcing would have subsumed our approach or is orthogonal.

3. **Offline-imagination-to-diamonds** is the Minecraft analog of our Crafter-diamond milestone
   ([[0007-crafter-mastery-milestone]]): an agent that reasons through a world model to reach
   high-tier achievements without live environment access. Understanding how they do it at scale
   informs whether the core design (offline dataset + imagination RL) is feasible at our scale or
   intrinsically requires the 2B-param / TPU compute band.

## Architecture — the transformer world model in detail

### State representation: from RSSM to tokens

DreamerV3's RSSM represents world state as `(h, z)`: a deterministic GRU hidden state `h` plus
a stochastic discrete latent `z` (32 categories × 32 classes). Dreamer 4 discards this
recurrent formulation entirely. Instead:

- **Causal tokenizer** compresses each frame into `Nz = 64` spatial latent tokens via masked
  autoencoding with a tanh bottleneck; trained with MSE + LPIPS perceptual losses. The result
  is nearly lossless for practical purposes.
- **Block-causal transformer** (the dynamics model) operates on the **full interleaved sequence**
  of `(latent_tokens₁, action₁, latent_tokens₂, action₂, …)` across time, with 64 spatial tokens
  per frame. The "context" at inference time is not a fixed-size hidden state but a sliding
  window of recent frames in the KV cache.
- **Context window:** ~192 frames ≈ 9.6 seconds of video at 20 FPS. Beyond this window, old
  frames are evicted from the KV cache and the model no longer has direct access — a fundamental
  difference from an RSSM whose hidden state summarizes all history (however imperfectly).

Key consequence: Dreamer 4 can attend over a rich multi-frame visual history but forgets anything
outside the 9.6 s window. DreamerV3's GRU state is more lossy but extends to arbitrary horizons.

### Attention structure

The transformer factorizes attention spatially and temporally to stay tractable:

- **Spatial layers** (3 of every 4 layers): full attention within a single frame's 64 tokens,
  no cross-frame attention; the KV cache is not grown per spatial layer.
- **Temporal layers** (every 4th layer): causal attention across time, attending to all past
  frames within the context window.
- Additional: Group Query Attention (GQA), RMSNorm, RoPE positional embeddings, SwiGLU
  activations, QKNorm, attention logit soft-capping.

This sparse-temporal factorization means the KV cache cost scales with context frames only for
temporal layers (1/4 of layers), making real-time autoregressive generation feasible: 21 FPS
on a single H100 GPU.

### Action conditioning

Actions (low-level keyboard + mouse in Minecraft) are encoded as learned embeddings and summed
into the interleaved sequence alongside frame tokens. The system learns action grounding from
a small fraction of labeled data; the majority of world-model knowledge comes from unlabeled video.
Cross-distribution generalization is strong: training actions only on the Minecraft overworld
transfers well to the nether and end dimensions.

After world-model pretraining, **agent tokens** are inserted as an additional modality via BC
finetuning — the task embedding is an additional transformer input rather than a special state.

---

## Shortcut forcing — the critical objective

### What shortcuts are being prevented

In a standard diffusion or flow-matching world model, the denoising network `f_θ` receives a
noisy version `z̃` of the target latent and outputs a prediction. When the noise level is high
(very noisy input), the model's cleanest gradient signal comes from imitating its own predictions
at lower noise levels — a form of **self-shortcutting**: the network learns to recognize its
own intermediate artifacts rather than truly predicting the future from past context. The result
is high-frequency artifacts that compound during multi-step autoregressive generation (each
predicted frame is used as the next input, so small errors amplify). This is the "prediction
shortcut" problem: the WM learns to handle its own denoising noise distribution rather than
learning accurate semantics of future states from past observations.

### The mechanism

Shortcut forcing is a generalization of diffusion forcing that conditions the denoiser on an
explicit **step size** `d` in addition to the noise level `τ`:

```
ẑ¹ = f_θ(z̃, τ, d, a)
z̃ = (1−τ)·z⁰ + τ·z¹     (linear interpolation: z⁰ = pure noise, z¹ = clean)
```

The key training protocol: for a step size `d = 2δ` (i.e., a large denoising jump), the
**target** is constructed by bootstrapping — running the model twice with step size `δ` (half
the size) from the same noisy starting point. This forces the model to be consistent across
step sizes, which prevents the shortcutting failure mode:

- **x-space prediction** (predict clean `ẑ¹` directly, not the velocity/noise vector `v̂`):
  avoids accumulating high-frequency artifacts that velocity prediction introduces, since
  x-space targets do not carry information about the exact noise realization.
- **Ramp loss weighting:** `w(τ) = 0.9τ + 0.1` — higher weight at high signal levels (small
  noise), focusing the model on fine-grained accurate prediction of near-clean states.

**Speed result:** 4 denoising steps achieve quality close to the 64-step baseline — a ~16×
speedup over standard diffusion forcing. Combined with the sparse temporal attention, this
yields 21 FPS at inference.

### How this differs from DreamerV3's losses

DreamerV3 uses: (a) KL divergence between posterior and prior over stochastic latents
(with free-bits floor to prevent KL collapse), (b) symlog reconstruction on observations and
rewards, (c) two-hot distributional critic. These are all **variational** objectives — they
regularize the latent distribution and calibrate value heads. They do NOT address the
shortcutting problem in the sense above; DreamerV3's RSSM is a one-step-ahead predictor over
a recurrent state, so the multi-step shortcutting failure mode specific to diffusion models
simply doesn't apply to it.

Shortcut forcing and DreamerV3's loss kit solve **different problems**: shortcut forcing is
a training-stability fix for diffusion-based WMs; the KL/free-bits/two-hot stack is a
calibration and distribution-regulation fix for variational RSSM-based WMs.

---

## Model exploitation: how shortcut forcing relates to our exps 0017–0025

Our imagination actor-critic experiments hit a family of failures where the actor drives the
world model into hallucinated high-reward regions:

- **Exp 0017:** missing continue predictor → actor farms imagined reward past episode termination
  (`imagined_return` 2–3 when real max is ~1.0, eval 0%).
- **Exp 0018–0019:** stochastic latents partially help but value still inflated.
- **Exps 0020–0025:** fixed by three concurrent interventions: continue predictor + DreamerV3
  critic-on-replay (β_repval = 0.3, grounds critic in real returns) + two-hot distributional
  critic (bounds reward scale) + DINO patch tokens (richer representation).

Does shortcut forcing subsume our fixes? **No — it addresses a different layer:**

- Shortcut forcing prevents **model generation quality degradation** during training (the WM's
  own denoising artifacts accumulate across rollouts). This is a failure mode of diffusion WMs,
  not RSSM WMs.
- Our exploitation failures were **policy-side**: the actor optimizing against a WM that
  (a) had no termination model, (b) was off-distribution from the actor's induced state
  distribution, and (c) had an uncalibrated value head. These are problems regardless of the
  WM architecture.
- The two approaches are **complementary**: shortcut forcing keeps the WM on-manifold during
  multi-step rollouts; critic-on-replay + continue predictor + two-hot keep the actor grounded
  to real returns from that WM.

Dreamer 4 addresses the actor-side exploitation via KL regularization to a BC prior (prevents
the RL policy from diverging too far from human demonstrations), which is RLHF-style and only
feasible when you HAVE a behavioral cloning prior from offline data. For our rung-2/rung-3
online RL setting, our critic-on-replay approach is more appropriate.

---

## Offline agent training — imagination-only RL

### Three-phase pipeline

**Phase 1 — WM pretraining:** tokenizer + dynamics model trained on 2,500+ hours of diverse
Minecraft video (labeled + unlabeled). Action grounding learned from a small labeled subset
(~100 hours of contractor data with keyboard/mouse annotations).

**Phase 2 — Finetuning with agent tokens:** behavior cloning on the labeled dataset to initialize
policy and reward heads, inserting agent tokens as additional transformer modality. Multi-token
prediction (MTP) with L=8 for policy and reward heads. Symexp twohot outputs for reward
robustness (same idea as DreamerV3's two-hot).

**Phase 3 — Imagination RL:** policy optimized by rolling out the transformer WM and training
with **PMPO** (sign-based policy gradient; ignores advantage magnitude, uses its sign) +
KL regularization to the BC prior:
- Value head trained via TD-learning with λ-returns (γ = 0.997, matching DreamerV3).
- KL constraint prevents the RL-optimized policy from moving too far from the BC baseline
  (RLHF-style stabilization, appropriate for offline-only; no live interaction to recalibrate).
- Task conditioning: ~20 Minecraft subtasks trained in parallel, each as a task embedding
  input to agent tokens; scalar reward per task.

### Long-horizon model exploitation avoidance

For the deep achievement chain (wood → planks → crafting table → sticks → wooden pickaxe →
stone → stone pickaxe → iron ore → furnace → iron ingot → iron pickaxe → diamond ore → diamond
= 20,000+ steps), model exploitation at long horizons is managed by:

1. **KL regularization to BC prior** — the RL policy can't stray far from the human-like
   behavior cloning baseline, which was itself verified on real Minecraft trajectories. This is
   Dreamer 4's primary exploitation guard (in place of our critic-on-replay).
2. **Multitask parallel training** — ~20 subtasks rather than direct diamond-from-scratch;
   the agent internalizes the tech-tree structure through reward shaping across tasks.
3. **On-manifold WM generation** (shortcut forcing keeps frames realistic) — reduces the
   risk of the policy discovering off-manifold latent shortcuts that score high reward in
   spurious regions.

**Performance:** 0.7% success rate on diamonds from offline data, beating VPT (finetuned on
labeled data) while using 100× less labeled data.

---

## Compute and data

| Item | Figure |
|---|---|
| Model size | ~2 billion parameters |
| Training hardware | 256–1024 TPU v5p chips |
| Pretraining data | 2,500+ hours of Minecraft video (~unlabeled) |
| Action-labeled data | ~100 hours of contractor video |
| Inference hardware | Single H100 GPU |
| Inference speed | 21 FPS (context: 192 frames / 9.6 s) |
| Shortcut steps | 4 (vs 64 for full diffusion) |
| Training cost | 48+ hours for ablation runs (8× RTX 3090; unofficial impl) |

Versus DreamerV3 for reference: DreamerV3 XL is ~200M params; the XS/S sizes (8–25M params)
are the home-lab viable range. Dreamer 4 is ~10× larger than DreamerV3 XL and requires TPU-v5p
clusters for training; it is about 40× slower than DreamerV3 at inference.

---

## Strategic analysis for our program

### (a) Should we migrate from RSSM to a transformer WM for rung-3/Crafter?

**Verdict: No, not at our scale — but borrow the ideas.**

The transformer WM's advantages (multi-frame context window, video-quality generation,
scalability to 2B params) are real, but they come at enormous compute cost. Dreamer 4 trains on
hundreds of TPUs and requires an H100 for real-time inference. Our local laptop has 8 GB, and a
rented ~16 GB card is our practical ceiling for bigger runs. The smallest transformer WM that
would improve on RSSM at our scale would need
extensive ablation work to identify (the Craftax/Crafter transformer WM literature — arxiv:2502.01591,
arxiv:2605.16457 — is the more relevant comparison; both are much smaller than Dreamer 4 and
explicitly target Craftax at home-lab-viable scale).

The RSSM is Dreamer 4's predecessor and is *designed* for the low-compute regime. Our
ConditionedRSSM with frozen DINO encoder is competitive with DreamerV3 on our current tasks
and has untapped headroom (Achievement Distillation, Curious Replay). Migrating to a transformer
WM before exhausting the RSSM's ceiling would be a scale result we can't match, not a sample-
efficiency gain. **File the transformer WM decision as rung-5 / rung-6 territory, or defer to the
arxiv:2502.01591 small-transformer ablation results before deciding.**

There is ONE Dreamer 4 idea that is directly portable at our scale without the compute cost:
**sparse temporal attention** (factorize space/time; do full spatial attention, sparse temporal
attention every N layers). This reduces transformer KV-cache cost significantly and is the
principle behind why Dreamer 4 achieves 21 FPS. If we ever move to a small transformer WM
at rung-3, this is the efficiency lever to reach for first.

### (b) Is shortcut forcing worth adopting for honest imagination?

**Verdict: Not applicable to our RSSM; the underlying concern IS relevant, but our fix is different.**

Shortcut forcing is specifically a training-stability fix for **diffusion-based dynamics models**
that predict via iterative denoising. Our RSSM is a deterministic-path GRU + stochastic discrete
latents trained with a variational (KL) objective. The denoising-shortcut failure mode doesn't
apply.

That said, the underlying concern — that a world model learns to exploit its own inductive biases
rather than accurately predict the future — is the same family of problem we hit in exps 0017–0025.
Our fixes (continue predictor + critic-on-replay + two-hot) address this at the **actor-critic
calibration** level, which is the right layer for an RSSM. Keep these; they are principled
interventions confirmed in our setting.

### (c) What does offline-imagination-to-diamond imply for our Crafter-diamond milestone (ADR-0007)?

**Verdict: Strongly validates the imagination-RL thesis; reveals a scale gap; surfaces a portable structural lesson.**

Dreamer 4's offline diamond result IS the Minecraft proof-of-concept for what ADR-0007 asks us
to do in Crafter. The structural parallel is tight:

| Dreamer 4 (Minecraft) | Our program (Crafter, ADR-0007) |
|---|---|
| 2,500 h offline video | Online collection, no video corpus needed |
| BC prior → imagination RL | Actor-critic in imagination, online |
| 20 subtasks (tech-tree shaping) | Directable competence (per-achievement conditioning) |
| ~20,000 actions for diamond | Tech-tree: ~10 achievements in sequence |
| Text-conditioned task tokens | Dynamics manual + goal conditioning (§1–3) |

Three lessons that are directly applicable at our scale:

1. **Multitask parallel training through the achievement hierarchy is load-bearing.** Dreamer 4
   uses ~20 parallel subtasks toward diamonds; this is the same structural requirement as our
   Achievement Distillation direction (arxiv:2307.03486). The deep tree won't come from a single
   aggregate-score objective; it needs per-step achievement conditioning. This is additional
   motivation for implementing directable-by-task conditioning (ADR-0007, mechanism 2).

2. **The KL-to-BC prior is their exploitation guard.** For us, online RL means the BC prior is
   unavailable (or very weak). Our critic-on-replay serves the same stabilizing function at a
   lower cost; keep it as the primary exploitation guard. If we ever add an offline dataset,
   revisit a KL term.

3. **The context-window problem.** Dreamer 4's 9.6-second / 192-frame window is actually SHORT
   for Minecraft's deep tech tree. The authors note the model forgets beyond the window; the agent
   must rediscover wood/stone/iron each episode because it can't attend to actions from 10 minutes
   ago. Our RSSM state is lossier per step but can summarize arbitrary-length history through the
   GRU. This is one place the RSSM's recurrence is genuinely superior for long-horizon tasks at
   fixed compute; it's a real trade-off, not merely a scale issue.

**Feasibility at home-lab scale:** The offline-imagination recipe is NOT feasible at our scale
in the Dreamer 4 form — the 2B-parameter WM trained on thousands of GPU-hours of video is the
load-bearing component. But the PRINCIPLE is feasible: a small RSSM trained online on Crafter
episodes (our existing stack) + achievement-conditioned imagination RL IS the same program at
smaller scale. The delta is the reading channel (dynamics manual) which Dreamer 4 doesn't have.
That reading channel is our moat — it's the bet that reading replaces compute-scale (a small
model that reads the rules beats a large model that discovers them by trial-and-error).

---

## Open questions

- Does shortcut forcing apply to discrete-latent diffusion variants (e.g., a transformer over
  RSSM-scale discrete tokens)? If so, could it replace the KL/free-bits objective while adding
  multi-step generation quality?
- Dreamer 4's context window forgets history beyond 9.6 s — for Crafter's deep tech tree (~10-min
  episodes), would a hybrid (transformer-for-spatial + GRU-for-history) outperform a pure
  transformer or pure RSSM?
- The PMPO (sign-based advantages) seems like a simpler alternative to the entropy-regularized
  REINFORCE we currently use. Worth evaluating at rung-3 scale?

## Links

[[imagination-training]] · [[dreamerv3-2023]] · [[0007-crafter-mastery-milestone]] ·
[[rung4-manual-conditioned-agent]] · [[0017-imagination-actor-critic]] · [[0018-continue-predictor]] ·
[[0021-critic-on-replay]] · [[0023-twohot-value]] · [[0025-reward-head-patch]] ·
[[environment-ladder]] · [[compute-strategy]] · [[hierarchy-and-credit]]
