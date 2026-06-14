---
status: draft
owner: agent
scope: local
sources: [arxiv:2308.01399]
verified: true
last_reviewed: 2026-06-14
---

# Dynalang: Learning to Model the World with Language (Lin et al., 2023)

**Lab:** UC Berkeley (Abbeel, Dragan groups) · **Code:** https://dynalang.github.io ·
**Read state:** read (HTML full-text, two passes; quantitative Messenger numbers
unavailable — paper reports learning curves only, no final-performance table)

**Authors:** Jacqueline Yau Lin, Yuqing Du, Olivia Watkins, Danijar Hafner, Pieter Abbeel,
Dan Klein, Anca Dragan

**Venue:** preprint (arXiv:2308.01399, Aug 2023; submitted to NeurIPS 2023 workshop track).

## One-paragraph summary

Dynalang extends a DreamerV3-style RSSM to be a **multimodal world model over both video
frames and language tokens**. Language enters the model TOKEN-BY-TOKEN, interleaved with
observations over real time (one token per timestep, padded with zeros when unavailable).
The world model is trained to reconstruct — and imagine forward — BOTH image patches AND
language tokens. This reconstruction objective on text is the paper's central contribution:
it gives a **dense self-supervised gradient on language tokens** that flows even in timesteps
where task reward is zero, allowing the world model to learn grounded language representations
without requiring dense reward shaping or curriculum learning. The actor-critic is then
trained entirely inside imagination, exactly as DreamerV3. Dynalang beats EMMA on Messenger
(qualitatively; no published table), and demonstrates transfer across HomeGrid (gridworld
hints), VLN-CE (vision-language navigation), and LangRoom tasks.

## Why it matters here

This is the **closest published architecture to our rung-4 agent**: frozen encoder + RSSM +
language in imagination + evaluated on Messenger (the referent-swap benchmark). The
**auxiliary text-prediction loss is the key grounding ignition mechanism** we are trying to
understand for HO-0007 (the sparse-reward ignition wall we hit in exps 0032/0033/0034). It
is the direct alternative to our current approach of reward shaping for ignition. See
Critical analysis section below for why the transfer to our static-manual setting is
non-trivial.

## Details

### 1. Architecture: how language enters the RSSM

The overall latent structure is identical to DreamerV3's RSSM:
- **Deterministic state** `h_t` (recurrent, GRU-based)
- **Stochastic discrete code** `z_t` (one-hot categoricals, sampled from the encoder
  posterior)

The RSSM has three functional components:

**Sequence model (prior / dynamics):**
```
ẑ_{t+1}, h_{t+1} = seq(z_t, h_t, a_t)
```
This predicts the next stochastic code given the current belief and action. Language does
NOT enter here directly — the sequence model is action-conditioned only.

**Multimodal encoder (posterior):**
```
z_t ~ enc(x_t, l_t, h_t)
```
The image frame `x_t` and language token `l_t` are fed JOINTLY through an MLP encoder
(alongside the previous recurrent state `h_t`) to produce the posterior `z_t`. Image is
processed first through a strided CNN; language is embedded; both embeddings are
concatenated to `h_t` and passed to the MLP. Conditioning is by **concatenation, not
cross-attention**.

Key detail: **one token per timestep** — language is interleaved with observations over real
time. At timestep `t`, the agent receives exactly one frame AND one token (or a zero-padding
token when no text is available). For Messenger, the game manual appears as a stream of
tokens spread across the timesteps in the episode opening.

**Text encoder (per environment):**
- HomeGrid: one-hot token encodings (small vocabulary, learned from scratch)
- Messenger and VLN-CE: **T5-small** frozen embeddings — "the last hidden layer
  representation for each token" (not pooled; per-token representations)

**Multimodal decoder:**
```
x̂_t, l̂_t, r̂_t, ĉ_t = dec(z_t, h_t)
```
The decoder takes the CURRENT belief state `(z_t, h_t)` and produces reconstructions of the
image, the current language token, the reward, and the episode continuation flag. All decoder
heads are MLPs (except image which is a strided CNN decoder).

### 2. Auxiliary future-text-prediction loss — the grounding mechanism

The "future text prediction" name in the paper is slightly misleading. What is actually
predicted is the **current token** `l_t` reconstructed from the current belief state
`(z_t, h_t)`. The "future" is implicit via the RSSM's forward dynamics: the prior
`ẑ_{t+1}` must predict the next representation that will be corrected into `z_{t+1}` when
the next token arrives, so forward prediction of the next BELIEF implicitly encodes forward
prediction of the next language token.

**Loss formula:**

Total world model loss (ℒ_WM):
```
ℒ_WM = ℒ_repr + ℒ_pred
```

Representation (encoder) loss:
```
ℒ_repr = -log p(x_t | z_t, h_t)        [image recon, MSE]
        + catxent(l̂_t, l_t)             [language recon, cross-entropy]
        + catxent(twohot(r_t), r̂_t)     [reward recon, two-hot CE]
        + bce(ĉ_t, c_t)                  [continuation, binary CE]
        + β_reg · max(1, KL[z_t || sg(ẑ_t)])   [encoder KL, β_reg = 0.1]
```

Prediction (prior) loss:
```
ℒ_pred = β_pred · max(1, KL[sg(z_t) || ẑ_t])   [β_pred = 0.5]
```

**Loss weighting:** the language reconstruction term has **no explicit beta coefficient** —
it appears with unit weight equal to the image reconstruction term. There is no published
ablation on the language loss weight.

**What gives the dense gradient:** The language reconstruction loss `catxent(l̂_t, l_t)`
fires at EVERY timestep that carries a language token (i.e., throughout the manual reading
phase at episode start). This is decoupled from task reward, which is sparse (Messenger:
sparse binary win/lose). The gradient flows from the language prediction head back through
`(z_t, h_t)` into the RSSM, pushing the latent state to encode language content. In
contrast, the reward signal flows only at rare episode-terminal or rule-triggered events.
The language loss is therefore **N tokens of dense signal per episode opening**, vs O(1)
reward signals per episode.

### 3. Training regime

**No pretraining required for core results.** The default training is online RL with a
single replay buffer containing multimodal experience (frames + tokens + actions +
rewards), training the WM and actor-critic jointly.

**Optional text-only pretraining:** The paper describes a pretraining option where image
and action inputs are zeroed out and the model trains purely on text sequences (in-domain
manuals or TinyStories at 500M tokens scale). The actor-critic is re-initialized after
pretraining. This is evaluated on Messenger/VLN-CE but appears optional — the method works
without it on simpler tasks. When used, pretraining provides a head-start on language
representations, reducing the "cold start" where the RSSM has not yet learned to encode
text.

**No curriculum is described.** Dynalang does not report curriculum learning as a required
ingredient. The dense text-prediction loss is the mechanism that makes end-to-end learning
tractable without curriculum — the gradient is dense regardless of task reward structure.
This contrasts sharply with RTFM (curriculum REQUIRED) and Messenger/EMMA (three-stage
curriculum REQUIRED).

**Data source:** Purely online environment interaction. No human demonstrations.

### 4. Messenger results and generalization

**Messenger task:** The agent navigates a 2D grid to find a "messenger" entity (one of
several), retrieve a message, deliver it to a "goal" entity, and avoid enemies. The
critical detail: **entity roles and movement dynamics are randomized per episode** — which
entity is the messenger, goal, or enemy is specified in a natural-language manual given at
episode start as a sequence of token-by-token rules. The agent must read and ground the
manual to know what to approach/avoid. Stage 1 (S1): 1-step reasoning; Stage 2 (S2):
2-step; Stage 3 (S3): most difficult multi-step reasoning.

**Dynalang vs EMMA (from paper, Fig. 6):** Dynalang "outperforms EMMA, IMPALA, and R2D2"
and "learns more efficiently than EMMA." No final performance numbers are tabulated; all
comparisons are learning curves. IMPALA and R2D2 "achieve non-trivial performance only on
earlier stages," implying they fail S3. Dynalang reaches visually ~2× EMMA's asymptote on
S3 from the figure.

**Generalization / referent swap:** The paper does NOT report a referent-swap evaluation.
Messenger's built-in difficulty IS compositional generalization — entity role names randomize
per episode, so there is structural generalization implicit in the training distribution —
but no held-out "swapped manual" ablation analogous to our swap_follow metric is reported.
The EMMA paper (2101.07393) introduced the swap-entity eval; Dynalang does not replicate it.

**HomeGrid:** Five task types (find, get, clean up, rearrange, open) across 38 tasks.
Three hint categories (future observations, dynamics, corrections). After 50M steps,
Dynalang improves when language provides diverse hints; a baseline WITHOUT language
grounding "degrades" (language without grounding hurts). Quantitative: no table.

**VLN-CE (Vision-Language Navigation):** "Succeeds at significantly more instructions"
than R2D2 baseline but "is not yet competitive with state-of-the-art" navigation methods.
No exact SPL/SR numbers published.

### 5. The dense-gradient mechanism — why text prediction ignites grounding under sparse reward

The mechanism has two parts:

**(a) Decoupling language learning from task reward.** By making the WM predict text tokens
(a self-supervised objective), the WM receives gradients about language content at every
episode timestep that carries a token, regardless of whether the agent achieves anything.
In a sparse-reward task like Messenger (binary win/lose per episode), the task gradient is
O(1 gradient signal per episode); the language gradient is O(|manual| gradient signals per
episode). With long manuals, this is a 10-100× denser signal.

**(b) The latent state is SHARED.** The belief `(z_t, h_t)` must simultaneously predict
text tokens (dense supervision) AND reward (sparse). The KL regularizer (β_reg = 0.1)
keeps the prior prediction aligned with the posterior. So the dense language gradient
trains the same latent that the policy imagination depends on. When the belief has been
pushed by text reconstruction to encode manual content, the actor-critic can exploit that
structure through imagination — the grounding "propagates" to behavior via imagination.

**(c) It eliminates the cold-start problem.** Without the text loss, the RSSM has no
incentive to represent language until a sparse task signal connects behavior to language.
In practice that means the RSSM ignores language (it's "free" to zero out the language
encoder), and the agent never learns to read. The text reconstruction loss makes ignoring
language costly even before any reward arrives.

## Critical analysis for our static-manual setting

**CRITICAL NOTE — static vs streaming language is the key mismatch.**

### The mismatch

Dynalang's language is **streaming**: tokens arrive one per timestep, interleaved with
observations over real environment time. This means "predict the next text token" is
well-defined: at timestep `t`, the model predicts `l_t` from `(z_t, h_t)` where `h_t`
was already conditioned on `l_1, ..., l_{t-1}` in previous steps. Prediction is causal
and non-trivial.

Our rung-4 architecture conditions on a **static manual**: the entire manual is given ONCE
at episode reset, encoded via cross-attention (a frozen text encoder produces token
embeddings M; the RSSM's dynamics head attends to M). The manual does not stream through
the RSSM over time. This breaks the "predict the next text token" objective in two ways:

1. **Reconstruction from belief is trivially solved by copying the cross-attention input.**
   If M is already available as a cross-attention key-value, the decoder can attend to M
   and output the correct tokens with near-zero gradient. The loss collapses to zero
   without the RSSM having learned anything about the MEANING of the manual — it just
   routes M back through the decoder. A static-manual reconstruction loss is potentially
   circular/trivial given the manual is already an input.

2. **There is no temporal arrival structure to predict.** Token-by-token streaming creates
   a well-posed causal prediction problem. A static manual has no "next token" to predict
   from temporal context.

### Candidate analogs (what we'd actually implement)

Four options exist for adapting the Dynalang dense-gradient insight to a static-manual
setting. They vary in tractability and triviality risk:

**(A) Masked manual reconstruction from belief (our preferred candidate).**
After reading the manual and running some environment steps, require the belief `(z_t, h_t)`
to reconstruct **masked-out portions of M** WITHOUT attending to M during reconstruction
(the decoder head is blocked from the cross-attention keys). This forces the belief to have
encoded the manual content into `h_t`/`z_t` internals, not just routing M to output.
Implementation: a masked-language-modeling head on the belief state, M is masked at decode
time. Risk: if the mask fraction is small or masks easy tokens, the loss is trivially solved
by shallow pattern matching to unmasked tokens. Mitigation: mask high-information spans
(the recipe gesture words) and evaluate grounding by measuring whether the probe FAILS when
manual content changes (the mechanistic test in [[rung4-manual-conditioned-agent]] §4c).

**(B) Predict manual-implied dynamics from belief.**
Rather than predicting text tokens, train the WM to predict the OUTCOME of applying the
recipe: given belief `(z_t, h_t)` after seeing state `s_t`, predict whether gesture `g`
will be rewarded in this episode. This is a direct supervision on the dynamics the manual
encodes. Implementation: sample `(belief, gesture)` pairs from replay, predict the
manual-specified outcome. Reward: trains the same pathway as the RL signal. Risk:
this IS essentially a second reward signal, which might reintroduce the reward-farming
confound we fixed in 0034.

**(C) Predict masked manual tokens WITH cross-attention from embedding ONLY (no belief
shortcut) — ablated teacher.**
Provide the decoder with a CORRUPTED manual (e.g., masked or shuffled tokens) and train to
reconstruct via the belief. The key constraint: during the reconstruction loss backward
pass, the cross-attention gradient is stopped (stop-gradient on M). This makes the
reconstruction path go through the belief only. Risk: expensive; gradient stopping has
subtle effects on learned representations.

**(D) Adversarial swap signal (maintain the referent-swap as an auxiliary training loss,
not just an eval).**
During training, occasionally feed the WRONG manual and penalize (via a contrastive or
triplet loss) if the belief is indistinguishable from the correct-manual belief. This
directly optimizes for the referent-swap test. Risk: requires paired (correct, swapped)
experience during training, which may need extra env rollouts or offline samples.

### Our current situation and recommendation

In exps 0032/0033/0034 we discovered two sequential blockers: (1) the vision shortcut
(agent read the staged state, not the manual), fixed by `one_shot` in HO-0006; (2) a
pure sparse-reward ignition wall (0–6 reward events/60 timesteps, no gradient). HO-0007
is the open requirement for a denser training signal.

**Dynalang's streaming text-prediction does NOT transfer directly** because our manual is
static and cross-attention already provides M to the model. The Dynalang lesson that DOES
transfer is the design principle: **separate language understanding supervision from task
reward**, using a loss that fires on every timestep where language is relevant.

**Recommended adaptation (candidate for next experiment):** option (A) — masked manual
reconstruction from belief with blocked cross-attention at decode time, combined with
curriculum shaping (HO-0007). The masked reconstruction gives a dense gradient during the
episode (it can be computed at every belief state update, not just at episode reward),
while the blocked cross-attention prevents the trivial copy-through solution. This
preserves the anti-baking guarantee of [[rung4-manual-conditioned-agent]] §2 (the belief
must encode dynamics content to reconstruct masked tokens, which is exactly what grounding
means), and it does not reopen the vision-shortcut or reward-farming confounds caught by
0033/0034.

**Triviality risk of option (A)** is real and must be measured: compute a manual-invariance
control (same belief, shuffled/wrong manual reconstruction accuracy). If reconstruction
accuracy is high regardless of which manual was given, the loss is trivial and alternative
(B) or (D) should be tried.

## Open questions

- Does masked manual reconstruction (option A) genuinely prevent the copy-through shortcut,
  or does the belief simply memorize M byte-by-byte? Needs the mechanistic probe.
- How does the Dynalang text-pretraining interact with frozen text encoders (our design)?
  Dynalang fine-tunes T5-small embeddings implicitly via the representation loss; our frozen
  encoder does not update. Does this weaken the dense text gradient?
- Does any published paper implement a masked-manual-reconstruction auxiliary loss in a
  static-conditioned WM? (Check LED-WM (2511.22904) — it may be the closest analog.)
- What is Dynalang's performance on S3 Messenger in absolute numbers? The paper's learning
  curves don't give this; needed for comparison when we have our own Messenger results.

## Links

[[rung4-manual-conditioned-agent]] · [[language-grounding]] · [[imagination-training]] ·
[[0032-rtfm-grounding]] · [[0033-rtfm-length1-grounding]] · [[0034-rtfm-oneshot-ignition]] ·
[[dreamerv3-2023]] · [[grounding-env-spec]] · [[concepts/language-grounding]]
