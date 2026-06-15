---
status: draft
owner: world-model
scope: local
verified: false
last_reviewed: 2026-06-15
---

# Validated-reading reward — intrinsic motivation to test what you read

**Status: draft proposal (design discussion 2026-06-15).** A candidate mechanism for the rung-4
grounding wall. Motivates [[0048-rtfm-validated-reading]].

## The problem this addresses

[[0040-rtfm-actor-conditioning]] localized the durable rung-4 wall to an **objective/identifiability
gap**, not architecture or reading fidelity: `swap_follow ≈ 0` everywhere (the agent never performs a
*swapped* manual's recipe), because collection is always `ManualMode.CORRECT`, where "obey the
displayed manual" and "earn the reward" are the **same policy** — nothing pressures the agent toward
obedience *per se*. The reading is genuine (WM reconstructs the manual, `inv_ratio>1`, oracle pct
0.88); the actor just never has to route it into action. Every lever since (MPC 0044, oracle probe
0045, sampled rollouts 0046, conservative reward 0047) attacked *execution/calibration* — the wrong
axis. The conservative-reward thread (0047) is returning negative, which is the wall-relocation
tripwire: stop calibrating, attack the objective.

## Core idea

Reward the agent **intrinsically for confirming, in the real environment, a prediction it derived from
reading the manual**:

> read the manual (theory) → the WM forms a prediction → **act to test it** → reward iff *reality*
> confirms.

This makes "the manual's content was worth acting on" the thing that pays — the agent learns to trust
and follow manuals because doing so *predicts reality*, which is exactly the read→act grounding that
`swap_follow` measures.

## The wireheading constraint (load-bearing)

The reward must be **un-fakeable**: gated on the real environment's response, which the agent does not
control. "My manual-derived prediction matched the *real* next state" is safe; "I declare I succeeded"
is wireheadable (a dopamine button). This is the RND pattern ([[rnd-2018]]) — a fixed external judge —
with the real env as the judge, and the structural core of Marino et al.'s hypothesis-verification
([[marino-hypothesis-2020]]).

## The mechanism: marginal predictive value, validated

Not "reward accurate predictions" (raw) but **reward the manual's *marginal* contribution to real
prediction accuracy**:

```
r_intrinsic(s, a, s'_real) = clip_≥0(  pred_error(s, a, s'_real | NULL/wrong manual)
                                      − pred_error(s, a, s'_real | correct manual)  )
```

i.e. how much *less* wrong the manual-conditioned WM was about the *real* transition than a
null/shuffled-manual baseline. This is a VIME-style information-gain reward ([[vime-2016]]) applied
**contrastively** across the manual condition.

**Why marginal (not raw) — it dodges both classic traps automatically:**
- **Dark room** (reward accurate predictions → seek trivially predictable states): in a boring state
  both the with-manual and without-manual models predict equally well → marginal ≈ 0 → no reward.
- **Noisy TV** (reward prediction error → seek noise): in a stochastic state both models fail equally
  on the noise → marginal ≈ 0 → no reward (holds as long as the manual concerns the *deterministic*
  part of dynamics). This emerges from the marginal framing without a separate aleatoric estimator
  (cf. [[rnd-2018]], and Oudeyer-Kaplan learning-progress: reward error *reduction*, not low error).

## Why it breaks the identifiability gap

Under a **wrong/swapped** manual the manual-conditioned prediction is **disconfirmed** by reality →
no intrinsic reward. Under the **correct** manual → confirmed → reward. In correct-mode training
manuals always pan out, so the agent learns "manuals reliably predict reality → read and follow them."
At a swapped eval it then follows the *displayed* manual (it has no way to know it is swapped without
acting) → `swap_follow` rises. Crucially this is **anti-baking-clean by construction**: the reward is
grounded in real prediction-confirmation, never a privileged label, so it can't be gamed by memorizing
the answer — reality is the incorruptible judge. It also needs **no swapped-mode collection** (unlike
the dropped extrinsic-shaping plan, which flirted with teaching the test).

## Building blocks we already have

- **Manual-invariance diagnostic** (`manual_aux.manual_invariance_diagnostic`, `inv_correct` vs
  `inv_wrong`) is *exactly* the two contrastive predictions — today a passive readout; this proposal
  turns it into a per-transition reward, validated against the **real** next embedding (not a
  reconstruction target).
- Frozen DINO encoder + `ConditionedRSSM` give the forward prediction; computed at **collection** time
  (real transitions) and added as an intrinsic reward channel.

## Literature (scout 2026-06-15; lit gate for [[0048-rtfm-validated-reading]])

[[vime-2016]] (IG reward to specialize) · [[marino-hypothesis-2020]] (act-to-verify, real-env judge —
closest neighbor; lacks reading + marginal) · [[icm-2017]] (foil: rewards error, inverted sign) ·
[[rnd-2018]] (anti-wirehead pattern) · Oudeyer-Kaplan learning-progress (dark-room fix). Novelty:
the full combination appears relatively novel (moderate confidence; a citation-walk from
[[marino-hypothesis-2020]] is the publish-grade check).

## Open risks

- **Residual hackability** — even marginal IG can be gamed if the agent influences what it predicts;
  detach the right tensors (as with the exp-0047 conservative head).
- **Deterministic-dynamics assumption** for the noisy-TV property; verify our recipe dynamics qualify.
- **Reward scale / weighting** vs the sparse tutorial reward; and whether to anneal (likely keep on —
  it is anti-baking-clean, unlike the env shaping).

## Why it matters beyond this benchmark

An agent intrinsically motivated to **test what it reads against reality** is doing science — read a
hypothesis, test it, get the dopamine when it holds. That is a *capability*, not a benchmark patch,
and it directly serves the world-model thesis (a model that predicts, then acts to validate). See
[[hierarchical-imagination-agent]] §5 (calibrated confidence) and [[language-grounding]].
