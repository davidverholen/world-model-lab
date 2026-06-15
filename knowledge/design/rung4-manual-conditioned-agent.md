---
status: draft
owner: world-model
scope: local
verified: false
last_reviewed: 2026-06-14
---

# Rung 4: manual-conditioned world model (reading-to-learn-dynamics)

**Status: proposed design, for review.** The agent side of the grounding
program; consumes the crafter-rtfm env ([[grounding-env-spec]]). Builds on the validated
rung-3 foundation: frozen DINO encoder + RSSM, both shown to carry full tech-tree perception
end-to-end ([[0031-semantic-foundation-probe]]).

## 1. Goal and the one failure mode that defines success

**Goal:** an agent that, each episode, **reads a manual describing that episode's dynamics**
(recipes / referents randomized per episode, hidden from pure exploration) and acts on it —
*reading-to-learn-DYNAMICS*, not instruction-following (text = the rules, not text = the goal).

**The load-bearing risk — "baking" (the anti-baking constraint).** The trivial-but-wrong
solution is an agent that, during training, **compiles manual content into its weights** — a
fixed manual-token → behavior mapping — so it *appears* to use the manual but never actually
reads at test time. A baked agent succeeds on training manuals and **fails on a novel or
swapped manual.** Avoiding this is not a nice-to-have; it *is* the experiment. The entire
design is organized around making baking impossible to hide and unrewarding to learn.

The unfakeable definition of success (the **referent-swap** gold standard): given a **swapped**
manual, a *reading* agent follows the swap (does the swapped thing); a *baked* agent ignores it
(does the memorized/default thing). Success metric is this behavior, **not** task reward.

## 2. Architecture — condition the WORLD MODEL, not the policy

```
manual text ─► [frozen text encoder] ─► M (TOKEN embeddings, not a pooled vector)
                                          │  (cross-attention — order/token-aware; pre-check §2)
frame ─► [frozen DINO] ─► embed ─► [RSSM dynamics | conditioned on M] ─► belief
                                          │                                  │
                                   reward head(·|M)                     [actor]  ← plans through
                                                                                   the M-conditioned WM
```

- **Frozen visual encoder (DINO)** — kept; validated ([[0031-semantic-foundation-probe]]).
- **Text encoder — frozen, TOKEN-LEVEL (not a pooled sentence vector).** A frozen pretrained
  text encoder keeps the frozen-foundation thesis ([[frozen-encoder-lean]]) and resists
  overfitting to the templated vocabulary. **Pre-check finding (2026-06-14, `world_model.text_probe`):
  a POOLED MiniLM-L6 sentence vector is INSUFFICIENT** — on held-out r1 recipe manuals a linear
  probe decoded the (verbatim-in-text) recipe gesture at only 0.23–0.37 (10-class, chance 0.17);
  token-level mean+max lifted it to 0.42–0.55, confirming the info is in the TOKENS and pooling
  dilutes it. So we use **token embeddings**, and — because a recipe is an *ordered* gesture and
  even mean+max is order-invariant — the conditioning must be **order/token-aware** (see below),
  not a single pooled code.
- **The manual conditions the DYNAMICS (RSSM) and the reward head — NOT (primarily) the
  policy.** This is the central, deliberate choice and the strongest anti-baking lever:
  - *Why it's the grounded choice.* The manual is information about *how the world works* this
    episode. Reading it should change what the agent *predicts will happen* (`stone + sword →
    stone breaks`), and the policy is then derived by **planning through the updated model**
    (imagination AC). That is literally "read-to-learn-dynamics."
  - *Why it resists baking.* A policy conditioned on text can learn `token X → action Y`
    shortcuts. But for a *world model* to shortcut, it would have to correctly **predict the
    per-episode-randomized dynamics from the manual** — which *is* reading. A WM that predicts
    the right episode-specific dynamics from the manual has, by construction, grounded the text.
    This also gives us a **direct mechanistic probe** (§4c): does the m-conditioned WM predict
    the correct dynamics?

## 3. Anti-baking is layered — no single layer suffices

Baking is defeated only by stacking env + eval + diagnostic + architecture. Each alone is
gameable; together they leave no hiding place.

**(a) Env-side — memorization is made structurally useless.** crafter-rtfm randomizes the
recipe/referent **per episode** over a large/effectively-infinite space, and gates the
achievement behind mandatory reading (no-text agent provably fails — the rule is *hidden from
exploration* and not cheaply discoverable by trial-and-error). So a memorized manual→behavior
map has *negative* value; the only policy that works across episodes is one that reads. We
**rely on** this and must not build an agent-side backdoor that circumvents it (see (d)).

**(b) Eval-side — the referent-swap + held-out generalization, as the success metric.** Four
modes (correct / none / swapped / shuffled). Grounded iff: `correct ≫ none` (reading is used)
**AND** behavior **follows the swapped manual** (reading, not memorizing). Crucially, **test on
HELD-OUT manual configs** (recipes/referents never seen in training) — generalization to novel
manuals is the proof the *reading skill* was learned, not the training manuals. (This mirrors
crafter-rtfm handoff [[HO-0004]] — the same bar, now on a world-model agent.)

**(c) Diagnostic-side — probe the belief for the manual's rule (our 0029/0031 method).** After
reading "stone needs the sword," can we linearly decode from the belief / the WM's predictions
that the agent *expects* sword→stone (and does NOT under a swapped/absent manual)? A probe that
tracks the manual content is mechanistic evidence of reading; a probe that's invariant to the
manual is evidence of baking. Cheap, and it catches baking the reward metric might miss.

**(d) Architecture-side — no shortcut that bypasses reading.** Two guards: (i) the manual is the
*only* viable source of the episode rule (env enforces; the agent must not be handed the rule
through another channel, e.g. a leaked label or a within-episode oracle); (ii) condition the WM,
not the policy (§2), so "using the manual" means "predicting the right dynamics," not "matching
a token to an action." Within-episode belief updating from *observations* is fine and good — the
env just makes the rule un-discoverable that way, so reading remains necessary.

## 4. Training / eval protocol

- **Train** on a distribution of episodes, each with a freshly-sampled manual+rule; the agent
  experiences many configs but never enough repetition to memorize any one (the space is large).
- **Eval** on **held-out configs** across the four harness modes; report the swap-following rate
  and the correct-vs-none gap as the headline, plus the belief-probe (§4c) as mechanistic backup.
- **Staged, matching crafter-rtfm's mechanism ladder** (start where reading is simplest to learn,
  add difficulty): R0World (world-conditioned) → R1 (randomized recipes) → R2 (referent-swap).
  Don't move up a rung until the swap test passes on the current one.

## 5. What we need from crafter-rtfm — DELIVERED (HO-0005, accepted)

Opened and **accepted** via the commons handoff layer (HO-0005, resolving_commit 6f3d71e@crafter-rtfm;
verified by an independent consumer smoke). Confirmed: `obs["manual"]` raw text (fixed per episode);
`info["manual_facts"]` (true rule) + `info["displayed_facts"]` (shown rule) — privileged, no leak
into obs; the four-mode harness as a programmatic eval API; **content-disjoint** train/test splits
(`splits.split_seeds`, 0/0 rule overlap, ~1972 configs); seed reproducibility; no-text-fails gates
for R0World/R1. (R2-*mechanism* deferred — the SWAPPED *mode* on r1 supplies the swap eval we need.)

## 6. Open questions (for review)

- **Text encoder + conditioning — RESOLVED by the §2 pre-check.** A pooled sentence vector is out;
  use frozen **token embeddings** + **cross-attention** (order/token-aware), since the recipe is an
  ordered gesture pooling destroys. Open sub-question: which frozen encoder's tokens (MiniLM is
  enough so far; DistilBERT/T5-small are token-level alternatives) — decide once cross-attention is
  wired and the swap test runs end-to-end.
- **Does WM-conditioning alone suffice, or does the actor also need `M`** for tractable planning?
  Default: WM-only first (maximally baking-resistant); add policy-conditioning only if planning
  through the m-conditioned WM is too weak, and re-verify the swap test if we do.
- **Curriculum:** is there a reading-easy → reading-hard ordering that avoids the agent first
  learning a no-read baseline it then has to unlearn?

## Links

[[grounding-env-spec]] · [[0031-semantic-foundation-probe]] · [[frozen-encoder-lean]] · [[no-hardcoded-env]] · [[environment-ladder]] · [[architecture-strategy]]
