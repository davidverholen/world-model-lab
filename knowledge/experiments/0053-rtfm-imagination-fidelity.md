---
status: draft
owner: world-model
scope: local
verified: true
last_reviewed: 2026-06-16
---

# 0053: imagination-fidelity probe — is the ~0.10 wall the world model, or downstream?

After the lever ladder (objective→folded-VR 0.10, coef-saturate, 2× compute, hierarchy ~0,
[[0052-rtfm-decoupled-vr-head]] collapse) showed that pushing the **grounding incentive** harder keeps
failing, the question turned from "make grounding stronger" to **"where does the ~0.10 length-2
swap_follow wall actually live?"** This is the wall-relocation pivot (PROCESS §Efficiency guardrails:
diagnose twice, then localize/build). exp0053 builds the diagnostic capability rather than trying
another reward variant.

## Method — imagination vs reality, under the dream's own plan

Tool: `scripts/imagine_report.py`. From a SHARED real start state `s0` (belief right after the manual
is read), (a) roll the greedy actor forward in **imagination** (`rssm.img_step`), then (b) replay that
exact imagined action plan in the **real env** (`rssm.obs_step`) and compare the belief trajectories.

- **Divergence** `||belief(imagined s_k) − belief(real s_k)||`, anchored at 0 (shared s0).
- The RSSM belief carries a **sampled** stochastic latent, so a perfect model still shows nonzero
  divergence. Control: a **real-vs-real sampling floor** (a 2nd real pass over the SAME observations,
  z resampled). imagined-vs-real ABOVE the floor = genuine open-loop model error.
- Reported on the full belief (h+z, what the policy/heads consume) AND the **deterministic h** alone
  (where recurrent-dynamics error shows cleanly, un-inflated by per-step z sampling).
- Also a **side-by-side GIF**: imagined path (imagined belief → nearest real frame by retrieval; the
  WM predicts latents, no pixel decoder) vs reality under the same actions — you *see* the divergence.

Run on the working baseline ([[0051-rtfm-flat-vr-optimization]] coef-0.5, ~0.10) and the collapsed
[[0052-rtfm-decoupled-vr-head]] model, CORRECT + SWAPPED, 16 seeds, horizon 8 (single checkpoint s0).

## Result — the world model imagines FAITHFULLY; the wall is downstream

Deterministic-h divergence at step 8 (imagined-vs-real ÷ sampling floor; 1× = perfectly faithful):

| model | CORRECT | SWAPPED |
|---|---|---|
| baseline (exp0051, ~0.10) | **1.5×** (cos 0.90) | **2.7×** (cos 0.84) |
| collapsed (exp0052, VR-head) | 1.9× (cos 0.84) | 2.0× (cos 0.79) |

1. **Imagination is reasonably faithful for both** — only ~1.5–2.7× the inherent sampling floor;
   neither hallucinates wildly in latent space. **The ~0.10 wall is NOT an open-loop world-model
   accuracy failure.** The bottleneck lives downstream of the dynamics — policy / credit-assignment /
   reward-readout. (Corroborates the lever-ladder meta-pattern.)
2. **The baseline WM genuinely conditions imagination on the manual** — it diverges *much* more in
   SWAPPED (2.7×) than CORRECT (1.5×). If it ignored the text the two would match; the asymmetry is a
   real WM-level reading signal.
3. **The collapsed model lost that asymmetry** (1.9× ≈ 2.0×) — its imagination is roughly
   manual-*insensitive*, mirroring the manual-blind policy collapse. The exp0052 VR-head collapse muted
   the WM's manual-conditioning, not only the policy.

## Caveats

Single checkpoint (s0) per model, 16 eval seeds, horizon 8, divergence in the RSSM latent (DINO-derived)
space. The **direct** "actor chases fantasy reward" test is the imagined-vs-real **reward gap**, which
needs the `rew` head in the checkpoint — added to the save format here, so it lands on the next run
([[0054]]: working baseline re-run with heads saved → reward-gap probe).

## Lesson / next

The diagnostic localized the wall: **dynamics imagine fine → the failure is downstream** (policy or
reward readout), not the model's predictive accuracy. This kills "improve the world model" and "push
grounding harder" as the next move, and points at the reward-readout (is the task reward head
over-optimistic on OOD imagined states, [[0045-rtfm-oracle-probe]]/[[0047-rtfm-conservative-reward]]
redux?) vs pure credit-assignment ([[0043-rtfm-execution-wall]], needing planning/horizon). exp0054
measures the reward gap to choose between them. Tooling
(`imagine_report.py`, `visualize_rtfm.py`) is reusable and its GIFs are embeddable in report pages —
the first bricks of the imagination-viewer for the [[mentored-learning-loop]].

## Links

[[0051-rtfm-flat-vr-optimization]] · [[0052-rtfm-decoupled-vr-head]] · [[0045-rtfm-oracle-probe]] · [[0047-rtfm-conservative-reward]] · [[0043-rtfm-execution-wall]] · [[validated-reading-reward]] · [[mentored-learning-loop]]
