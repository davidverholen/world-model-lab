---
status: planned
owner: world-model
scope: local
verified: false
last_reviewed: 2026-06-15
---

# 0048: validated-reading intrinsic reward — pay the agent for testing what it read

**Status: PRE-REGISTERED (not yet implemented).** The redirect after the conservative-reward thread
([[0047-rtfm-conservative-reward]]) confirmed the wall is not reward-head calibration. Design:
[[validated-reading-reward]]. Fresh-session pick-up: read that design note + [[0040-rtfm-actor-conditioning]]
+ this page, then implement §Setup.

**Supersedes the earlier extrinsic exp0048 plan** (keep env reading-shaping on + swapped-mode
collection). Dropped because: (a) the env shaping was already on (annealed from 1.0) during correct-mode
collection and `swap_follow` stayed 0 — "provide an obedience reward in correct-mode" is already shown
insufficient; (b) rewarding obedience to a swapped manual that earns no true achievement flirts with
*teaching the test*. The intrinsic version below is anti-baking-clean and attacks the gap directly.

## Hypothesis

The durable rung-4 wall is the [[0040-rtfm-actor-conditioning]] identifiability gap: in correct-mode,
"obey the displayed manual" ≡ "earn the reward", so nothing trains obedience per se → `swap_follow ≈ 0`
across every experiment. Add an **intrinsic reward for confirming a manual-derived prediction against
the real environment** (the manual's *marginal* predictive value, validated by the real transition —
see [[validated-reading-reward]]):

```
r_intrinsic = clip_≥0( pred_error(real s' | null/shuffled manual) − pred_error(real s' | correct manual) )
```

added to the training reward at collection time. **Predict:** the agent learns "manuals reliably
predict reality → follow them", so on held-out length-2 **`swap_follow` lifts clearly off ~0** with
`correct ≫ none` and `swapped ≪ correct` preserved (genuine reading, not baking) — the first lever to
move the metric every prior approach left pinned.

## Counter-outcomes (named)

- **Null** (`swap_follow` stays ~0, intrinsic reward fires but obedience does not transfer): the gap is
  not reward-*provision* but representational/architectural — the actor cannot route reading into
  action even when obedience is rewarded → relocate to the binding architecture / hierarchy.
- **Dark-room exploit** (intrinsic reward concentrates on trivial/stationary states; agent farms it
  without reading): the marginal framing failed to suppress it → audit where reward lands; add the
  learning-progress / aleatoric guard.
- **Bakes** (`swapped` tracks `correct`): not genuine grounding — but the mechanism rewards real
  prediction-confirmation, not a label, so this would indicate a vision leak, not the reward.

## Setup (implementation plan — not yet built)

- **Reward computation (collection time, `collect_rtfm`):** per real transition, encode the real next
  obs; run the WM one step from the current belief under (i) the correct manual tokens and (ii) a
  null/shuffled manual; intrinsic reward = clip≥0 of (null pred-error − correct pred-error) against the
  **real** next embedding. Reuse the `manual_aux` invariance machinery for the two predictions.
  **Detach** to prevent the actor from hacking what it predicts. Store as an extra reward channel (or
  add to `r_read`) with a coefficient flag `--validated-reading-coef` (0 = off / unchanged).
- **Ablations:** coef sweep (small, like 0047) once the mechanism is validated; correct-vs-null vs
  correct-vs-shuffled baseline; with/without clip.
- **Dispatch order:** validate on **length-1 first** (cheaper; grounding partially worked there) — does
  `swap_follow` ignite at all? — then length-2 (the real bar). Else identical to 0045–0047 (4 seeds,
  curriculum+aux WM, `--mpc-eval --oracle-probe`, K=1).
- **Smoke + CPU test** the reward term (sign: manual that predicts the real transition better → positive
  reward; detached; null-manual baseline) before dispatch.

## Standing bar (success) — tag: `LADDER-EXIT`

`swap_follow` clearly off ~0 on held-out length-2 (after igniting at length-1), with `correct ≫ none`
and `swapped ≪ correct`. That is the first direct crack in the exp-0040 identifiability wall and the
read→act grounding the whole rung needs → unblocks the next rung. Stall note: this is a genuinely new
mechanism (intrinsic validated-reading) with literature anchors ([[vime-2016]], [[marino-hypothesis-2020]]),
not a known-class repeat — the calibration thread (0044–0047) is a *different* axis now closed.

## Links

[[validated-reading-reward]] · [[0040-rtfm-actor-conditioning]] · [[0047-rtfm-conservative-reward]] · [[vime-2016]] · [[marino-hypothesis-2020]] · [[icm-2017]] · [[rnd-2018]] · [[language-grounding]] · [[hierarchical-imagination-agent]]
