---
status: current
owner: human
scope: local
sources: [arxiv:2509.24527]
verified: true
last_reviewed: 2026-06-12
---

# 0005: Minecraft as the pre-real-world milestone — three learning signals

**Status:** accepted (2026-06-12, proposed by the maintainer)

## Context

The ladder jumped from games to real-world transfer with nothing in between that
forces the *integration* questions (multi-modal learning, language grounding) the
real world demands. Minecraft fills that slot without robotics hardware, and is
uniquely resource-rich: VPT's action-labeled video corpus (IDM pseudo-labeling of
~70k h YouTube), MineRL/MineDojo datasets + text corpus, and published reference
agents (DreamerV3 online diamonds; Dreamer 4 offline diamonds; VPT; Voyager).

## Decision

Insert **"play Minecraft successfully"** as the final game rung (5b), before
real-world transfer. Training combines three signals, in deliberate order of
research risk:

1. **Self-play** — the flywheel (rungs 1–5 machinery, scaled);
2. **Action-labeled video** — VPT-style: learn/borrow an inverse dynamics model,
   pseudo-label public gameplay video, pretrain the world model offline
   (Dreamer-4-style) before any interaction;
3. **Text** (tutorials/wiki) — the open-research component: embed language into
   the latent world model (Dynalang direction; VL-JEPA/LLM-JEPA line in QUEUE),
   not merely LLM-as-planner (Voyager direction) — though that is the pragmatic
   fallback.

**Text-signal staircase (maintainer, 2026-06-12):** de-risk language grounding on cheap
envs first — (i) Messenger/RTFM, where reading the per-episode manual is
*necessary* to win (clean grounding signal, MiniGrid-scale compute);
(ii) text-augmented Crafter, where tutorials are helpful-but-not-necessary
(metric: sample-efficiency delta with vs without text); (iii) Minecraft with real
tutorials. Steps (i)–(ii) need no new hardware.

"Successfully" (provisional, refine at rung entry): obtain a diamond at a published
sample-efficiency tier, plus a breadth metric (MineDojo-style task suite subset).

## Consequences

- Rungs 3–5 gain a unifying target; Crafter is explicitly Minecraft-in-miniature.
- Compute reality: this rung needs rented bursts or upgraded hardware
  ([[compute-strategy]] triggers apply); video pretraining is the heaviest item.
- New research thread opens (language-in-world-model) with its own intake needs:
  VPT, MineDojo, Voyager, Dynalang, STEVE-1 queued (ids to verify at ingest).
- Real-world transfer (old rung 6) remains the horizon after this.

## Links

[[environment-ladder]] · [[dreamer4-2025]] · [[compute-strategy]] · [[retention]]
(continual-learning fixes must hold at this scale) · ROADMAP.md
