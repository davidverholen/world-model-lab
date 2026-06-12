---
status: stub
owner: agent
scope: local
sources: [arxiv:2509.24527]
verified: true
last_reviewed: 2026-06-12
---

# Dreamer 4: Training Agents Inside of Scalable World Models (Hafner, Yan, Lillicrap, 2025)

**Lab:** Google DeepMind · **Project:** https://danijar.com/project/dreamer4/ · **Read state:** abstract-only

## One-paragraph summary

2B-parameter agent; scales the world model to a fast transformer video model trained
with a novel "shortcut forcing" objective for real-time interactive inference. Policy
is trained by RL entirely *inside* the world model from a **fixed offline dataset** —
first agent to obtain Minecraft diamonds (20,000+ low-level actions from raw pixels)
without any environment interaction, beating OpenAI's VPT with 100× less data.

## Relevance to our experiments

End-state evidence for [[imagination-training]] at scale, and the strongest argument
that world models are the path to learning when interaction is expensive — exactly the
property needed for eventual real-world transfer (ladder rung 6).

## Links

[[imagination-training]] · [[dreamerv3-2023]] · [[environment-ladder]]
