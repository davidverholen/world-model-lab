# world-model

Experiments with **world models**: agents that learn to predict their environment in
latent space — not tokens, not pixels — and use that prediction to act. The working
goal is autonomous game-playing agents that learn from their own play, starting with
tiny gridworlds and climbing toward real-world transfer ([ROADMAP.md](ROADMAP.md)).

## Quick start

```bash
uv sync                                   # install (Python 3.12, PyTorch, MiniGrid)
uv run pytest -q                          # smoke tests
uv run python -m world_model.collect      # collect random MiniGrid trajectories,
                                          # train a first latent predictor, watch it collapse
```

## What's here

| Path | What |
|---|---|
| `src/world_model/` | envs (Gymnasium/MiniGrid), models (encoder, latent dynamics), agents, replay |
| `tests/` | fast CPU smoke tests |
| `knowledge/` | curated, agent-maintained research wiki — papers, concepts, ADRs, experiment records |
| `ROADMAP.md` | the environment ladder |
| `CLAUDE.md` / `AGENTS.md` | rules for AI agents working in this repo |

## The knowledge base

`knowledge/` is a three-layer wiki (immutable sources → maintained pages → schema)
adapted from [Karpathy's LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
and operated under the curation model of the
[Context Architecture whitepaper](https://www.linkedin.com/in/david-verholen-14aa23aa/):
every page carries `status`/`owner`/`verified` metadata, unread material stays in an
intake queue, and learnings from experiments are routed into durable pages instead of
dying in chat history. Start at [knowledge/INDEX.md](knowledge/INDEX.md).

The whole setup is packaged as a reusable skill in
`.claude/skills/context-architecture/` — drop it into any project to get the same
structure.
