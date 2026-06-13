# world-model

Experiments with **world models**: agents that learn to predict their environment in
latent space — not tokens, not pixels — and use that prediction to act. The working
goal is autonomous game-playing agents that learn from their own play, starting with
tiny gridworlds and climbing toward real-world transfer ([ROADMAP.md](ROADMAP.md)).

## Quick start

```bash
uv sync            # install (Python 3.12, PyTorch, MiniGrid; CUDA used if present, CPU works)
uv run pytest -q   # fast CPU smoke tests
```

**1. See the core problem — and its fix.** Naive latent prediction collapses (the
encoder cheats by mapping everything to one point); LeJEPA's SIGReg term stops it:

```bash
uv run python -m world_model.collect                       # watch the latents collapse
uv run python -m world_model.collect --sigreg-weight 0.05  # SIGReg on — latents survive
```

**2. Train an agent, then watch it play.** Train a recurrent world model on DoorKey
(partial observability + sparse reward), then let it plan inside the learned latent
model — solving the door-then-key task it never saw a reward shaped for:

```bash
# train (uses a GPU if available; ~minutes). Produces a checkpoint.
uv run python -m world_model.train_recurrent \
    --env-id MiniGrid-DoorKey-5x5-v0 --rounds 2 --save runs/doorkey.pt

# watch it play in a live window (--record out.gif to save a GIF instead)
uv run python -m world_model.play \
    --checkpoint runs/doorkey.pt --env-id MiniGrid-DoorKey-5x5-v0
```

Run `python -m world_model.play` with no `--checkpoint` for a quick random-agent
render on Empty-8x8 (handy to confirm the display works). The experiment behind this
path — and how it beats a model-free PPO baseline — is written up in
[knowledge/experiments/](knowledge/experiments/).

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
[Context Architecture whitepaper](https://github.com/davidverholen/context-architecture)
(rendered PDF under the repo's Releases):
every page carries `status`/`owner`/`verified` metadata, unread material stays in an
intake queue, and learnings from experiments are routed into durable pages instead of
dying in chat history. Start at [knowledge/INDEX.md](knowledge/INDEX.md).

The whole setup is packaged as a reusable skill in
`.claude/skills/context-architecture/` — drop it into any project to get the same
structure.

## License

MIT — see [LICENSE](LICENSE).
