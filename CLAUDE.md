# world-model

Experiments with world models — agents that learn to predict their environment
(latents, not tokens) and use that prediction to act. Long-term goal: autonomous
game-playing agents climbing an environment ladder (MiniGrid → Crafter → Atari →
continuous/3D → real-world transfer). See `ROADMAP.md`.

## Commands

```bash
uv run pytest -q                          # tests
uv run ruff check . && uv run ruff format .   # lint/format
uv run python -m world_model.collect      # smoke pipeline: collect + train latent predictor
uv run python -m world_model.play         # watch an agent play (window); --record out.gif
```

uv-managed (Python 3.12, PyTorch).

## Hardware

- **Local (this machine, Linux):** RTX 4070 Laptop, 8 GB — default for development
  and short (<~1 h) experiments. Keep models small; prefer batch-size/precision
  tweaks over architecture growth when memory binds. Note: power-capped at ~45 W and
  thermally throttles within minutes of sustained load (measured — see
  knowledge/design/compute-strategy.md); don't schedule multi-hour training here.
- **Remote (Windows desktop):** RTX 5070 Ti, 16 GB — available for bigger runs
  (rung 3+, longer training). Not yet wired up: needs uv + CUDA PyTorch there and a
  way to dispatch runs (simplest: git pull + `uv run` over SSH; decide when first
  needed and record as an ADR).

## Layout

- `src/world_model/` — package: `envs/` (gymnasium/minigrid wrappers), `models/`
  (encoders, latent predictors), `agents/`, `training/` (replay, loops)
- `tests/` — pytest; keep the smoke suite fast (<30 s, CPU-only)
- `knowledge/` — the curated knowledge base (see below)
- `experiments/` (runs/ dirs, gitignored) vs `knowledge/experiments/` (the durable
  experiment record — hypothesis/setup/result/lesson pages)

## Knowledge base — binding rules

`knowledge/` is an agent-maintained wiki run under a context-architecture operating
model. Full conventions: `knowledge/_schema/SCHEMA.md`; process:
`knowledge/_schema/PROCESS.md`. The short version every session must follow:

1. **Start from `knowledge/INDEX.md`**; load only pages relevant to the task
   (minimum needed context — never bulk-load the wiki).
2. **Respect frontmatter**: `status: stub|stale` or `verified: false` pages must not
   be cited as authority. `owner: human` pages (all of `decisions/`, schema, results
   sections) — propose changes, don't silently rewrite meaning.
3. **Unread material goes to `knowledge/sources/QUEUE.md`**, never directly into wiki
   pages. Ingestion = `/ingest-source`.
4. **Route learnings before ending a session** that produced any: experiment outcomes →
   `knowledge/experiments/`, architecture choices → propose ADR, corrections → fix page
   + LOG entry. Routing table in PROCESS.md.
5. **Append to `knowledge/LOG.md`** whenever the wiki changed.
6. **Milestone = curate + commit** (`/milestone`): after every meaningful unit of work,
   run the checkpoint — verify (tests/lint) → knowledge curation → commit everything.
   No milestone leaves uncommitted changes behind; no commit carries an uncurated KB.

## Skills

- `/context-architecture` — the full, reusable knowledge-base system (setup + operations);
  use it to re-instantiate this structure in other projects
- `/ingest-source` — properly ingest a paper/post/repo into the KB
- `/kb-lint` — wiki health check (contradictions, stale pages, orphans)
- `/run-experiment` — hypothesis → run → durable experiment page
- `/scout-sources` — search for new world-model work, verify, queue
- `/milestone` — checkpoint: verify (tests/lint/publish-check + reviewer agent on
  non-trivial diffs) → curate knowledge → commit everything
- `/sweep` — multi-seed/multi-config runs with aggregated results table (scripts/sweep.py)

## Model delegation

When spawning subagents, route by complexity (standing authorization from Dave):
research design, training-failure debugging, and `owner: human` meaning stay in the
main loop (Fable); substantial implementation subagents and the **reviewer** agent
(research-code diff review at milestones) → **opus**; structured
KB work (ingest, lint, scout, experiment write-ups) → **sonnet** (curator and scout
agents default to this); mechanical sweeps (link checks, INDEX consistency, code
search) → **haiku**. When unsure, inherit the main-loop model.

## Code conventions

- Python 3.12, ruff (line length 100), type hints on public APIs.
- Modules carry docstrings that link to the relevant `knowledge/` page instead of
  re-explaining theory.
- Every experiment must be reproducible: seed, commit hash, and command line recorded
  in its `knowledge/experiments/` page.
