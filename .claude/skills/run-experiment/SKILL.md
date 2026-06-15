---
name: run-experiment
description: >
  Run a world-model experiment end-to-end: hypothesis → setup → run → durable
  experiment page → route learnings. Use when asked to "run an experiment",
  "test whether X", or to execute items from ROADMAP.md.
---

# Run an experiment

Experiments are only real if they end as a page in `knowledge/experiments/`.

1. **Before running**: create `knowledge/experiments/NNNN-<slug>.md` from
   `knowledge/_schema/templates/experiment.md` with the hypothesis written down
   *first* (falsifiable, one sentence) and the planned setup. Check the
   [[environment-ladder]] — the experiment must live on the current rung (ADR 0003:
   no rung-skipping).
2. **Run**: `uv run python -m <module>` with explicit seeds. Record command line,
   seeds, and `git rev-parse --short HEAD` in the page. Artifacts go to `runs/`
   (gitignored); the page links them. Monitor an in-flight run live with
   `uv run python scripts/plot_experiment.py runs/expNNNN --watch 30` — a responsive
   auto-refreshing dashboard window rendered into the gitignored `runs/<exp>/_preview/`.
3. **Record**: results into the page exactly as observed — numbers first, no
   interpretation in the Result section. Results are facts: never retro-edit a
   Result section; follow-up experiments supersede. Then plot the trajectory:
   `uv run python scripts/plot_experiment.py runs/expNNNN` renders the all-metrics overview
   and lists the panel slugs; re-run with `--panels <slug,slug>` to render full-width ONLY
   the panels you'll feature (so committed assets = overview + featured panels, nothing else).
   In the **Trajectory** section embed each featured panel above its own one-paragraph reading;
   put the `overview.png` in the **All-metrics overview** section at the bottom. Read the
   multi-seed aggregate, not one seed. Figure = fact; paragraph = interpretation.
4. **Route learnings** (PROCESS.md table): update affected concept pages; propose an
   ADR if an architectural choice fell out; new recurring concept → page candidate.
5. **Bookkeep**: INDEX.md line, LOG.md entry, tick ROADMAP.md if it closes an item.

GPU note: local laptop GPU (8 GB) — see CLAUDE.md hardware section for limits and
the remote 16 GB option before sizing up models.
