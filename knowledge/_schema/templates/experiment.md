---
status: draft
owner: world-model       # results are facts; recorded, not retro-edited
scope: local
sources: []
verified: true
last_reviewed: YYYY-MM-DD
---

# NNNN: <Experiment name>

## Hypothesis

Falsifiable claim + named counter-outcome(s).
**Bar:** `LADDER-EXIT` | `EXTRA-RIGOR` — the success criterion and which kind it is
(PROCESS.md §Efficiency guardrails). Note the thread's stall count if this is a redesign.

## Setup

env, model config, seeds, commit hash, command line.

## Result

Numbers, curves (link tensorboard/run dir). No interpretation here.

## Trajectory

Survey the panels with `uv run python scripts/plot_experiment.py runs/expNNNN` (renders the
all-metrics overview + lists the panel slugs), then render full-width ONLY the panels you'll feature:
`uv run python scripts/plot_experiment.py runs/expNNNN --panels grounding-headline,eval-scores`.
Multi-seed mean ± min–max band on a transparent background (reads on light or dark). Embed each
featured panel above the paragraph that reads it — a figure is a fact (the curve); the prose is the
interpretation. Read the multi-seed aggregate, not one seed (the band is the point — a single seed
can mislead).

### <panel that carries the headline, e.g. grounding headline>

![expNNNN grounding-headline](../../assets/exp-NNNN/grounding-headline-0-1.png)

_What this curve does and why it matters: where the regime changed, what the band says, any
divergence between imagined and real performance._

<!-- add more `### panel` + figure + reading blocks only as the interpretation needs them -->

## Lesson

What we now believe, and which concept pages were updated because of it.

## Links

[[...]]

## All-metrics overview

<!-- the full 3-column contact sheet of every metric, for at-a-glance reference; always rendered -->

![expNNNN overview](../../assets/exp-NNNN/overview.png)
