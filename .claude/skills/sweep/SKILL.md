---
name: sweep
description: >
  Run a multi-seed / multi-config parameter sweep and get an aggregated markdown
  results table. Use when an experiment needs several runs ("3 seeds", "with and
  without X", "grid over lambda") instead of hand-writing shell loops.
---

# Parameter sweep

Tool: `scripts/sweep.py` (authoritative usage in its docstring).

```bash
uv run python scripts/sweep.py \
    --grid seed=0,1,2 --grid sigreg-weight=0,0.05 \
    --name exp00NN \
    -- python -m world_model.collect --env-id MiniGrid-Empty-8x8-v0
```

- `--remote` dispatches each combo through `scripts/remote.sh run` (clean tree
  required; use for runs >~1h total — see knowledge/design/compute-strategy.md).
- Name the sweep after the experiment page (`exp0007`), so `runs/sweeps/exp0007/`
  links trivially from `knowledge/experiments/0007-*.md`.
- Paste `RESULTS.md` into the experiment page's Result section and prune columns
  to what the hypothesis needs; keep the raw table in runs/ (gitignored).
- Metrics are scraped from each run's final output lines (`name=value` /
  `name: value`) — print what you want captured.
