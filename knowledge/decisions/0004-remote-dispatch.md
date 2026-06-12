---
status: current
owner: human
scope: local
sources: []
verified: true
last_reviewed: 2026-06-12
---

# 0004: Remote GPU dispatch — SSH + git push over Tailscale, nothing fancier

**Status:** accepted (2026-06-12)

## Context

The dev laptop's GPU is power-capped (~45 W) and thermally throttles within minutes
([[compute-strategy]]); an otherwise idle Windows desktop (RTX 5070 Ti, 16 GB) is
~4–6× faster for GPU-bound training and SSH-reachable (e.g. via Tailscale). Options
considered: (a) SSH + git push + `uv run`; (b) a job queue / orchestrator (Ray,
slurm-lite, custom daemon); (c) cloud-style CI runners; (d) file sync (Syncthing)
+ manual runs.

## Decision

Option (a), implemented as `scripts/remote.sh` (setup/gpu/run/pull) + `docs/REMOTE.md`:
bare git repo on the remote as transport, working clone reset to the pushed commit,
`uv sync` for env parity (win32 gets torch+cu130 via marker-gated index in
pyproject.toml), results copied back with scp. Windows OpenSSH Server (Tailscale's
SSH server doesn't exist on Windows). All machine identity lives outside the repo:
SSH alias in the user's `~/.ssh/config`, knobs via env vars or gitignored
`.env.remote` (committed example: `.env.remote.example`) — the repo stays
publishable with no references to personal machines.

Dispatch requires a clean tree and pushes HEAD: every remote run is pinned to a
commit hash, which the experiment-reproducibility rule requires anyway.

## Consequences

- Zero standing infrastructure; debuggable with plain ssh/git; one new script.
- No queueing — one run at a time, manually started. Acceptable for a solo project;
  revisit (b) only when we want unattended sweeps (that's also where vast.ai
  bursts enter, see [[compute-strategy]]).
- Windows-side quirks (cmd.exe default shell, administrators_authorized_keys)
  are documented in docs/REMOTE.md rather than abstracted away.

## Links

[[compute-strategy]] · [[0003-environment-ladder]] · docs/REMOTE.md
