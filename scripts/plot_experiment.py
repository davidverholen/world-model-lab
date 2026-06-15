"""Parse experiment run logs into a multi-panel trajectory plot for the experiment report.

The rung-3/rung-4 training loops all emit per-round metrics as ``key=value`` floats under
``round N:`` headers (see train_rtfm.py / train_crafter.py / train_dreamer.py), and multi-seed
runs land as ``s0.log .. sN.log`` in one ``runs/<exp>/`` dir. That regularity is what makes this
GENERIC: we scan every ``key=value`` token, group by round, average across seeds (min-max band),
and lay the metrics out in panels. Metric -> panel grouping uses the known taxonomy below; any
unrecognised metric still gets plotted in its own panel, so non-RTFM logs work too.

    # 1. survey: render the 3-col overview + list the available panel slugs
    uv run python scripts/plot_experiment.py runs/exp0046
    # 2. feature: also render full-width PNGs for the panels you'll embed
    uv run python scripts/plot_experiment.py runs/exp0046 --panels grounding-headline,eval-scores

Always writes assets/exp-<NNNN>/overview.png (the 3-column contact sheet of ALL metrics — goes at
the bottom of the report as an at-a-glance overview). With --panels, ALSO writes one full-width
assets/exp-<NNNN>/<slug>.png per requested panel — so the committed assets are exactly the figures
the report uses (overview + featured panels), nothing else. Prints markdown embed snippets. The
author picks which panels carry the story; interpretation text stays hand-written.
"""

from __future__ import annotations

import argparse
import re
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Theme-agnostic styling: transparent background + neutral-gray chrome (#888 reads on both light
# and dark) + saturated line colors that stay legible against either. The figure is saved with a
# transparent background so the host page's theme shows through. See plot() savefig(transparent=).
_GRAY = "#888888"
plt.rcParams.update(
    {
        "text.color": _GRAY,
        "axes.labelcolor": _GRAY,
        "axes.titlecolor": _GRAY,
        "axes.edgecolor": _GRAY,
        "xtick.color": _GRAY,
        "ytick.color": _GRAY,
        "grid.color": _GRAY,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        # Saturated, mid-luminance palette — each color holds up on both white and dark backgrounds.
        "axes.prop_cycle": plt.cycler(
            color=["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#a855f7", "#06b6d4"]
        ),
    }
)

# Metric -> panel grouping. Order = panel order. Metrics share a panel only when they share a
# scale/meaning (so the y-axis stays readable). Anything not listed is auto-bucketed (one panel
# per unknown metric) so the script never silently drops a series.
PANELS: list[tuple[str, list[str]]] = [
    ("schedule (curriculum + anneal)", ["len", "shaping_coef"]),
    ("collection signal", ["events"]),
    ("world-model losses", ["recon", "kl", "manual_aux"]),
    ("optimization losses", ["actor_loss", "critic_loss"]),
    ("imagined_return (optimism)", ["imagined_return"]),
    ("eval scores [0-1]", ["correct", "none", "swapped"]),
    ("grounding headline [0-1]", ["grounding", "swap_follow"]),
    ("reading diagnostic", ["inv_ratio", "inv_correct", "inv_wrong"]),
    (
        "MPC eval [0-1]",
        ["mpc_correct", "mpc_none", "mpc_swapped", "mpc_grounding", "mpc_swap_follow"],
    ),
    ("oracle probe", ["oracle_pct", "oracle_ret", "oracle_rand_ret"]),
]
# Metrics where a horizontal reference line aids reading.
REF_LINES = {"inv_ratio": 1.0, "grounding": 0.0}
# Bookkeeping keys that are monotonic or pure counts — not worth a panel (kept out of the
# auto-bucket so they don't clutter). buffer = replay size (monotonic); *_n = sample counts.
IGNORE = {"buffer", "n", "oracle_n"}

_KV = re.compile(r"([A-Za-z_]\w*)=([-+]?\d*\.?\d+)")
_ROUND = re.compile(r"^round\s+(\d+):")


def parse_log(path: Path) -> dict[str, list[float]]:
    """Return {metric: [value per round]} for one seed log. Rounds with a missing metric get NaN."""
    rounds: dict[int, dict[str, float]] = defaultdict(dict)
    cur = -1
    for raw in path.read_text().splitlines():
        line = raw.strip()
        m = _ROUND.match(line)
        if m:
            cur = int(m.group(1))
        if cur < 0:
            continue
        prefix = ""
        if line.startswith("MPC "):
            prefix = "mpc_"
        elif line.startswith("ORACLE "):
            prefix = "oracle_"  # pct -> oracle_pct; oracle_ret/rand_ret already explicit
        for key, val in _KV.findall(line):
            name = prefix + key if prefix and not key.startswith(prefix.rstrip("_")) else key
            rounds[cur][name] = float(val)
        # The "-> 18 tutorial events; 60 manuals; buffer 17199" line has no key=value events count.
        ev = re.search(r"(\d+)\s+\w+\s+events", line)
        if ev:
            rounds[cur]["events"] = float(ev.group(1))
    if not rounds:
        return {}
    n = max(rounds) + 1
    keys = {k for r in rounds.values() for k in r}
    return {k: [rounds.get(i, {}).get(k, np.nan) for i in range(n)] for k in keys}


def aggregate(seeds: list[dict[str, list[float]]]) -> dict[str, np.ndarray]:
    """Stack per-seed series into {metric: (n_seeds, n_rounds)} padded with NaN."""
    keys = {k for s in seeds for k in s}
    n = max((len(v) for s in seeds for v in s.values()), default=0)
    out = {}
    for k in keys:
        rows = []
        for s in seeds:
            row = s.get(k, [])
            rows.append(row + [np.nan] * (n - len(row)))
        out[k] = np.array(rows, dtype=float)
    return out


def trim_incomplete(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Drop trailing rounds that are under-filled (e.g. a cut-off final ``round N: collecting...``
    header that logged only the schedule). Keeps rounds with >=50% of the peak metric count."""
    if not data:
        return data
    filled = sum(np.any(~np.isnan(v), axis=0).astype(int) for v in data.values())
    if not len(filled):
        return data
    keep = filled >= 0.5 * filled.max()
    last = int(np.max(np.where(keep))) + 1 if keep.any() else len(filled)
    return {k: v[:, :last] for k, v in data.items()}


def _slug(title: str) -> str:
    """'grounding headline [0-1]' -> 'grounding-headline'."""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", title.lower())).strip("-")


def select_panels(data: dict[str, np.ndarray]) -> list[tuple[str, list[str]]]:
    present = {k for k, v in data.items() if not np.all(np.isnan(v))} - IGNORE
    panels = [(t, [m for m in ms if m in present]) for t, ms in PANELS]
    panels = [(t, ms) for t, ms in panels if ms]
    known = {m for _, ms in panels for m in ms}
    for k in sorted(present - known):
        panels.append((k, [k]))  # auto-bucket: one panel per unknown metric
    return panels


def _draw(ax, data: dict[str, np.ndarray], ptitle: str, metrics: list[str], n_seeds: int) -> None:
    """Draw one panel's curves (mean line + min–max band + ref line) onto a given axis."""
    for m in metrics:
        v = data[m]
        x = np.arange(v.shape[1])
        # All-NaN columns (a metric logged only in some rounds, e.g. MPC at len-2) are
        # intentional gaps; the nan-reductions warn on them harmlessly.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(v, axis=0)
            lo, hi = np.nanmin(v, axis=0), np.nanmax(v, axis=0)
        ax.plot(x, mean, "o-", ms=4, label=m)
        if n_seeds > 1:
            ax.fill_between(x, lo, hi, alpha=0.15)
        if m in REF_LINES:
            ax.axhline(REF_LINES[m], color=_GRAY, ls="--", lw=0.7)
    ax.set_title(ptitle, fontsize=10)
    ax.set_xlabel("round")
    ax.grid(alpha=0.3)
    if len(metrics) > 1:
        ax.legend(fontsize=8, framealpha=0.0)


def render_overview(data: dict[str, np.ndarray], out_dir: Path, subtitle: str) -> str:
    """Render the 3-column contact sheet of ALL panels to out_dir/overview.png (the bottom-of-page
    'all metrics' figure). Returns the slug ('overview')."""
    data = trim_incomplete(data)
    n_seeds = max((v.shape[0] for v in data.values()), default=1)
    panels = select_panels(data)
    ncols = 3
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.4 * nrows), squeeze=False)
    band = f"mean of {n_seeds} seeds, band = min–max" if n_seeds > 1 else "single seed"
    fig.suptitle(f"{subtitle} — all metrics ({band})", fontsize=14, fontweight="bold")
    for ax, (ptitle, metrics) in zip(axes.flat, panels, strict=False):
        _draw(ax, data, ptitle, metrics, n_seeds)
    for ax in axes.flat[len(panels) :]:
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "overview.png", dpi=110, transparent=True)
    plt.close(fig)
    return "overview"


def render_full(
    data: dict[str, np.ndarray], out_dir: Path, subtitle: str, want: set[str] | None
) -> list[tuple[str, str]]:
    """Render full-width figures ONLY for the requested panel slugs (``want``). With want=None,
    render none (the author hasn't chosen yet). Returns [(slug, title)] for what was written."""
    data = trim_incomplete(data)
    n_seeds = max((v.shape[0] for v in data.values()), default=1)
    band = f"mean of {n_seeds} seeds, band = min–max" if n_seeds > 1 else "single seed"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, str]] = []
    for ptitle, metrics in select_panels(data):
        slug = _slug(ptitle)
        if want is None or slug not in want:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        _draw(ax, data, ptitle, metrics, n_seeds)
        ax.set_title(f"{subtitle} — {ptitle}", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"round   ({band})")
        fig.tight_layout()
        fig.savefig(out_dir / f"{slug}.png", dpi=120, transparent=True)
        plt.close(fig)
        written.append((slug, ptitle))
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="runs/<exp> dir or a single .log file")
    ap.add_argument("--exp", help="experiment slug, e.g. 0046 (default: inferred from path)")
    ap.add_argument(
        "--panels",
        help="comma/space-separated panel slugs to render FULL-WIDTH (the ones embedded in the "
        "report). Omit to render only the overview + list available slugs.",
    )
    args = ap.parse_args()

    p = Path(args.path)
    logs = sorted(p.glob("s*.log")) if p.is_dir() else [p]
    if not logs:
        raise SystemExit(f"no s*.log under {p}")
    seeds = []
    for f in logs:
        parsed = parse_log(f)
        if parsed:
            seeds.append(parsed)
        else:
            print(f"skipped {f}: no round metrics")  # don't silently shrink the seed count
    if not seeds:
        raise SystemExit(f"no parseable round metrics in {logs}")

    # Slug from the run DIR name (exp0045), not a single log's stem (s0 -> "0"). --exp overrides.
    exp = args.exp or re.sub(r"\D", "", p.name if p.is_dir() else p.parent.name)
    n_rounds = max(len(v) for s in seeds for v in s.values())
    subtitle = f"exp{exp} ({len(seeds)} seeds, {n_rounds} rounds)"
    out_dir = Path("assets") / f"exp-{exp}"
    data = aggregate(seeds)

    # The 3-column overview is ALWAYS rendered (the bottom-of-page "all metrics" figure). Full-width
    # panels render ONLY for slugs the author selects via --panels, so committed assets are exactly
    # the figures the report uses (overview + featured panels) and nothing else.
    render_overview(data, out_dir, subtitle)
    want = set(re.split(r"[,\s]+", args.panels.strip())) if args.panels else None
    written = render_full(data, out_dir, subtitle, want)
    rel = lambda name: (Path("..") / ".." / out_dir / name).as_posix()  # noqa: E731

    all_panels = select_panels(trim_incomplete(data))
    print(f"wrote overview.png + {len(written)} full-width panel(s) to {out_dir}/")
    print("\navailable panel slugs:")
    for ptitle, _ in all_panels:
        print(f"  {_slug(ptitle):<28} {ptitle}")

    if want is None:
        print("\nRe-run with --panels <slug,slug> to render the featured panels full-width.")
    else:
        print("\n--- Trajectory section: embed each featured panel above its reading ---\n")
        for slug, ptitle in written:
            print(f"### {ptitle}\n![exp{exp} {slug}]({rel(slug + '.png')})\n\n_<reading>_\n")

    print("--- bottom of page: all-metrics overview ---\n")
    print(f"## All-metrics overview\n\n![exp{exp} overview]({rel('overview.png')})")


if __name__ == "__main__":
    main()
