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
    # 3. live: serve a responsive auto-refreshing dashboard (reach it from your phone via tailscale)
    uv run python scripts/plot_experiment.py runs/exp0046 --watch 30   # http://0.0.0.0:8000/

Always writes assets/exp-<NNNN>/overview.png (the 3-column contact sheet of ALL metrics — goes at
the bottom of the report as an at-a-glance overview). With --panels, ALSO writes one full-width
assets/exp-<NNNN>/<slug>.png per requested panel — so the committed assets are exactly the figures
the report uses (overview + featured panels), nothing else. Prints markdown embed snippets. The
author picks which panels carry the story; interpretation text stays hand-written.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import re
import subprocess
import threading
import time
import warnings
import webbrowser
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
# In a sweep comparison, also skip the schedule (len/shaping_coef are identical across arms).
COMPARE_SKIP = IGNORE | {"len", "shaping_coef"}

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


def load(path: Path, quiet: bool = False) -> tuple[dict[str, np.ndarray] | None, int, int]:
    """Parse + aggregate every s*.log under path (or a single file). Returns (data, n_seeds,
    n_rounds); data is None if nothing parseable (so watch can wait for the first round)."""
    logs = sorted(path.glob("s*.log")) if path.is_dir() else [path]
    if not logs:
        raise SystemExit(f"no s*.log under {path}")
    seeds = []
    for f in logs:
        parsed = parse_log(f)
        if parsed:
            seeds.append(parsed)
        elif not quiet:
            print(f"skipped {f}: no round metrics")  # don't silently shrink the seed count
    if not seeds:
        return None, 0, 0
    n_rounds = max(len(v) for s in seeds for v in s.values())
    return aggregate(seeds), len(seeds), n_rounds


def load_arms(base: Path) -> list[tuple[str, dict[str, np.ndarray]]]:
    """Discover the arms of a split experiment: sibling run dirs sharing the base id, e.g.
    base=runs/exp0047 → runs/exp0047c01, ...c30. Each arm is a normal run; its label is the dir
    name with the shared prefix stripped (exp0047c10 → 'c10'). Returns [(label, data)]."""
    arms = []
    for d in sorted(base.parent.glob(f"{base.name}*")):
        # Only real run dirs are arms — skips the sibling _preview/ output dir (no s*.log) and any
        # other non-run match, so the glob can't pull in its own scratch output.
        if d.is_dir() and any(d.glob("s*.log")):
            data, _, _ = load(d, quiet=True)
            if data is not None:
                arms.append((d.name[len(base.name) :] or d.name, data))
    return arms


def render_compare(
    arms: list[tuple[str, dict[str, np.ndarray]]], out_dir: Path, base: str, want: set[str] | None
) -> list[tuple[str, str]]:
    """One chart PER METRIC, overlaying every arm. Each seed is a thin line colored by arm (an arm's
    seeds share a color), legend = arm labels — no mean/band, since with many arms the thin lines
    ARE the comparison. Returns [(slug, metric)]."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = sorted(
        {m for _, d in arms for m, v in d.items() if not np.all(np.isnan(v))} - COMPARE_SKIP
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    written: list[tuple[str, str]] = []
    for m in metrics:
        slug = _slug(m)
        if want is not None and slug not in want:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for i, (label, d) in enumerate(arms):
            if m not in d:
                continue
            color = colors[i % len(colors)]
            for j, row in enumerate(d[m]):  # one thin line per seed; one legend entry per arm
                if not np.all(np.isnan(row)):
                    ax.plot(
                        np.arange(len(row)),
                        row,
                        "-",
                        lw=1.1,
                        alpha=0.85,
                        color=color,
                        label=label if j == 0 else "_nolegend_",
                    )
        if m in REF_LINES:
            ax.axhline(REF_LINES[m], color=_GRAY, ls="--", lw=0.7)
        ax.set_title(f"{base} — {m} (per arm)", fontsize=11, fontweight="bold")
        ax.set_xlabel("round")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, framealpha=0.0, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"{slug}.png", dpi=120, transparent=True)
        plt.close(fig)
        written.append((slug, m))
    return written


def write_preview_html(out_dir: Path, subtitle: str, slugs: list[str], interval: int, tick: int):
    """Write a self-refreshing, responsive dashboard. A CSS grid (auto-fit minmax) reflows the
    panels as the window resizes — wide → several columns, narrow → one big readable column. The
    page meta-refreshes every ``interval``s; the ``?t=tick`` cache-buster pulls the fresh PNGs."""
    cells = "\n".join(f'    <figure><img src="{s}.png?t={tick}" alt="{s}"></figure>' for s in slugs)
    (out_dir / "index.html").write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{interval}">
<title>{subtitle} — live</title>
<style>
  :root {{ color-scheme: dark light; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:1rem; background:#1e1e1e; color:#cfcfcf;
          font-family: system-ui, sans-serif; overflow-x: hidden; }}
  header {{ display:flex; gap:1rem; align-items:baseline; flex-wrap:wrap; margin:0 0 1rem; }}
  h1 {{ font-size:1.05rem; margin:0; }}
  .meta {{ color:#888; font-size:.85rem; }}
  /* min(520px, 100%) so a narrow phone gets one full-width column instead of overflowing */
  .grid {{ display:grid; gap:1rem;
           grid-template-columns: repeat(auto-fit, minmax(min(520px, 100%), 1fr)); }}
  figure {{ margin:0; padding:.4rem; background:#111; border:1px solid #333; border-radius:8px;
            min-width:0; }}
  img {{ width:100%; max-width:100%; height:auto; display:block; }}
</style></head>
<body>
  <header><h1>{subtitle}</h1>
  <span class="meta">live · auto-refresh {interval}s · resize to reflow</span></header>
  <div class="grid">
{cells}
  </div>
</body></html>"""
    )


def _tailscale_ip() -> str | None:
    """Best-effort tailnet IPv4 (so the printed URL is phone-reachable). None if no tailscale."""
    with contextlib.suppress(Exception):
        out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=2)
        for line in out.stdout.splitlines():
            if line.strip():
                return line.strip()
    return None


def serve_dir(directory: Path, host: str, port: int) -> int:
    """Start a background HTTP server rooted at ``directory`` (binds ``host`` — use 0.0.0.0 so the
    dashboard is reachable from another device on the tailnet). Returns the bound port; tries a few
    if the first is taken. The thread is a daemon, so Ctrl-C in the render loop ends everything."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    for cand in range(port, port + 10):
        try:
            httpd = http.server.ThreadingHTTPServer((host, cand), handler)
        except OSError:
            continue
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return cand
    raise SystemExit(f"no free port in {port}..{port + 9}")


def watch(
    p: Path, exp: str, want: set[str] | None, interval: int, host: str, port: int, compare: bool
) -> None:
    """Serve a gitignored _preview/ over HTTP and re-render it on an interval. Single-run mode: all
    panels for one run (the responsive grid IS the overview). --compare mode: one chart per metric
    overlaying every sweep arm. --panels narrows either. Never touches committed assets/."""
    preview = (
        (p.parent / f"{p.name}_preview")
        if compare
        else (p if p.is_dir() else p.parent) / "_preview"
    )
    preview.mkdir(parents=True, exist_ok=True)
    index = preview / "index.html"
    if not index.exists():  # placeholder so the server has something before the first round lands
        index.write_text("<!doctype html><meta http-equiv=refresh content=2><body>waiting…</body>")

    bound = serve_dir(preview, host, port)
    print(f"serving {preview}/ on {host}:{bound}")
    print(f"  this machine:   http://localhost:{bound}/")
    ts = _tailscale_ip()
    if ts:
        print(f"  on your tailnet: http://{ts}:{bound}/   (open this on your phone)")
    print(f"  reflows on resize, refreshes every {interval}s. Ctrl-C to stop.")
    with contextlib.suppress(Exception):
        webbrowser.open(f"http://localhost:{bound}/")

    tick = 0
    try:
        while True:
            if compare:
                arms = load_arms(p)
                if arms:
                    subtitle = f"{p.name} sweep · {len(arms)} arms"
                    show = [s for s, _ in render_compare(arms, preview, p.name, want)]
                    write_preview_html(preview, subtitle, show, interval, tick)
                    tick += 1
            else:
                data, n_seeds, n_rounds = load(p, quiet=True)
                if data is not None:
                    subtitle = f"exp{exp} ({n_seeds} seeds, {n_rounds} rounds)"
                    slugs = [_slug(t) for t, _ in select_panels(trim_incomplete(data))]
                    show = [s for s in slugs if want is None or s in want]
                    render_full(data, preview, subtitle, set(show))
                    write_preview_html(preview, subtitle, show, interval, tick)
                    tick += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nstopped.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="runs/<exp> dir or a single .log file")
    ap.add_argument("--exp", help="experiment slug, e.g. 0046 (default: inferred from path)")
    ap.add_argument(
        "--panels",
        help="comma/space-separated panel slugs to render FULL-WIDTH (the ones embedded in the "
        "report). Omit to render only the overview + list available slugs.",
    )
    ap.add_argument(
        "--watch",
        type=int,
        metavar="SECONDS",
        help="live-preview mode: serve a responsive HTTP dashboard, re-rendering every SECONDS "
        "into gitignored runs/<exp>/_preview/ (in-flight run; never writes assets/).",
    )
    ap.add_argument(
        "--host",
        default="0.0.0.0",  # noqa: S104 — intentional: reachable from another device on the tailnet
        help="watch-server bind address (default 0.0.0.0 = all interfaces, incl. tailnet).",
    )
    ap.add_argument("--port", type=int, default=8000, help="watch-server port (default 8000).")
    ap.add_argument(
        "--compare",
        action="store_true",
        help="split-experiment mode: treat PATH as a base id (e.g. runs/exp0047) and overlay every "
        "sibling sweep arm (runs/exp0047*) — one chart per metric, a thin line per seed per arm.",
    )
    args = ap.parse_args()

    p = Path(args.path)
    # Slug from the run DIR name (exp0045), not a single log's stem (s0 -> "0"). --exp overrides.
    exp = args.exp or re.sub(r"\D", "", p.name if p.is_dir() else p.parent.name)
    want = set(re.split(r"[,\s]+", args.panels.strip())) if args.panels else None

    if args.watch:
        watch(p, exp, want, args.watch, args.host, args.port, args.compare)
        return

    if args.compare:
        arms = load_arms(p)
        if not arms:
            raise SystemExit(f"no sweep arms found matching {p}*")
        out_dir = Path("assets") / f"exp-{exp}"
        written = render_compare(arms, out_dir, p.name, want)
        print(
            f"wrote {len(written)} comparison chart(s) to {out_dir}/ for arms: "
            f"{', '.join(label for label, _ in arms)}"
        )
        for slug, metric in written:
            print(f"  {slug:<28} {metric}")
        return

    data, n_seeds, n_rounds = load(p)
    if data is None:
        raise SystemExit(f"no parseable round metrics under {p}")
    subtitle = f"exp{exp} ({n_seeds} seeds, {n_rounds} rounds)"
    out_dir = Path("assets") / f"exp-{exp}"

    # The 3-column overview is ALWAYS rendered (the bottom-of-page "all metrics" figure). Full-width
    # panels render ONLY for slugs the author selects via --panels, so committed assets are exactly
    # the figures the report uses (overview + featured panels) and nothing else.
    render_overview(data, out_dir, subtitle)
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
