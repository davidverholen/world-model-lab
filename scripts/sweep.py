"""Run a parameter sweep and aggregate results into a markdown table.

Replaces the hand-written shell loops we kept rewriting for multi-seed/multi-arm
experiments (exp 0002/0003 and the GPU benchmark). Cartesian product of --grid
axes, sequential execution (local or via scripts/remote.sh), numeric metrics
scraped from each run's final output lines, one markdown row per combo —
ready to paste into a knowledge/experiments/ page.

    uv run python scripts/sweep.py \
        --grid seed=0,1,2 --grid sigreg-weight=0,0.05 \
        --name exp0003 \
        -- python -m world_model.collect --env-id MiniGrid-Empty-8x8-v0 --updates 1000

    # remote (dispatches each combo through scripts/remote.sh run):
    uv run python scripts/sweep.py --remote --grid seed=0,1 -- python -m ...

Logs land in runs/sweeps/<name>/; the table is printed and saved alongside.
"""

import argparse
import itertools
import re
import subprocess
import sys
from pathlib import Path

METRIC_RE = re.compile(r"([A-Za-z_][\w ^/-]{0,30}?)\s*[=:]\s*(-?\d+(?:\.\d+)?(?:e-?\d+)?)\b")
TAIL_LINES = 15  # only scrape metrics from the end of the output


def parse_metrics(output: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for line in output.strip().splitlines()[-TAIL_LINES:]:
        for name, value in METRIC_RE.findall(line):
            metrics[name.strip()] = value
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--grid",
        action="append",
        required=True,
        metavar="param=v1,v2",
        help="sweep axis; flag name without leading dashes",
    )
    parser.add_argument("--name", default="sweep", help="results dir name under runs/sweeps/")
    parser.add_argument("--remote", action="store_true", help="dispatch via scripts/remote.sh run")
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="base command after '--' (e.g. python -m world_model.collect ...)",
    )
    args = parser.parse_args()

    base = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not base:
        parser.error("no base command given after --")

    axes: list[tuple[str, list[str]]] = []
    for spec in args.grid:
        param, _, values = spec.partition("=")
        axes.append((param, values.split(",")))

    out_dir = Path("runs/sweeps") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    combos = list(itertools.product(*(v for _, v in axes)))
    for i, combo in enumerate(combos):
        params = dict(zip([p for p, _ in axes], combo, strict=True))
        extra = [x for p, v in params.items() for x in (f"--{p}", v)]
        if args.remote:
            cmd = ["./scripts/remote.sh", "run", *base, *extra]
        else:
            cmd = ["uv", "run", *base, *extra]
        label = " ".join(f"{p}={v}" for p, v in params.items())
        print(f"[{i + 1}/{len(combos)}] {label}", flush=True)

        proc = subprocess.run(cmd, capture_output=True, text=True)
        output = proc.stdout + proc.stderr
        (out_dir / f"run_{i:03d}.log").write_text(f"# {label}\n# exit={proc.returncode}\n{output}")
        row = {**params, "exit": str(proc.returncode), **parse_metrics(output)}
        rows.append(row)

    columns = list(dict.fromkeys(k for row in rows for k in row))
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
        *("| " + " | ".join(row.get(c, "") for c in columns) + " |" for row in rows),
    ]
    table = "\n".join(lines)
    (out_dir / "RESULTS.md").write_text(table + "\n")
    print(f"\n{table}\n\nlogs + RESULTS.md in {out_dir}/")
    sys.exit(max(int(r["exit"]) for r in rows))


if __name__ == "__main__":
    main()
