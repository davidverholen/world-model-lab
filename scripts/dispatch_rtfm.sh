#!/usr/bin/env bash
# Launch a rung-4 RTFM experiment as N parallel seeds on the local GPU.
#
# Convention: every run's checkpoints AND logs live under runs/<exp>/ (gitignored),
# one file per seed — so it is always clear which experiment produced which
# checkpoint. No more flat soup of runs/*.pt with no attribution.
#
#   scripts/dispatch_rtfm.sh <exp> <nseeds> [train_rtfm args...]
#   e.g. scripts/dispatch_rtfm.sh exp0035 4 --rounds 20 --length 1 --one-shot
#
# Produces: runs/<exp>/s<seed>.pt  (checkpoint)  +  runs/<exp>/s<seed>.log  (log)
# Blocks until all seeds finish (run the whole script in the background to detach).
set -euo pipefail

exp="${1:?usage: dispatch_rtfm.sh <exp> <nseeds> [train args...]}"
nseeds="${2:?need nseeds}"
shift 2

dir="runs/$exp"
mkdir -p "$dir"
echo "dispatching $nseeds seeds of $exp -> $dir/  args: $*"
for s in $(seq 0 $((nseeds - 1))); do
  uv run python -m world_model.train_rtfm "$@" \
    --save "$dir/s$s.pt" --seed "$s" >"$dir/s$s.log" 2>&1 &
done
echo "pids: $(jobs -p | tr '\n' ' ')"
wait
echo "all $nseeds seeds of $exp done"
