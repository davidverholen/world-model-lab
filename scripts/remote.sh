#!/usr/bin/env bash
# Dispatch world-model runs to the GPU desktop over SSH (Tailscale).
# Design: knowledge/decisions/0004-remote-dispatch.md. Windows-side setup: docs/REMOTE.md.
#
#   scripts/remote.sh doctor           show resolved config, connectivity, remote state
#   scripts/remote.sh setup            one-time: bare repo + clone + uv sync on the desktop
#   scripts/remote.sh gpu              nvidia-smi on the desktop
#   scripts/remote.sh run <cmd...>     push HEAD, sync remote checkout, run command there
#                                      e.g. scripts/remote.sh run python -m world_model.collect
#   scripts/remote.sh shell '<bash>'   like run, but raw bash: enables 'a & b & wait'
#                                      for parallel seeds on the one mostly-idle GPU
#   scripts/remote.sh kill             kill remote python (runs survive ssh disconnect!)
#   scripts/remote.sh pull             copy remote runs/ back to ./runs/remote/
#
# Dispatches committed state only (HEAD) — uncommitted changes stay local on purpose;
# it keeps every remote run reproducible by commit hash (matches the milestone process).
set -euo pipefail

# config precedence: env vars > .env.remote (gitignored) > defaults
_remote_env="${WM_REMOTE:-}"
_remote_dir_env="${WM_REMOTE_DIR:-}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$script_dir/../.env.remote" ]; then
  # shellcheck source=/dev/null
  . "$script_dir/../.env.remote"
fi
REMOTE="${_remote_env:-${WM_REMOTE:-wm-desktop}}"
REMOTE_DIR="${_remote_dir_env:-${WM_REMOTE_DIR:-world-model}}"

cmd="${1:-help}"
shift || true

case "$cmd" in
doctor)
  echo "config resolution (env > .env.remote > defaults):"
  echo "  WM_REMOTE     = $REMOTE"
  echo "  WM_REMOTE_DIR = $REMOTE_DIR"
  echo "ssh resolves '$REMOTE' to:"
  ssh -G "$REMOTE" 2>/dev/null | awk '/^(hostname|user|port|identityfile) /{print "  "$0}'
  echo "connectivity:"
  if out=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "$REMOTE" \
    "echo \"  shell: \$0 on \$(uname -s)\" && git -C $REMOTE_DIR rev-parse --short HEAD 2>/dev/null | sed 's/^/  remote checkout @ /' && nvidia-smi --query-gpu=name,memory.total,temperature.gpu --format=csv,noheader | sed 's/^/  gpu: /'" 2>&1); then
    echo "$out"
    echo "  local HEAD       @ $(git rev-parse --short HEAD)$([ -n "$(git status --porcelain)" ] && echo ' (dirty — run would refuse)')"
  else
    echo "  UNREACHABLE: $out"
  fi
  ;;
setup)
  # Idempotent: the entire remote state (bare repo + checkout + venv) is a cache
  # owned by this script — delete it on the desktop anytime and re-run setup.
  ssh "$REMOTE" "git init --bare world-model.git"
  git remote add desktop "$REMOTE:world-model.git" 2>/dev/null ||
    git remote set-url desktop "$REMOTE:world-model.git"
  git push desktop HEAD:refs/heads/master -f
  # bare-init may default HEAD to 'main'; clones would check out nothing
  ssh "$REMOTE" "git -C world-model.git symbolic-ref HEAD refs/heads/master"
  ssh "$REMOTE" "git clone world-model.git $REMOTE_DIR" 2>/dev/null ||
    ssh "$REMOTE" "cd $REMOTE_DIR && git fetch origin master && git reset --hard origin/master"
  ssh "$REMOTE" "cd $REMOTE_DIR && uv sync"
  ssh "$REMOTE" "cd $REMOTE_DIR && uv run python -c \"import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\""
  ;;
gpu)
  ssh "$REMOTE" "nvidia-smi"
  ;;
run)
  [ -z "$(git status --porcelain)" ] || {
    echo "working tree dirty — commit first (remote runs are pinned to a commit)" >&2
    exit 1
  }
  git push desktop HEAD:refs/heads/dispatch -f
  echo "running on $REMOTE @ $(git rev-parse --short HEAD): uv run $*"
  # PYTHONUNBUFFERED: stream logs live instead of 4KB block-buffering (negligible cost —
  # we print ~per-200-updates, not in a hot loop). Detached runs otherwise look empty mid-run.
  ssh "$REMOTE" "cd $REMOTE_DIR && git fetch origin dispatch && git reset --hard FETCH_HEAD && uv sync && export PYTHONUNBUFFERED=1 && uv run $*"
  ;;
shell)
  [ -z "$(git status --porcelain)" ] || {
    echo "working tree dirty — commit first (remote runs are pinned to a commit)" >&2
    exit 1
  }
  git push desktop HEAD:refs/heads/dispatch -f
  echo "shell on $REMOTE @ $(git rev-parse --short HEAD): $*"
  # export PYTHONUNBUFFERED so backgrounded per-seed python subshells inherit live logging.
  ssh "$REMOTE" "cd $REMOTE_DIR && git fetch origin dispatch && git reset --hard FETCH_HEAD && uv sync && export PYTHONUNBUFFERED=1 && $*"
  ;;
kill)
  ssh "$REMOTE" "taskkill //IM python.exe //F" || true
  ;;
pull)
  mkdir -p runs/remote
  scp -r "$REMOTE:$REMOTE_DIR/runs/*" runs/remote/ || echo "(no remote runs yet)"
  echo "remote results in runs/remote/"
  ;;
*)
  grep "^#   " "$0" | sed 's/^#   //'
  ;;
esac
