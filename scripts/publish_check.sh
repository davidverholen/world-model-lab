#!/usr/bin/env bash
# Scan tracked files for personal/machine references that must not reach a public
# repo: home paths, tailscale CGNAT IPs, SSH keys, plus any extra patterns from the
# gitignored .publish-check.local (one grep -E pattern per line — put your personal
# hostnames there; they can't live in this committed script for the same reason).
# Used by /milestone (verify step). Exit 1 = findings.
set -euo pipefail
cd "$(dirname "$0")/.."

patterns=(
  '/home/[a-z0-9_-]+/'
  '100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]+\.[0-9]+' # tailscale CGNAT range
  'ssh-(rsa|ed25519|ecdsa) AAAA'
  '192\.168\.[0-9]+\.[0-9]+'
)
if [ -f .publish-check.local ]; then
  while IFS= read -r p; do
    [ -n "$p" ] && [ "${p:0:1}" != "#" ] && patterns+=("$p")
  done <.publish-check.local
fi

found=0
for p in "${patterns[@]}"; do
  if hits=$(git grep -nIE "$p" -- ':!uv.lock' 2>/dev/null); then
    echo "PATTERN: $p"
    echo "$hits" | sed 's/^/  /'
    found=1
  fi
done

if [ "$found" -eq 0 ]; then
  echo "publish-check: clean (tracked files contain no personal machine references)"
else
  echo "publish-check: FINDINGS above — fix before committing/publishing" >&2
  exit 1
fi
