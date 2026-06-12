#!/usr/bin/env bash
# Download local PDF copies of every arXiv source in knowledge/sources/SOURCES.md
# into knowledge/sources/files/ (gitignored — personal reading archive, ~1-5 MB per
# paper). Idempotent: skips files that already exist. Re-run after new ingests.
set -euo pipefail
cd "$(dirname "$0")/.."
dest="knowledge/sources/files"
mkdir -p "$dest"

ids=$(grep -oE 'arxiv:[0-9]+\.[0-9]+' knowledge/sources/SOURCES.md | sort -u | cut -d: -f2)
total=0 fetched=0
for id in $ids; do
  total=$((total + 1))
  out="$dest/arxiv-$id.pdf"
  [ -s "$out" ] && continue
  echo "fetching $id ..."
  curl -fsSL --retry 2 "https://arxiv.org/pdf/$id" -o "$out" || {
    echo "  FAILED: $id" >&2
    rm -f "$out"
    continue
  }
  fetched=$((fetched + 1))
  sleep 2 # be polite to arXiv
done
echo "done: $fetched new, $((total - fetched)) already present, in $dest/"
