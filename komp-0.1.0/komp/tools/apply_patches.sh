#!/usr/bin/env bash
# Apply the KOMP patch series (patches/series) to the pinned upstream tree in third_party/triton.
# Idempotent: patches that are already applied are skipped.
set -eu
KOMP=$(cd "$(dirname "$0")/.." && pwd)
T="$KOMP/third_party/triton"
[ -e "$T/.git" ] || git -C "$KOMP/.." submodule update --init --depth 1 komp/third_party/triton
PIN=$(cat "$KOMP/patches/PIN")
HEAD=$(git -C "$T" rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "third_party/triton is at $HEAD, expected the pin $PIN" >&2; exit 1; }
while read -r p; do
  case "$p" in ''|'#'*) continue ;; esac
  if git -C "$T" apply -R --check "$KOMP/patches/$p" 2>/dev/null; then
    echo "already applied: $p"
  else
    git -C "$T" apply "$KOMP/patches/$p"
    echo "applied: $p"
  fi
done < "$KOMP/patches/series"
