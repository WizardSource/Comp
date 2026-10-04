#!/usr/bin/env bash
# Regenerate patches/0001-language-extra-kurn.patch from the working tree of third_party/triton.
# The KOMP fork only adds files (listed in patches/FILES); upstream files stay untouched.
set -eu
KOMP=$(cd "$(dirname "$0")/.." && pwd)
T="$KOMP/third_party/triton"
mapfile -t files < <(grep -v '^#' "$KOMP/patches/FILES" | sed '/^$/d')
cd "$T"
git add -f -N "${files[@]}"
{
  echo "KOMP: KURN block formats for Triton (new files only, no upstream file modified)"
  echo "Base: triton $(git describe --tags 2>/dev/null || git rev-parse --short HEAD) ($(git rev-parse HEAD))"
  echo
  git diff -- "${files[@]}"
} > "$KOMP/patches/0001-language-extra-kurn.patch"
git reset -q -- "${files[@]}"
echo "wrote patches/0001-language-extra-kurn.patch ($(grep -c '^+[^+]' "$KOMP/patches/0001-language-extra-kurn.patch") added lines)"
