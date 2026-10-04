#!/usr/bin/env bash
# Install the KOMP fork into a Python environment that has the matching upstream wheel (triton==3.8.0).
#
#   komp/tools/install_fork.sh [PYTHON]             apply the patch series, copy the files it adds into triton
#   komp/tools/install_fork.sh [PYTHON] --uninstall remove them again
#
# The series only adds pure-Python files, so the fork's compiled part (libtriton, LLVM) is byte-identical
# to the wheel and no source build is needed. If a patch ever modifies an upstream file, this script refuses
# and the fork must be built from third_party/triton instead (pip install ./komp/third_party/triton).
set -eu
KOMP=$(cd "$(dirname "$0")/.." && pwd)
PY=${1:-python3}
WANT=$(cat "$KOMP/patches/VERSION")
HAVE=$("$PY" -c 'import triton; print(triton.__version__)')
[ "$HAVE" = "$WANT" ] || { echo "$PY has triton $HAVE; the KOMP pin is $WANT (pip install triton==$WANT)" >&2; exit 1; }
SITE=$("$PY" -c 'import os, triton; print(os.path.dirname(os.path.dirname(triton.__file__)))')
mapfile -t files < <(grep -v '^#' "$KOMP/patches/FILES" | sed '/^$/d')
if [ "${2:-}" = "--uninstall" ]; then
  for f in "${files[@]}"; do rm -f "$SITE/${f#python/}"; rmdir "$(dirname "$SITE/${f#python/}")" 2>/dev/null || true; done
  echo "removed the KOMP files from $SITE/triton"
  exit 0
fi
if grep -h '^--- a/' $(sed '/^#/d;/^$/d;s|^|'"$KOMP"'/patches/|' "$KOMP/patches/series") | grep -q .; then
  echo "the patch series modifies upstream files: build the fork from source (pip install ./komp/third_party/triton)" >&2
  exit 1
fi
"$KOMP/tools/apply_patches.sh"
for f in "${files[@]}"; do
  install -D -m 644 "$KOMP/third_party/triton/$f" "$SITE/${f#python/}"
done
"$PY" -c 'from triton.language.extra import kurn; print("KOMP fork installed: triton", __import__("triton").__version__, "+ triton.language.extra.kurn")'
