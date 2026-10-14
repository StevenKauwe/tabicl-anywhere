#!/bin/sh
# Rebuild the page (tools/build-site.sh) and copy exactly the deployable files into docs/, which GitHub Pages serves
# from the main branch at https://stevenkauwe.github.io/tabicl-anywhere/. Commit docs/ afterwards.
#
# Usage: tools/deploy-docs.sh [CLASSIFIER REGRESSOR]     (passed on to tools/build-site.sh)
#        tools/deploy-docs.sh --no-build                 (only copy what is already in site/)
#
# If site/weights holds no parts (a fresh clone without fixtures/ or checkpoints), docs/weights is left as it is.
set -e
cd "$(dirname "$0")/.."
if [ "$1" = "--no-build" ]; then shift; else sh tools/build-site.sh "$@"; fi

for f in site/index.html site/page.html site/app.js site/worker.js \
         site/pkg/tabicl_anywhere_wasm.js site/pkg/tabicl_anywhere_wasm_bg.wasm site/pkg/tabicl_anywhere_wasm_relaxed.wasm \
         site/pkg/tabicl_anywhere_wasm_scalar.wasm site/weights/LICENSE-TabICL.txt; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
done

mkdir -p docs/pkg docs/weights
for f in index.html page.html app.js worker.js; do cp site/$f docs/$f; done
rm -f docs/pkg/*
# one glue file and the engine's three builds (a visitor's browser downloads only one of them; site/worker.js)
cp site/pkg/tabicl_anywhere_wasm.js site/pkg/tabicl_anywhere_wasm_bg.wasm site/pkg/tabicl_anywhere_wasm_relaxed.wasm site/pkg/tabicl_anywhere_wasm_scalar.wasm docs/pkg/
cp site/weights/LICENSE-TabICL.txt docs/weights/
for kind in classifier regressor; do
  if ls site/weights/$kind-v2-half.*.bin >/dev/null 2>&1; then
    rm -f docs/weights/$kind-v2-half.*
    cp site/weights/$kind-v2-half.json site/weights/$kind-v2-half.*.bin docs/weights/
  else
    echo "no weight parts in site/weights for $kind: leaving docs/weights/$kind-v2-half.* as it is"
  fi
done
: > docs/.nojekyll

# GitHub refuses files over 100 MB and warns over 50 MB; the parts are at most 14 MB.
big=$(find docs -type f -size +50M)
[ -z "$big" ] || { echo "files over 50 MB in docs/: $big" >&2; exit 1; }
du -ah docs | sort -k2
