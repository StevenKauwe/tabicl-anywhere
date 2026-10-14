#!/bin/sh
# Put together everything the page in site/ needs:
#   site/pkg/            the WebAssembly engine (built by tools/build-wasm.sh)
#   site/weights/        half-size weights cut into parts of at most 14 MB, with a list of the parts
#   site/index.html      site/page.html wrapped as a complete document, for GitHub Pages or any static host
# (site/page.html itself is what is published as a Claude artifact, which adds its own wrapper.)
#
# Usage: tools/build-site.sh [CLASSIFIER REGRESSOR]
#   each a published .ckpt or a .tirs weight file; default: fixtures/classifier-v2.tirs fixtures/regressor-v2.tirs.
#   Weights already in site/weights are kept if no source is found.
set -e
cd "$(dirname "$0")/.."
T="${CARGO_TARGET_DIR:-target}"
sh tools/build-wasm.sh
cargo build --release --bin browsertabicl
mkdir -p site/weights
for pair in "classifier:${1:-fixtures/classifier-v2.tirs}" "regressor:${2:-fixtures/regressor-v2.tirs}"; do
  kind="${pair%%:*}"; src="${pair#*:}"
  if [ -f "$src" ]; then
    rm -f site/weights/$kind-v2-half.*
    "$T/release/browsertabicl" convert "$src" "$T/$kind-v2-half.tirs" --half
    "$T/release/browsertabicl" split "$T/$kind-v2-half.tirs" "site/weights/$kind-v2-half"
  else
    echo "no $src: leaving site/weights/$kind-v2-half.* as it is"
  fi
done
{
  cat <<'HEAD'
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BrowserTabICL</title>
<link rel="icon" href="data:,">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<!-- Generated from page.html by tools/build-site.sh: edit page.html, not this file. -->
</head>
<body>
HEAD
  cat site/page.html
  printf '</body>\n</html>\n'
} > site/index.html
touch site/.nojekyll
du -ah site | sort -k2
