#!/bin/sh
# Build tabicl-anywhere for the browser (pkg/web) and for Node (pkg/node). Needs: rustup target wasm32-unknown-unknown,
# and `cargo install wasm-bindgen-cli` at the version of wasm-bindgen in Cargo.lock.
#
# Three builds of the same code, because a browser refuses a whole engine file if it holds a single instruction
# the browser does not know (site/worker.js asks the browser which ones it accepts and loads only that file):
#   relaxed   SIMD plus relaxed SIMD (fused multiply-add): Chrome and Edge among others. About 15% quicker to fit.
#   simd      plain 128-bit SIMD: every current browser, including Safari (all browsers on iPhone and iPad).
#   scalar    no SIMD at all: browsers from before 2021-2023. About 3 times slower.
# The JavaScript glue wasm-bindgen writes is the same for all three (checked below), so there is one copy of it.
# Each build has its own target directory, so switching between them does not rebuild everything.
set -e
cd "$(dirname "$0")/.."
T="${CARGO_TARGET_DIR:-target}"
build() { # name, target dir, rustflags, extra cargo arguments
  RUSTFLAGS="$3" cargo build --release -p tabicl-anywhere-wasm --target wasm32-unknown-unknown --target-dir "$2" $4
  W="$2"/wasm32-unknown-unknown/release/tabicl_anywhere_wasm.wasm
  # (the names of functions are left out of the browser build: 0.5 MB that only a debugger uses)
  wasm-bindgen --target web --remove-name-section --out-dir "pkg/web-$1" "$W"
  wasm-bindgen --target nodejs --out-dir "pkg/node-$1" "$W"
}
build simd "$T" "-C target-feature=+simd128" ""
build relaxed "$T/wasm-relaxed" "-C target-feature=+simd128,+relaxed-simd" "--features relaxed-simd"
build scalar "$T/wasm-scalar" "" ""
for b in relaxed scalar; do
  cmp -s pkg/web-simd/tabicl_anywhere_wasm.js pkg/web-$b/tabicl_anywhere_wasm.js || { echo "the $b build's JavaScript glue differs from the simd build's" >&2; exit 1; }
done
# the default names (what tools/test-wasm.cjs and older scripts use) are the simd build
rm -rf pkg/web pkg/node
cp -r pkg/web-simd pkg/web
cp -r pkg/node-simd pkg/node
# the page in site/ loads the browser build from site/pkg: one glue file, three engine files
mkdir -p site/pkg
rm -f site/pkg/*
cp pkg/web-simd/tabicl_anywhere_wasm.js pkg/web-simd/tabicl_anywhere_wasm_bg.wasm site/pkg/
cp pkg/web-relaxed/tabicl_anywhere_wasm_bg.wasm site/pkg/tabicl_anywhere_wasm_relaxed.wasm
cp pkg/web-scalar/tabicl_anywhere_wasm_bg.wasm site/pkg/tabicl_anywhere_wasm_scalar.wasm
ls -la site/pkg
