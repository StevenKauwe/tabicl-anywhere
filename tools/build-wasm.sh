#!/bin/sh
# Build tabicl-anywhere for the browser (pkg/web) and for Node (pkg/node). Needs: rustup target wasm32-unknown-unknown,
# and `cargo install wasm-bindgen-cli` at the version of wasm-bindgen in Cargo.lock.
set -e
cd "$(dirname "$0")/.."
RUSTFLAGS="-C target-feature=+simd128" cargo build --release -p tabicl-anywhere-wasm --target wasm32-unknown-unknown
W="${CARGO_TARGET_DIR:-target}"/wasm32-unknown-unknown/release/tabicl_anywhere_wasm.wasm
# (the names of functions are left out of the browser build: 0.5 MB that only a debugger uses)
wasm-bindgen --target web --remove-name-section --out-dir pkg/web "$W"
wasm-bindgen --target nodejs --out-dir pkg/node "$W"
# the page in site/ loads the browser build from site/pkg
mkdir -p site/pkg
cp pkg/web/tabicl_anywhere_wasm.js pkg/web/tabicl_anywhere_wasm_bg.wasm site/pkg/
ls -la pkg/web/*.wasm
