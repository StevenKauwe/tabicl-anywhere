#!/bin/sh
# Build BrowserTabICL for the browser (pkg/web) and for Node (pkg/node). Needs: rustup target wasm32-unknown-unknown,
# and `cargo install wasm-bindgen-cli` at the version of wasm-bindgen in Cargo.lock.
set -e
cd "$(dirname "$0")/.."
RUSTFLAGS="-C target-feature=+simd128" cargo build --release -p browsertabicl-wasm --target wasm32-unknown-unknown
W="${CARGO_TARGET_DIR:-target}"/wasm32-unknown-unknown/release/browsertabicl_wasm.wasm
wasm-bindgen --target web --out-dir pkg/web "$W"
wasm-bindgen --target nodejs --out-dir pkg/node "$W"
# the page in site/ loads the browser build from site/pkg
mkdir -p site/pkg
cp pkg/web/browsertabicl_wasm.js pkg/web/browsertabicl_wasm_bg.wasm site/pkg/
ls -la pkg/web/*.wasm
