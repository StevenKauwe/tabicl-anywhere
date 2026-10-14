## What this changes

<!-- One or two sentences, and the issue it addresses if there is one. -->

## How it was checked

<!-- Tests added or run. If numbers change: the difference from the reference (or from the existing path) before
     and after. If speed matters: what you measured, on what machine. -->

- [ ] `cargo fmt --all --check` and `cargo clippy --all-targets --features gpu -- -D warnings` are clean
- [ ] `cargo test --release` passes (and `--features gpu` on a machine with a graphics card, if card code changed)
- [ ] `node tools/test-wasm.cjs` passes, if the engine or `wasm/` changed
- [ ] CHANGELOG.md has a line under "Unreleased", if users will notice the change
