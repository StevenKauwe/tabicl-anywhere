# Contributing to BrowserTabICL

Thank you for helping. Bug reports, questions and pull requests are all welcome on the
[issue tracker](https://github.com/StevenKauwe/browsertabicl/issues). By taking part you agree to follow the
[code of conduct](CODE_OF_CONDUCT.md).

## What this project is

An independent implementation of the TabICL v2 tabular model in safe Rust (`#![forbid(unsafe_code)]` in the
core crate), for the command line, as a library, on a graphics card, and in the browser as WebAssembly. The
published PyTorch implementation of TabICL (the "reference") is used only as the thing to agree with: every
numerical part of this crate is tested against numbers the reference produced.

## Building

You need Rust 1.90 or newer (`rustup` installs it).

```
cargo build --release                    # command line and library, CPU only
cargo build --release --features gpu     # with graphics-card support (Vulkan, Metal, DirectX 12)
tools/build-wasm.sh                      # the browser and Node engine (needs the wasm32-unknown-unknown target
                                         # and wasm-bindgen-cli at the version in Cargo.lock)
tools/build-site.sh                      # the web page in site/; tools/deploy-docs.sh copies it to docs/
```

## Testing

```
cargo test --release                     # everything that runs on a CPU
cargo test --release --features gpu      # also the graphics-card tests (skipped when there is no card)
node tools/test-wasm.cjs                 # the WebAssembly engine under Node (after tools/build-wasm.sh)
node tools/test-site.mjs                 # the web page's own logic under Node
NODE_PATH=... node tools/test-browser.mjs   # the page in headless Chromium (see the file's header)
```

Use `--release`: the tests run real models and are slow without optimisation.

### Test data: what is committed and what is generated

* `tests/data/` (committed, about 0.6 MB) is enough for every comparison to run, which is what CI does:
  * `tiny-clf.ckpt`, `tiny-reg.ckpt`: tiny randomly initialised TabICL v2 checkpoints with the same architecture
    as the published ones (made by `tools/make_tiny.py`), and `tiny-*.tirs`, the reference's numbers for them.
    These are compared at full precision (differences of about 1e-6).
  * `table-b`, `many-a`, `reg-a`, `reg-b` `.tirs`: small tables with the reference's answers from the real
    published weights. Compared using the 16-bit copy of the real weights in `docs/weights/` (the web page's).
* `fixtures/` (not committed, about 1.2 GB) holds the full-precision published weights and larger tables. Tests
  that need it print `fixtures missing, skipping ...` and pass. To generate it you need the reference:

```
python -m venv ref/venv && ref/venv/bin/pip install tabicl          # the reference, in ref/ (git-ignored)
# the published checkpoints, from https://huggingface.co/jingang/TabICL, into ref/ckpt/
cargo run --release -- convert ref/ckpt/tabicl-classifier-v2-20260212.ckpt fixtures/classifier-v2.tirs
cargo run --release -- convert ref/ckpt/tabicl-regressor-v2-20260212.ckpt fixtures/regressor-v2.tirs
cargo run --release -- convert fixtures/classifier-v2.tirs fixtures/classifier-v2-half.tirs --half
cargo run --release -- convert fixtures/regressor-v2.tirs fixtures/regressor-v2-half.tirs --half
PY="ref/venv/bin/python -I"   # -I: never import from the current folder
CK=ref/ckpt/tabicl-classifier-v2-20260212.ckpt RK=ref/ckpt/tabicl-regressor-v2-20260212.ckpt
# arguments: train rows, test rows, features, [classes,] seed, [ensemble members]
$PY tools/dump_raw.py   $CK fixtures/raw-small.tirs 64 8 7 3 SEED
$PY tools/dump_raw.py   $CK fixtures/raw-mid.tirs 500 20 20 5 SEED
$PY tools/dump_table.py $CK fixtures/table-a.tirs 300 40 12 3 SEED 8
$PY tools/dump_table.py $CK fixtures/table-b.tirs 150 30 6 2 SEED 3
$PY tools/dump_table.py $CK fixtures/many-a.tirs 400 40 7 14 SEED 2 nocache
$PY tools/dump_table.py $CK fixtures/many-b.tirs 1500 40 6 120 SEED 1 nocache
$PY tools/dump_reg.py   $RK fixtures/reg-a.tirs 300 40 8 SEED 4
$PY tools/dump_reg.py   $RK fixtures/reg-b.tirs 120 25 5 SEED 1
```

The shapes above are those of the existing fixtures; the seeds they were made with were not recorded, and any
seed works (the tests read every number they compare from the fixture itself). A table with more than 10
classes needs `nocache`, as the reference cannot cache those. The copies of `table-b`, `many-a`, `reg-a` and
`reg-b` in `tests/data` are these fixtures.

To regenerate the tiny test data in `tests/data`:

```
$PY tools/make_tiny.py tests/data/tiny-clf.ckpt classifier 1
$PY tools/make_tiny.py tests/data/tiny-reg.ckpt regressor 2
$PY tools/dump_raw.py   tests/data/tiny-clf.ckpt tests/data/tiny-raw.tirs 120 16 7 3 5
$PY tools/dump_table.py tests/data/tiny-clf.ckpt tests/data/tiny-table.tirs 150 24 8 3 6 8
$PY tools/dump_table.py tests/data/tiny-clf.ckpt tests/data/tiny-many.tirs 200 16 7 14 7 2 nocache
$PY tools/dump_table.py tests/data/tiny-clf.ckpt tests/data/tiny-one.tirs 60 10 6 2 3 1
$PY tools/dump_reg.py   tests/data/tiny-reg.ckpt tests/data/tiny-reg-a.tirs 120 16 5 8 4
$PY tools/dump_reg.py   tests/data/tiny-reg.ckpt tests/data/tiny-reg-b.tirs 80 12 4 9 1
```

### Why the reference is not in the repository

`ref/` holds a copy of the reference's source, its Python environments (several GB with PyTorch) and the
published checkpoints (110 MB each). None of it is needed to build or use BrowserTabICL, it is third-party code
under its own licence, and the checkpoints are better fetched from their source. It is listed in `.gitignore`.
The weights the web page needs are committed in `docs/weights/` at 16-bit precision, with their licence.

### Coverage

```
cargo install cargo-llvm-cov
cargo llvm-cov --release --workspace --exclude browsertabicl-wasm --features gpu --html
```

CI publishes a CPU-only report on every run (its machines have no graphics card).

## Style

* `cargo fmt` (settings in `rustfmt.toml`: lines up to 200 columns) and
  `cargo clippy --all-targets --features gpu -- -D warnings` must be clean; CI checks both.
* No `unsafe` in the core crate. The `wasm/` crate exists because the generated JavaScript glue needs it.
* A change that touches numbers comes with a test against the reference, or against an existing path that is
  already tested (for example "the same as the CPU path to 1e-4").
* Say in the pull request what you measured: accuracy against the reference, and speed if it matters.

## Releases

Versions follow [Semantic Versioning](https://semver.org); changes are listed in [CHANGELOG.md](CHANGELOG.md).
