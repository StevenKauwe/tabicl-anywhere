# BrowserTabICL

An independent implementation of the TabICL v2 tabular model (classification and regression) in safe Rust
(`#![forbid(unsafe_code)]` in the core). It runs natively on CPU, optionally with a graphics card, and in the
browser as WebAssembly. The published PyTorch implementation is used only as the thing to agree with.

* **Fit** runs the training table through the network once and writes one file (a "bundle"): preprocessing
  numbers, ensemble settings and the key/value cache, laid out in the order prediction reads it.
* **Predict** streams that bundle front to back, a bounded piece at a time, from a file, bytes in memory, or
  anything that serves a byte range (`store::ReadAt`), holding about 12 MB.
* Optional: 16-bit or 8-bit cache, encryption at rest that still streams, a warm in-memory tier with per-tenant
  budgets, graphics-card fitting.

## Try it in your browser

https://stevenkauwe.github.io/browsertabicl/ fits a model on your own CSV table and predicts new rows, entirely in
your browser; nothing is uploaded. GitHub Pages serves it from `docs/` on `main`, which `tools/deploy-docs.sh`
rebuilds from `site/` (see `site/README.md`).

## Everyday use

```
browsertabicl convert tabicl-classifier-v2-20260212.ckpt classifier-v2.tirs     # once; no Python needed

browsertabicl fit classifier-v2.tirs train.csv --target grade model.tirs --kv i16
browsertabicl predict classifier-v2.tirs model.tirs new.csv --stats > predictions.csv

browsertabicl keygen key.txt
browsertabicl seal model.tirs model.sealed key.txt
browsertabicl predict classifier-v2.tirs model.sealed new.csv --key key.txt
```

With the regression checkpoint the same two commands fit a numeric target and print mean, median and a 5-95% band.
Text columns become categories, empty cells are missing values, more than 10 classes are handled.

Library entry points: `table::{fit_classifier, Classifier, fit_regressor, Regressor}`; `predict::Stats` (atomic
counters for your metrics); `warm::Warm` (the in-memory tier); `store::{ReadAt, Storage, MemStore}` (bring your
own storage); `crypt::Sealed`; `gpu::Gpu` behind `--features gpu`.

## Browser / Node

```
tools/build-wasm.sh          # pkg/web and pkg/node, 1.4 MB of WebAssembly
node tools/test-wasm.cjs     # fits and predicts under Node and compares with the reference
```

```js
const weights = new Weights(bytesOfCkptOrTirs);            // accepts the published .ckpt directly
const fitted = weights.fitClassifier(x, y, rows, features, 8, "i16");   // Uint8Array you can store anywhere
const proba = weights.predictProba(fitted, xNew, newRows);
```

Single-threaded for now, and not yet tried in an actual browser (verified under Node).

## Does it agree with the reference?

`cargo test --release` compares against numbers dumped from the reference (`tools/dump_*.py`; fixtures are
generated, not checked in):

| what | worst difference |
|---|---|
| preprocessing (plain / power transform) | 1e-14 / 7e-8 |
| every cached key and value, raw scores | 5e-5 (the reference's own two paths differ by 5e-6) |
| class probabilities, 8-member ensemble, missing cells, skewed columns | 2e-6 |
| 14 and 120 classes (label digits + class tree) | 2e-5 |
| regression: mean, variance, median, quantiles incl. tails | 6e-5 |
| WebAssembly build, probabilities | 2e-6 |
| Rust checkpoint reader vs the Python export | byte-identical |

## Measured (i7-11700K, RTX 3090; 10,000 training rows x 100 features, 10 classes, one ensemble member)

| | BrowserTabICL | reference (PyTorch, CPU) |
|---|---|---|
| fit, CPU | 22.7 s, 263 MB peak | 20.5 s (54.6 s on one thread), 6.3 GB peak |
| cache on disk | 531 MB f32 / 287 MB i16 / 164 MB i8 | (in memory) |
| one row, one thread, streamed from disk | 57 ms f32 / 48 ms i16 / 42 ms i8, 12 MB | 31 ms, cache in memory |
| same, 4 reader threads | 40 / 30 / 29 ms, 14-15 MB | 27 ms (16 threads) |
| one row, warm tier (cache held in RAM) | 34 ms f32, 32 ms i16, 22 ms i16 with 4 threads | |
| 64 rows | 174 ms streamed, 145 ms warm | 88 ms (321 ms on one thread) |
| one row, model resident on the card | 8 ms | 16 ms (PyTorch CUDA, same 3090) |
| 64 / 256 / 2,048 rows on the card | 16 / 35 / 240 ms | 16 / 17 / 71 ms (PyTorch CUDA) |
| fit on the card | 2.5 s, 726 MB peak | 1.3 s (PyTorch CUDA) |
| model in graphics memory | 285 MB (16-bit) | |
| one row from an encrypted i16 cache | 261 ms (116 ms with 8 readers) | n/a |

i16 changes raw scores by 1e-4 and i8 by 1e-2 against f32; both gave the same answer on all 64 test rows.
The card's answers differ from the CPU's by at most 1e-4 in raw score (same answer on all 2,048 test rows), and a
cache fitted on the card predicts the same as one fitted on the CPU to 1e-4.
Low-memory fitting (`--mem`) writes a byte-identical cache and takes about twice as long.

## Not done yet

* Graphics card: prediction runs all three stages on it in one submission, from a tenant-scoped graphics-memory
  tier (`gpu::VramTier`; the first prediction is answered by the CPU while the card loads), and fitting runs all
  three stages on it too (`--gpu`; label-digit tables keep their column stage on the CPU). The shaders are
  portable WGSL: blocks of queries against blocks of keys, blocks of rows against blocks of weights. Up to 64 rows
  that matches or beats PyTorch's CUDA build; at 2,048 rows it is 3.4x behind and fitting is 1.9x behind. What is
  left is mostly arithmetic the card does at about a seventh of its peak rate (PyTorch uses half precision and
  the card's matrix hardware, which WGSL cannot reach), plus, in fitting, about 0.5 s of copying cells between
  main and graphics memory. Tried and not kept: holding all attention scores in graphics memory between two
  passes (moving them cost more than the arithmetic), 4 x 8 register blocks, four queries per invocation in the
  in-context attention. Not tried: folding each layer norm into the matrix after it, half-precision floats where
  the card offers them. The working buffers of the last call (about 3 GB after 2,048 rows) stay allocated
  outside the tier's budget until the next call. Also open: WebGPU in the browser (needs the loops made
  asynchronous).
* Browser: threads, a demo page, streaming a model over the network instead of holding it in memory.
* Gradient training (pre-training or fine-tuning).
* Speed on batches: the reference's matrix library is still about 2x ahead on 64 rows.

## Layout

`kern` inner loops (generic over `pulp::Simd`: AVX2, AVX-512, NEON, WebAssembly SIMD) - `store` bundle file and
storage traits - `crypt` sealing - `warm` in-memory tier - `ckpt` checkpoint reader - `model` weights - `nn`
blocks - `fit` / `predict` network passes - `gpu` (optional) - `prep` preprocessing and Python-compatible shuffles -
`csv` - `table` classifier and regressor - `bin/browsertabicl` command line - `wasm/` JavaScript bindings.

## Licences

* The code in this repository is ours: BSD 3-Clause, Copyright (c) 2026, Steven Kauwe (`LICENSE`).
* The model weights (`docs/weights/*.bin`, a half-precision copy of the published TabICL v2 checkpoints) are
  TabICL's, Copyright (c) 2025, Soda team @ Inria, and are redistributed under their BSD 3-Clause licence, which is
  kept beside them (`docs/weights/LICENSE-TabICL.txt`, also `site/weights/`).
* BrowserTabICL is an independent implementation. It is not affiliated with or endorsed by the TabICL authors.
