# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-14

First release.

### Added

- An independent implementation of the TabICL v2 classifier and regressor in safe Rust
  (`#![forbid(unsafe_code)]`), agreeing with the published PyTorch implementation to about 1e-5 on class
  probabilities and regression quantiles (measured differences are listed in the README).
- Fitting writes a single file (a "bundle") holding the preprocessing, the ensemble settings and the key/value
  cache; prediction streams it from a file, from memory, or from any byte-range source, holding about 12 MB.
- The reference's table preprocessing (imputation, constant-column removal, standardisation, Yeo-Johnson power
  transform, outlier squashing) and its ensemble (Latin-square feature orders, class shifts), matched member for
  member, including Python's random number generator.
- More than 10 classes (label digits and a class tree, as the reference does).
- 16-bit and 8-bit storage of the cache; half-size (16-bit) weights.
- Encryption at rest that still streams (XChaCha20-Poly1305 per 64 KiB block; see SECURITY.md).
- A warm in-memory tier with per-tenant byte budgets, and a graphics-memory tier.
- Optional graphics-card fitting and prediction (`--features gpu`: Vulkan, Metal, DirectX 12 and WebGPU through
  wgpu, portable WGSL shaders).
- A reader for published PyTorch checkpoints that needs no Python.
- The command-line program `tabicl-anywhere` (`convert`, `fit`, `predict`, `keygen`, `seal`, `split`, `join`, and
  benchmarks), reading CSV files with text columns and missing cells.
- WebAssembly bindings and a static web page that fits and predicts entirely in the visitor's browser, served
  from `docs/` by GitHub Pages, with the 16-bit weights split into parts of at most 14 MB.
- Tests against the reference's numbers that run without the large generated fixtures (a tiny committed model
  and small tables), property tests of the file formats and preprocessing, and CI on Linux, macOS and Windows.

[Unreleased]: https://github.com/StevenKauwe/tabicl-anywhere/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/StevenKauwe/tabicl-anywhere/releases/tag/v0.1.0
