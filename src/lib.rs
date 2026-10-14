//! tabicl-anywhere: TabICL (v2) on a CPU in safe Rust.
//!
//! * `fit` runs a training table through the network once and writes a cache bundle.
//! * `predict` streams that bundle (from a file, bytes in memory, or any byte-range source) to label new rows,
//!   holding only a bounded piece of it at a time.
//!
//! Layers, bottom up: `kern` (SIMD inner loops), `store` (the bundle file), `model` (weights), `nn` (blocks),
//! `fit` / `predict` (the two passes over an already-normalised table), `prep` (table preprocessing) and `table`
//! (the everyday classifier interface that ties them together).
#![forbid(unsafe_code)]
// The numeric kernels index several arrays with one loop counter, and slice into fixed-width pieces with
// `chunks_exact`: the clearest form for this arithmetic, and the one the compiler turns into vector code.
#![allow(clippy::needless_range_loop, clippy::chunks_exact_to_as_chunks)]

pub mod ckpt;
pub mod crypt;
pub mod csv;
pub mod fit;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod kern;
pub mod model;
pub mod nn;
pub mod predict;
pub mod prep;
pub mod store;
pub mod table;
pub mod wait;
pub mod warm;
