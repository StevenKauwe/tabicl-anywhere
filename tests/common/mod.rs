//! Where the tests find their data.
//!
//! * `tests/data/` (committed, about 0.6 MB): tiny randomly initialised TabICL checkpoints (`tiny-clf.ckpt`,
//!   `tiny-reg.ckpt`, made by `tools/make_tiny.py`) with the reference's numbers for them (`tiny-*.tirs`), and a
//!   few small tables with the reference's answers from the real published weights (`table-b`, `many-a`, `reg-a`,
//!   `reg-b`). Always present, so these comparisons run everywhere, CI included.
//! * `docs/weights/` (committed for the web page): the real weights at 16-bit precision, in parts.
//! * `fixtures/` (not committed, about 1.2 GB, made by `tools/dump_*.py`): full-precision weights and the larger
//!   tables. Tests that need it skip with a note when it is absent.
#![allow(dead_code)]

use browsertabicl::model::{Load, Model};
use browsertabicl::store::Bundle;
use std::path::PathBuf;
use std::sync::Arc;

pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// A bundle in `tests/data/` (must exist: these files are committed).
pub fn data(name: &str) -> Bundle {
    Bundle::open(root().join("tests/data").join(format!("{name}.tirs"))).unwrap_or_else(|e| panic!("tests/data/{name}.tirs: {e}"))
}

/// A bundle in `fixtures/`, if it has been generated.
pub fn fixture(name: &str) -> Option<Bundle> {
    Bundle::open(root().join("fixtures").join(format!("{name}.tirs"))).ok()
}

pub fn bytes_bundle(bytes: Vec<u8>) -> Bundle {
    Bundle::from_source(Box::new(Arc::new(bytes))).unwrap()
}

/// The tiny model's weight bundle (`clf` or `reg`), converted from its PyTorch checkpoint by our own reader.
pub fn tiny_weights(kind: &str) -> Bundle {
    let ckpt = std::fs::read(root().join(format!("tests/data/tiny-{kind}.ckpt"))).unwrap();
    bytes_bundle(browsertabicl::ckpt::bundle_bytes(ckpt, false).unwrap())
}

/// The real 16-bit weights from `docs/weights/` (`classifier` or `regressor`), joined from their parts.
pub fn docs_weights(kind: &str) -> Option<Bundle> {
    let dir = root().join("docs/weights");
    let manifest = std::fs::read_to_string(dir.join(format!("{kind}-v2-half.json"))).ok()?;
    let bytes = browsertabicl::store::join_parts(&manifest, |name| std::fs::read(dir.join(name))).unwrap();
    Some(bytes_bundle(bytes))
}

pub fn model(b: &Bundle) -> Model {
    Model::load(b, Load::Fit).unwrap()
}

pub fn f64s(fx: &Bundle, n: &str) -> Vec<f64> {
    bytemuck::pod_collect_to_vec(&fx.bytes(n).unwrap())
}

pub fn worst(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

pub fn worst32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// A fresh scratch path in the system's temporary folder.
pub fn scratch(what: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    std::env::temp_dir().join(format!("BrowserTabICL-test-{what}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)))
}

/// A table fixture's training rows, class indices (or numbers) and test rows.
pub struct Table {
    pub x: Vec<f64>,
    pub y: Vec<usize>,
    pub yf: Vec<f64>,
    pub n: usize,
    pub h: usize,
    pub xt: Vec<f64>,
    pub nt: usize,
    pub members: usize,
}

pub fn table(fx: &Bundle) -> Table {
    let x = f64s(fx, "x_train");
    let (y, yf) = if fx.entry("y_train").unwrap().dtype == "int32" {
        let y = fx.i32s("y_train").unwrap();
        (browsertabicl::table::encode_labels(&y.iter().map(|v| *v as i64).collect::<Vec<_>>()).0, Vec::new())
    } else {
        (Vec::new(), f64s(fx, "y_train"))
    };
    let n = y.len().max(yf.len());
    let h = x.len() / n;
    let xt = f64s(fx, "x_test");
    let nt = xt.len() / h;
    let members = (0..).take_while(|i| fx.has(&format!("member{i}_method"))).count();
    Table { x, y, yf, n, h, xt, nt, members }
}
