//! The accelerator hooks (`fit::Attend`, `predict::AttendCached`), which the graphics card implements, exercised
//! with a stand-in written here on the CPU in the plainest way (straight loops, 64-bit sums). Fitting and
//! predicting through the hooks must agree with the ordinary CPU path. This checks the plumbing around the hooks
//! (batching, layouts, the fall-backs when an accelerator declines) on any machine, without a graphics card; the
//! card's own arithmetic is checked against the CPU in tests/card.rs.

mod common;
use common::*;
use std::io;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use tabicl_anywhere::fit::{Accel, Attend};
use tabicl_anywhere::model::Model;
use tabicl_anywhere::nn::{row_forward, Rope, RowWs};
use tabicl_anywhere::predict::{layer_rows, AttendCached, Cache, CardAttend, IclWeights, PredictOpts, Stats};
use tabicl_anywhere::store::{Bundle, MemStore};
use tabicl_anywhere::table::{fit_classifier_on, Classifier, ClassifierOpts};
use tabicl_anywhere::wait::{ready, Later};

/// Softmax attention of each query row against keys and values given row by row (`[n][d]`), per head.
fn attention(q: &[f32], k: &[f32], v: &[f32], d: usize, heads: usize) -> Vec<f32> {
    let (rows, n, hd) = (q.len() / d, k.len() / d, d / heads);
    let mut out = vec![0.0f32; rows * d];
    for r in 0..rows {
        for h in 0..heads {
            let s: Vec<f64> = (0..n).map(|j| (0..hd).map(|i| q[r * d + h * hd + i] as f64 * k[j * d + h * hd + i] as f64).sum()).collect();
            let top = s.iter().copied().fold(f64::MIN, f64::max);
            let w: Vec<f64> = s.iter().map(|x| (x - top).exp()).collect();
            let sum: f64 = w.iter().sum();
            for i in 0..hd {
                out[r * d + h * hd + i] = ((0..n).map(|j| w[j] * v[j * d + h * hd + i] as f64).sum::<f64>() / sum) as f32;
            }
        }
    }
    out
}

/// A fitting accelerator: does the in-context attention (from the stored chunk layout) and, optionally, the row
/// stage, and counts what it was asked to do.
struct Stand {
    rows_stage: bool,
    attended: AtomicUsize,
    row_calls: AtomicUsize,
}

impl Attend for Stand {
    fn attend<'a>(&'a self, q: &'a [f32], kv: &'a [f32], rows: usize, chunk: usize, d: usize, heads: usize) -> Later<'a, io::Result<Vec<f32>>> {
        self.attended.fetch_add(1, Relaxed);
        // `kv` is the layer in its stored chunks: for each chunk of `rc` rows, keys `[d][rc]` then values `[d][rc]`
        let (mut k, mut v) = (vec![0.0f32; rows * d], vec![0.0f32; rows * d]);
        let mut at = 0;
        for r0 in (0..rows).step_by(chunk) {
            let rc = chunk.min(rows - r0);
            for j in 0..d {
                for r in 0..rc {
                    k[(r0 + r) * d + j] = kv[at + j * rc + r];
                    v[(r0 + r) * d + j] = kv[at + d * rc + j * rc + r];
                }
            }
            at += 2 * d * rc;
        }
        ready(Ok(attention(q, &k, &v, d, heads)))
    }

    fn row_batch(&self, _m: &Model, _features: usize) -> usize {
        if self.rows_stage {
            7
        } else {
            0
        }
    }

    fn rows<'a>(&'a self, m: &'a Model, cells: &'a [f32], n: usize, reps: &'a mut [f32]) -> Later<'a, io::Result<()>> {
        self.row_calls.fetch_add(1, Relaxed);
        let features = cells.len() / (n * m.e);
        let tokens = m.cls + features;
        let rope = Rope::new(&m.rope_freqs, tokens);
        let mut ws = RowWs::new(m, tokens);
        let mut tok = vec![0.0f32; tokens * m.e];
        for r in 0..n {
            tok[..m.cls * m.e].copy_from_slice(&m.cls_tokens);
            tok[m.cls * m.e..].copy_from_slice(&cells[r * features * m.e..(r + 1) * features * m.e]);
            row_forward(pulp::Scalar::new(), m, &rope, &mut tok, tokens, &mut ws, &mut reps[r * m.d..(r + 1) * m.d]);
        }
        ready(Ok(()))
    }
}

/// A prediction accelerator: attention against a fitted model it has been given, read with `layer_rows` (what the
/// card's tier uses to load a model).
struct StandCached {
    bundle: Bundle,
    calls: AtomicUsize,
    /// When set, the accelerator declines every request after this many (as the card does while it loads).
    decline_after: Mutex<Option<usize>>,
}

impl AttendCached for StandCached {
    fn attend<'a>(&'a self, prefix: &'a str, node: usize, layer: usize, q: &'a [f32], d: usize, heads: usize) -> Later<'a, io::Result<Option<Vec<f32>>>> {
        let n = self.calls.fetch_add(1, Relaxed);
        if self.decline_after.lock().unwrap().is_some_and(|after| n >= after) {
            return ready(Ok(None));
        }
        let run = || -> io::Result<Option<Vec<f32>>> {
            let cache = Cache::open(&self.bundle, prefix)?;
            let (k, v) = layer_rows(&cache, node, layer, d)?;
            Ok(Some(attention(q, &k, &v, d, heads)))
        };
        ready(run())
    }
}

fn setup() -> (Model, Table) {
    (model(&tiny_weights("clf")), table(&data("tiny-table")))
}

fn fit(m: &Model, t: &Table, o: &ClassifierOpts) -> Vec<u8> {
    let out = MemStore::new();
    fit_classifier_on(m, &t.x, &t.y, t.n, t.h, Box::new(out.clone()), o, &[]).unwrap();
    out.bytes()
}

fn proba(m: &Model, cb: &Bundle, t: &Table, po: &PredictOpts) -> Vec<f64> {
    Classifier::open(cb).unwrap().predict_proba(m, &mut IclWeights::Resident, &t.xt, t.nt, po, &Stats::default()).unwrap()
}

#[test]
fn fitting_through_the_hooks_matches_the_cpu() {
    let (m, t) = setup();
    let o = ClassifierOpts { n_estimators: 2, ..Default::default() };
    let plain = bytes_bundle(fit(&m, &t, &o));
    let want = proba(&m, &plain, &t, &PredictOpts::default());
    for rows_stage in [false, true] {
        let stand = Arc::new(Stand { rows_stage, attended: AtomicUsize::new(0), row_calls: AtomicUsize::new(0) });
        let mut oa = o.clone();
        oa.fit.accel = Accel(Some(stand.clone()));
        oa.fit.chunk = 37; // chunks that do not divide the rows evenly
        let got = proba(&m, &bytes_bundle(fit(&m, &t, &oa)), &t, &PredictOpts::default());
        let bad = worst(&got, &want);
        println!("fitting through the hooks (row stage too: {rows_stage}): worst probability difference {bad:.2e}");
        // the stand-in sums in another order (and in 64 bits); that is all that separates the two
        assert!(bad < 1e-5, "{bad:.2e}");
        assert_eq!(stand.attended.load(Relaxed), 2 * (m.icl_layers - 1), "every layer but the last, for each member");
        assert_eq!(stand.row_calls.load(Relaxed) > 0, rows_stage);
        assert!(format!("{:?}", oa.fit.accel).contains("accelerated"));
    }
    // an accelerator is not used for a layer that does not fit in memory: the CPU does it, same answers
    let stand = Arc::new(Stand { rows_stage: false, attended: AtomicUsize::new(0), row_calls: AtomicUsize::new(0) });
    let mut oa = o.clone();
    oa.fit.accel = Accel(Some(stand.clone()));
    oa.fit.mem_budget = 1;
    oa.fit.scratch = Some(std::env::temp_dir());
    assert_eq!(fit(&m, &t, &oa), fit(&m, &t, &ClassifierOpts { fit: tabicl_anywhere::fit::FitOpts { mem_budget: 1, ..Default::default() }, ..o.clone() }));
    assert_eq!(stand.attended.load(Relaxed), 0);
}

#[test]
fn prediction_through_the_hooks_matches_the_cpu() {
    let (m, t) = setup();
    for kv in [tabicl_anywhere::fit::Kv::F32, tabicl_anywhere::fit::Kv::I8] {
        let mut o = ClassifierOpts { n_estimators: 2, ..Default::default() };
        o.fit.kv = kv;
        let bytes = fit(&m, &t, &o);
        let cb = bytes_bundle(bytes.clone());
        let want = proba(&m, &cb, &t, &PredictOpts::default());
        let stand = Arc::new(StandCached { bundle: bytes_bundle(bytes), calls: AtomicUsize::new(0), decline_after: Mutex::new(None) });
        let po = PredictOpts { accel: Some(CardAttend(stand.clone())), ..Default::default() };
        assert_eq!(format!("{:?}", po.accel.as_ref().unwrap()), "accelerated");
        let st = Stats::default();
        let got = Classifier::open(&cb).unwrap().predict_proba(&m, &mut IclWeights::Resident, &t.xt, t.nt, &po, &st).unwrap();
        let bad = worst(&got, &want);
        println!("{kv:?}: prediction through the hook: worst probability difference {bad:.2e}");
        assert!(bad < 1e-5, "{bad:.2e}");
        assert_eq!(st.card_layers.load(Relaxed) as usize, 2 * m.icl_layers);
        // an accelerator that stops answering part way (as the card does while it loads a model): the CPU takes
        // over for those layers, and the answers stay the same
        *stand.decline_after.lock().unwrap() = Some(stand.calls.load(Relaxed) + 1);
        let got = proba(&m, &cb, &t, &po);
        assert!(worst(&got, &want) < 1e-5);
    }
}
