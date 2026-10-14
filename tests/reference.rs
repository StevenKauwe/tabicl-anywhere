//! Compare against numbers dumped from the reference implementation (tools/dump_raw.py, dump_table.py,
//! dump_reg.py).
//!
//! * `tiny_*`: a tiny randomly initialised model and the reference's numbers for it (tests/data, committed),
//!   at full precision: runs everywhere, CI included.
//! * `docs_half_*`: the real model at 16-bit precision (docs/weights, committed) on small tables whose reference
//!   answers came from the full-precision weights (tests/data): also runs everywhere.
//! * the rest: the real full-precision weights and larger tables in fixtures/ (not committed; skipped with a note
//!   when absent).
//!
//! `cargo test --release --test reference -- --nocapture` prints every measured difference.

mod common;
use common::*;

use tabicl_anywhere::fit::{fit_raw, raw_entries, FitOpts};
use tabicl_anywhere::model::{Load, Model};
use tabicl_anywhere::predict::{embed, in_context, represent, Cache, IclWeights, PredictOpts, Stats};
use tabicl_anywhere::store::{Bundle, BundleWriter};

fn worst(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Reference keys/values are `[heads][n][hd]` (or `[cols][heads][n][hd]`); ours are `[d][n]`.
fn to_ours(r: &[f32], heads: usize, n: usize, hd: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; r.len()];
    for h in 0..heads {
        for i in 0..n {
            for j in 0..hd {
                o[(h * hd + j) * n + i] = r[(h * n + i) * hd + j];
            }
        }
    }
    o
}

/// Every stage of the network against the reference: column and in-context caches, test embeddings, row
/// representations and raw scores. `tol` bounds every difference; `reopen` gives the weights again for the
/// streamed-weights path.
fn check(name: &str, wb: &Bundle, fx: &Bundle, reopen: impl Fn() -> Bundle, tol: f32) {
    let m = Model::load(wb, Load::Fit).unwrap();
    let (x, y) = (fx.f32s("x_train").unwrap(), fx.i32s("y_train").unwrap());
    let (n, h) = (y.len(), x.len() / y.len());
    let classes = *y.iter().max().unwrap() as usize + 1;
    let out = scratch(name);
    let opts = FitOpts { chunk: 48, ..Default::default() };
    let yf: Vec<f32> = y.iter().map(|v| *v as f32).collect();
    let plan = tabicl_anywhere::fit::Plan::new(&m, &yf, classes);
    let ents = raw_entries(&m, "", n, h, &opts, &plan);
    let ents: Vec<(&str, &str, Vec<usize>)> = ents.iter().map(|(a, b, c)| (a.as_str(), b.as_str(), c.clone())).collect();
    let w = BundleWriter::create(&out, &ents).unwrap();
    fit_raw(&m, &x, &yf, n, h, classes, &w, "", &opts, &plan).unwrap();
    drop(w);
    let cb = Bundle::open(&out).unwrap();

    // column cache: reference has the 4 class-token columns first; we store only real feature groups
    let ours = cb.f32s("col_kv").unwrap();
    let (e, inds, heads) = (m.e, m.inds, m.col_heads);
    let per = 2 * e * inds;
    for b in 0..m.col.len() {
        for (which, kind) in ["col_k", "col_v"].iter().enumerate() {
            let r = fx.f32s(&format!("{kind}{b}")).unwrap();
            let mut bad = 0.0f32;
            for g in 0..h {
                let want = to_ours(&r[(g + m.cls) * e * inds..(g + m.cls + 1) * e * inds], heads, inds, e / heads);
                let at = g * m.col.len() * per + b * per + which * e * inds;
                bad = bad.max(worst(&ours[at..at + e * inds], &want));
            }
            println!("{name} {kind}{b}: worst difference {bad:.2e}");
            assert!(bad < tol, "{kind}{b}");
        }
    }
    // in-context cache
    let d = m.d;
    for l in 0..m.icl_layers {
        let got: Vec<f32> = bytemuck::pod_collect_to_vec(&cb.bytes(&format!("n0_kv{l}")).unwrap());
        for (which, kind) in ["icl_k", "icl_v"].iter().enumerate() {
            let want = to_ours(&fx.f32s(&format!("{kind}{l}")).unwrap(), m.icl_heads, n, d / m.icl_heads);
            let mut bad = 0.0f32;
            for (ci, c) in got.chunks(2 * d * opts.chunk).enumerate() {
                let rc = c.len() / (2 * d);
                for j in 0..d {
                    let a = &c[which * d * rc + j * rc..which * d * rc + (j + 1) * rc];
                    bad = bad.max(worst(a, &want[j * n + ci * opts.chunk..j * n + ci * opts.chunk + rc]));
                }
            }
            println!("{name} {kind}{l}: worst difference {bad:.2e}");
            assert!(bad < 2.5 * tol, "{kind}{l}");
        }
    }
    // prediction, resident and streamed weights
    let xt = fx.f32s("x_test").unwrap();
    let nt = xt.len() / h;
    let cache = Cache::open(&cb, "").unwrap();
    let po = PredictOpts::default();
    let mut tok = embed(&m, &cache, &xt, nt, &po).unwrap();
    let want = fx.f32s("test_emb").unwrap();
    let tokens = m.cls + h;
    let mut bad = 0.0f32;
    for r in 0..nt {
        bad = bad.max(worst(&tok[(r * tokens + m.cls) * e..(r + 1) * tokens * e], &want[(r * tokens + m.cls) * e..(r + 1) * tokens * e]));
    }
    println!("{name} test embeddings: worst difference {bad:.2e}");
    assert!(bad < tol);
    let mut rep = represent(&m, &mut tok, nt);
    let bad = worst(&rep, &fx.f32s("test_rep").unwrap());
    println!("{name} test row representations: worst difference {bad:.2e}");
    assert!(bad < tol);
    let mut rep2 = rep.clone();
    let want = fx.f32s("logits").unwrap();
    let logits = in_context(&m, &mut IclWeights::Resident, &cache, 0, &mut rep, nt, &po, &Stats::default()).unwrap();
    let bad = worst(&logits, &want);
    println!("{name} logits: worst difference {bad:.2e} (values up to {:.1})", want.iter().fold(0.0f32, |a, b| a.max(b.abs())));
    assert!(bad < tol);
    let mut streamed = IclWeights::Streamed { bundle: reopen(), buf: Vec::new() };
    let logits2 = in_context(&m, &mut streamed, &cache, 0, &mut rep2, nt, &po, &Stats::default()).unwrap();
    assert_eq!(logits, logits2, "streamed weights must give identical numbers");
    let _ = std::fs::remove_file(out);
}

fn check_fixture(name: &str) {
    let (Some(wb), Some(fx)) = (fixture("classifier-v2"), fixture(name)) else {
        eprintln!("fixtures missing, skipping {name}");
        return;
    };
    check(name, &wb, &fx, || fixture("classifier-v2").unwrap(), 2e-3);
}

#[test]
fn small_table() {
    check_fixture("raw-small");
}

#[test]
fn mid_table() {
    check_fixture("raw-mid");
}

#[test]
fn tiny_raw() {
    check("tiny-raw", &tiny_weights("clf"), &data("tiny-raw"), || tiny_weights("clf"), TINY);
}

/// The tiny model is run at full precision on both sides, so only rounding separates the two: what remains is
/// summation order (SIMD lanes against PyTorch's kernels) over a few layers, about 1e-6 on numbers of order 1.
const TINY: f32 = 5e-5;

/// The whole classifier (preprocessing, ensemble, averaging) against the reference's `TabICLClassifier`.
/// Preprocessing is always compared at 1e-6 (it does not depend on the weights); probabilities at `tol`.
fn check_table(name: &str, wb: &Bundle, fx: &Bundle, tol: f64) {
    use tabicl_anywhere::prep::{ensemble, Prep};
    use tabicl_anywhere::table::{fit_classifier, Classifier, ClassifierOpts};
    let f64s = |n: &str| -> Vec<f64> { f64s(fx, n) };
    let (x, y) = (f64s("x_train"), fx.i32s("y_train").unwrap());
    let (n, h) = (y.len(), x.len() / y.len());
    let (y, labels) = tabicl_anywhere::table::encode_labels(&y.iter().map(|v| *v as i64).collect::<Vec<_>>());
    let classes = labels.len();
    // preprocessing, variant by variant
    let (prep, tables) = Prep::fit(&x, n, h, &[false, true], 4.0);
    let k = prep.keep.len();
    let xt = f64s("x_test");
    let nt = xt.len() / h;
    for (v, method) in ["none", "power"].iter().enumerate() {
        if !fx.has(&format!("train_{method}")) {
            continue; // a one-member ensemble uses only the first variant
        }
        let want = f64s(&format!("train_{method}"));
        let bad = tables[v].iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        println!("{name} preprocessing '{method}', training table: worst difference {bad:.2e}");
        assert!(bad < 1e-6, "{method} train");
        let want = f64s(&format!("test_{method}"));
        let mut row = vec![0.0f64; k];
        let mut bad = 0.0f64;
        for r in 0..nt {
            prep.apply(v, &xt[r * h..(r + 1) * h], &mut row);
            bad = row.iter().zip(&want[r * k..(r + 1) * k]).map(|(a, b)| (a - b).abs()).fold(bad, f64::max);
        }
        println!("{name} preprocessing '{method}', test table: worst difference {bad:.2e}");
        assert!(bad < 1e-6, "{method} test");
    }
    if fx.has("lambdas") {
        let want = f64s("lambdas");
        let bad = prep.norms[1].iter().zip(&want).map(|(c, w)| (c.lambda - w).abs()).fold(0.0, f64::max);
        println!("{name} power-transform exponents: worst difference {bad:.2e}");
    }
    // ensemble members: same set as the reference (its listing order depends on Python's string hashing)
    let n_members = (0..).take_while(|i| fx.has(&format!("member{i}_method"))).count();
    let ours = ensemble(k, classes, 2, n_members, 42);
    for i in 0..n_members {
        let norm = (fx.bytes(&format!("member{i}_method")).unwrap() == b"power") as usize;
        let feats: Vec<usize> = fx.i32s(&format!("member{i}_features")).unwrap().iter().map(|v| *v as usize).collect();
        let cls: Vec<usize> = fx.i32s(&format!("member{i}_classes")).unwrap().iter().map(|v| *v as usize).collect();
        assert!(ours.iter().any(|m| m.norm == norm && m.features == feats && m.classes == cls), "reference member {i} not among ours");
    }
    println!("{name}: {classes} classes; all {n_members} ensemble members match the reference's");
    // end to end
    let wm = Model::load(wb, Load::Fit).unwrap();
    let out = scratch(name);
    fit_classifier(&wm, &x, &y, n, h, &out, &ClassifierOpts { n_estimators: n_members, ..Default::default() }).unwrap();
    let cb = Bundle::open(&out).unwrap();
    let clf = Classifier::open(&cb).unwrap();
    let st = Stats::default();
    let proba = clf.predict_proba(&wm, &mut IclWeights::Resident, &xt, nt, &PredictOpts::default(), &st).unwrap();
    let want = f64s("proba");
    let bad = proba.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    println!("{name} class probabilities: worst difference {bad:.2e}\n{st}");
    assert!(bad < tol, "{name}: probabilities differ by {bad:.2e}");
    let _ = std::fs::remove_file(out);
}

fn check_table_fixture(name: &str) {
    let (Some(wb), Some(fx)) = (fixture("classifier-v2"), fixture(name)) else {
        eprintln!("fixtures missing, skipping {name}");
        return;
    };
    check_table(name, &wb, &fx, 2e-4);
}

#[test]
fn table_a() {
    check_table_fixture("table-a");
}

#[test]
fn table_b() {
    check_table_fixture("table-b");
}

#[test]
fn tiny_table() {
    check_table("tiny-table", &tiny_weights("clf"), &data("tiny-table"), 1e-5);
}

#[test]
fn tiny_one_member() {
    check_table("tiny-one", &tiny_weights("clf"), &data("tiny-one"), 1e-5);
}

#[test]
fn tiny_many_classes() {
    check_table("tiny-many", &tiny_weights("clf"), &data("tiny-many"), 1e-5);
}

/// The real model at 16-bit precision against the reference's full-precision answers. Narrowing every large
/// weight to 16 bits moved a probability by at most 7e-4 on these tables (2e-5 on table-b), so 5e-3 here:
/// room for other processors' rounding, still tight enough that any real mistake (which moves probabilities by
/// tenths) fails.
const HALF: f64 = 5e-3;

#[test]
fn docs_half_table_b() {
    let Some(wb) = docs_weights("classifier") else { return eprintln!("docs/weights missing, skipping") };
    check_table("table-b (16-bit weights)", &wb, &data("table-b"), HALF);
}

#[test]
fn docs_half_many_classes() {
    let Some(wb) = docs_weights("classifier") else { return eprintln!("docs/weights missing, skipping") };
    check_table("many-a (16-bit weights)", &wb, &data("many-a"), HALF);
}

/// The regression model end to end against the reference's `TabICLRegressor`. `tol` is a fraction of the
/// targets' spread (their deviation about 50, the value the generator centres them on).
fn check_regression(name: &str, wb: &Bundle, fx: &Bundle, members: usize, tol: f64) {
    use tabicl_anywhere::table::{fit_regressor, ClassifierOpts, Regressor};
    let f64s = |n: &str| -> Vec<f64> { f64s(fx, n) };
    let (x, y, xt, alphas) = (f64s("x_train"), f64s("y_train"), f64s("x_test"), f64s("alphas"));
    let (n, h) = (y.len(), x.len() / y.len());
    let nt = xt.len() / h;
    let wm = Model::load(wb, Load::Fit).unwrap();
    let out = scratch(name);
    fit_regressor(&wm, &x, &y, n, h, &out, &ClassifierOpts { n_estimators: members, ..Default::default() }).unwrap();
    let cb = Bundle::open(&out).unwrap();
    let reg = Regressor::open(&cb).unwrap();
    let got = reg.predict(&wm, &mut IclWeights::Resident, &xt, nt, &alphas, &PredictOpts::default(), &Stats::default()).unwrap();
    let spread = (f64s("y_test").iter().map(|v| (v - 50.0) * (v - 50.0)).sum::<f64>() / nt as f64).sqrt();
    for (kind, ours) in [("mean", &got.mean), ("variance", &got.variance), ("median", &got.median), ("quantiles", &got.quantiles)] {
        let want = f64s(kind);
        let bad = ours.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        println!("{name} {kind}: worst difference {bad:.2e} (targets spread about {spread:.1})");
        // the variance is not in the targets' units; it is compared only at full precision
        if kind != "variance" || tol < 1e-3 {
            assert!(bad < tol * spread, "{name} {kind}: {bad:.2e}");
        }
    }
    let _ = std::fs::remove_file(out);
}

fn check_regression_fixture(name: &str, members: usize) {
    let (Some(wb), Some(fx)) = (fixture("regressor-v2"), fixture(name)) else {
        eprintln!("fixtures missing, skipping {name}");
        return;
    };
    check_regression(name, &wb, &fx, members, 2e-4);
}

#[test]
fn regression_a() {
    check_regression_fixture("reg-a", 4);
}

#[test]
fn regression_b() {
    check_regression_fixture("reg-b", 1);
}

#[test]
fn tiny_regression_a() {
    check_regression("tiny-reg-a", &tiny_weights("reg"), &data("tiny-reg-a"), 4, 1e-5);
}

#[test]
fn tiny_regression_b() {
    check_regression("tiny-reg-b", &tiny_weights("reg"), &data("tiny-reg-b"), 1, 1e-5);
}

/// 16-bit weights moved a regression answer by at most 0.35% of the targets' spread (an extreme quantile), so
/// 1% here.
#[test]
fn docs_half_regression() {
    let Some(wb) = docs_weights("regressor") else { return eprintln!("docs/weights missing, skipping") };
    check_regression("reg-a (16-bit weights)", &wb, &data("reg-a"), 4, 0.01);
    check_regression("reg-b (16-bit weights)", &wb, &data("reg-b"), 1, 0.01);
}

#[test]
fn many_classes_a() {
    check_table_fixture("many-a");
}

#[test]
fn many_classes_b() {
    check_table_fixture("many-b");
}
