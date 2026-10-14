//! The whole fit-and-predict pipeline on the tiny committed model (tests/data), checking that every way of
//! storing, reading and running a fitted model gives the answers it should: the same numbers where the arithmetic
//! is the same, close numbers (with the bound stated) where it is deliberately narrower, and clear errors for bad
//! input. Runs in a second or two, everywhere.

mod common;
use browsertabicl::crypt::{new_key, seal_file, Sealed};
use browsertabicl::fit::Kv;
use browsertabicl::model::{Load, Model};
use browsertabicl::predict::{IclWeights, PredictOpts, Stats};
use browsertabicl::store::{Bundle, MemStore};
use browsertabicl::table::*;
use browsertabicl::warm::Warm;
use common::*;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;

fn fit_mem(m: &Model, t: &Table, o: &ClassifierOpts, extra: &[(&str, &[u8])]) -> Vec<u8> {
    let out = MemStore::new();
    fit_classifier_on(m, &t.x, &t.y, t.n, t.h, Box::new(out.clone()), o, extra).unwrap();
    out.bytes()
}

fn proba(m: &Model, w: &mut IclWeights, cb: &Bundle, t: &Table, po: &PredictOpts) -> Vec<f64> {
    Classifier::open(cb).unwrap().predict_proba(m, w, &t.xt, t.nt, po, &Stats::default()).unwrap()
}

fn setup() -> (Model, Table, ClassifierOpts) {
    let wb = tiny_weights("clf");
    let t = table(&data("tiny-table"));
    let o = ClassifierOpts { n_estimators: 4, ..Default::default() };
    (model(&wb), t, o)
}

#[test]
fn file_and_memory_give_identical_bundles() {
    let (m, t, o) = setup();
    let mem = fit_mem(&m, &t, &o, &[("note", b"hello")]);
    let path = scratch("file");
    fit_classifier_with(&m, &t.x, &t.y, t.n, t.h, &path, &o, &[("note", b"hello")]).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), mem, "a file and a MemStore must receive the same bytes");
    let fb = Bundle::open(&path).unwrap();
    assert_eq!(fb.bytes("note").unwrap(), b"hello");
    let _ = std::fs::remove_file(path);
}

#[test]
fn low_memory_fitting_writes_the_same_cache() {
    let (m, t, mut o) = setup();
    let normal = fit_mem(&m, &t, &o, &[]);
    o.fit.mem_budget = 1; // every intermediate goes to a scratch file
    o.fit.scratch = Some(std::env::temp_dir());
    assert_eq!(fit_mem(&m, &t, &o, &[]), normal, "spilling intermediates to disk must not change a single byte");
}

#[test]
fn chunk_size_does_not_change_answers() {
    let (m, t, mut o) = setup();
    let a = bytes_bundle(fit_mem(&m, &t, &o, &[]));
    o.fit.chunk = 7; // many small chunks, the last one partial
    let b = bytes_bundle(fit_mem(&m, &t, &o, &[]));
    let (pa, pb) = (proba(&m, &mut IclWeights::Resident, &a, &t, &PredictOpts::default()), proba(&m, &mut IclWeights::Resident, &b, &t, &PredictOpts::default()));
    assert!(worst(&pa, &pb) < 1e-6, "{}", worst(&pa, &pb));
}

#[test]
fn every_way_of_holding_weights_agrees() {
    let (m, t, o) = setup();
    let cb = bytes_bundle(fit_mem(&m, &t, &o, &[]));
    let want = proba(&m, &mut IclWeights::Resident, &cb, &t, &PredictOpts::default());
    let wb = tiny_weights("clf");
    let resident = Model::load(&wb, Load::Predict).unwrap();
    assert_eq!(proba(&resident, &mut IclWeights::Resident, &cb, &t, &PredictOpts::default()), want);
    let streamed = Model::load(&wb, Load::PredictStreamed).unwrap();
    assert!(streamed.icl.is_empty());
    let mut w = IclWeights::Streamed { bundle: tiny_weights("clf"), buf: Vec::new() };
    assert_eq!(proba(&streamed, &mut w, &cb, &t, &PredictOpts::default()), want);
}

#[test]
fn streamed_weights_without_layer_norm_biases() {
    // some checkpoints train their layer norms without a bias: streamed weights must then use zeros, exactly as
    // the resident ones do
    let src = tiny_weights("clf");
    let keep: Vec<_> = src.entries.iter().filter(|e| !(e.name.starts_with("icl_predictor.tf_icl") && e.name.ends_with("norm1.bias"))).cloned().collect();
    let ents: Vec<(&str, &str, Vec<usize>)> = keep.iter().map(|e| (e.name.as_str(), e.dtype.as_str(), e.dims.clone())).collect();
    let out = MemStore::new();
    let w = browsertabicl::store::BundleWriter::create_on(Box::new(out.clone()), &ents).unwrap();
    keep.iter().for_each(|e| w.write_bytes(&e.name, &src.bytes(&e.name).unwrap()).unwrap());
    drop(w);
    let bytes = out.bytes();
    let wb = bytes_bundle(bytes.clone());
    let m = model(&wb);
    let t = table(&data("tiny-table"));
    let cb = bytes_bundle(fit_mem(&m, &t, &ClassifierOpts { n_estimators: 2, ..Default::default() }, &[]));
    let want = proba(&m, &mut IclWeights::Resident, &cb, &t, &PredictOpts::default());
    let sm = Model::load(&wb, Load::PredictStreamed).unwrap();
    let mut sw = IclWeights::Streamed { bundle: bytes_bundle(bytes), buf: Vec::new() };
    assert_eq!(proba(&sm, &mut sw, &cb, &t, &PredictOpts::default()), want);
}

#[test]
fn prediction_settings_do_not_change_answers() {
    let (m, t, o) = setup();
    let cb = bytes_bundle(fit_mem(&m, &t, &o, &[]));
    let want = proba(&m, &mut IclWeights::Resident, &cb, &t, &PredictOpts::default());
    // one row per pass, tiny reads: same arithmetic, same numbers
    let po = PredictOpts { rows_per_pass: 1, io_bytes: 1, ..Default::default() };
    assert_eq!(proba(&m, &mut IclWeights::Resident, &cb, &t, &po), want);
    // several threads over the keys: the partial sums are combined in another order, so allow for rounding
    let po = PredictOpts { key_threads: 4, io_bytes: 64, ..Default::default() };
    let got = proba(&m, &mut IclWeights::Resident, &cb, &t, &po);
    assert!(worst(&got, &want) < 1e-6, "{}", worst(&got, &want));
    let st = Stats::default();
    Classifier::open(&cb).unwrap().predict_proba(&m, &mut IclWeights::Resident, &t.xt, t.nt, &PredictOpts::default(), &st).unwrap();
    assert_eq!(st.rows.load(Relaxed) as usize, t.nt * o.n_estimators);
    assert!(st.cache_bytes.load(Relaxed) > 0);
    assert!(st.to_string().contains("rows in"));
}

#[test]
fn rows_are_independent() {
    // a row's answer must not depend on which other rows are predicted with it
    let (m, t, o) = setup();
    let cb = bytes_bundle(fit_mem(&m, &t, &o, &[]));
    let clf = Classifier::open(&cb).unwrap();
    let all = clf.predict_proba(&m, &mut IclWeights::Resident, &t.xt, t.nt, &PredictOpts::default(), &Stats::default()).unwrap();
    for r in [0, t.nt / 2, t.nt - 1] {
        let one = clf.predict_proba(&m, &mut IclWeights::Resident, &t.xt[r * t.h..(r + 1) * t.h], 1, &PredictOpts::default(), &Stats::default()).unwrap();
        assert!(worst(&one, &all[r * clf.classes..(r + 1) * clf.classes]) < 1e-6);
    }
    for row in all.chunks_exact(clf.classes) {
        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9 && row.iter().all(|p| *p > 0.0 && *p < 1.0));
    }
}

#[test]
fn integer_caches_stay_close() {
    let (m, t, mut o) = setup();
    let want = proba(&m, &mut IclWeights::Resident, &bytes_bundle(fit_mem(&m, &t, &o, &[])), &t, &PredictOpts::default());
    let full = fit_mem(&m, &t, &o, &[]).len();
    // the bounds are what the README states for the real model (raw scores move by 1e-4 with 16 bits, 1e-2 with 8);
    // the tiny model measured 3e-8 and 3e-6
    for (kv, bound) in [(Kv::I16, 1e-4), (Kv::I8, 1e-2)] {
        o.fit.kv = kv;
        let bytes = fit_mem(&m, &t, &o, &[]);
        assert!(bytes.len() < full, "{kv:?} must be smaller");
        let got = proba(&m, &mut IclWeights::Resident, &bytes_bundle(bytes), &t, &PredictOpts::default());
        let bad = worst(&got, &want);
        println!("{kv:?}: worst probability difference {bad:.2e}");
        assert!(bad < bound, "{kv:?}: {bad:.2e}");
        let po = PredictOpts { key_threads: 3, rows_per_pass: 5, ..Default::default() };
        let threaded = proba(&m, &mut IclWeights::Resident, &bytes_bundle(fit_mem(&m, &t, &o, &[])), &t, &po);
        assert!(worst(&threaded, &got) < 1e-6);
    }
}

#[test]
fn sealed_bundle_predicts_identically_and_refuses_a_wrong_key() {
    let (m, t, o) = setup();
    let plain = scratch("plain");
    fit_classifier(&m, &t.x, &t.y, t.n, t.h, &plain, &o).unwrap();
    let sealed = scratch("sealed");
    let key = new_key().unwrap();
    seal_file(&plain, &sealed, &key).unwrap();
    let want = proba(&m, &mut IclWeights::Resident, &Bundle::open(&plain).unwrap(), &t, &PredictOpts::default());
    let sb = Bundle::from_source(Box::new(Sealed::open(std::fs::File::open(&sealed).unwrap(), &key).unwrap())).unwrap();
    assert_eq!(proba(&m, &mut IclWeights::Resident, &sb, &t, &PredictOpts::default()), want);
    let mut wrong = key;
    wrong[31] ^= 0x80;
    let opened = Sealed::open(std::fs::File::open(&sealed).unwrap(), &wrong).unwrap();
    assert!(Bundle::from_source(Box::new(opened)).is_err(), "a wrong key must not read");
    // the sealed file must not contain the plain bytes (spot check: the index text)
    let raw = std::fs::read(&sealed).unwrap();
    assert!(!raw.windows(10).any(|w| w == b"table/meta"));
    let _ = (std::fs::remove_file(plain), std::fs::remove_file(sealed));
}

#[test]
fn warm_tier_gives_identical_answers_and_keeps_tenants_apart() {
    let (m, t, o) = setup();
    let cb = bytes_bundle(fit_mem(&m, &t, &o, &[]));
    let want = proba(&m, &mut IclWeights::Resident, &cb, &t, &PredictOpts::default());
    let warm = Warm::new(64 << 20, 32 << 20, true);
    let po = PredictOpts { warm: Some(warm.handle("alice", "model")), ..Default::default() };
    assert_eq!(proba(&m, &mut IclWeights::Resident, &cb, &t, &po), want);
    let misses = warm.misses.load(Relaxed);
    assert!(misses > 0 && warm.held("alice").1 > 0);
    assert_eq!(proba(&m, &mut IclWeights::Resident, &cb, &t, &po), want);
    assert_eq!(warm.misses.load(Relaxed), misses, "the second prediction must be served from the tier");
    assert!(warm.hits.load(Relaxed) > 0);
    // another tenant with the same model name sees none of it
    let po_b = PredictOpts { warm: Some(warm.handle("bob", "model")), ..Default::default() };
    assert_eq!(proba(&m, &mut IclWeights::Resident, &cb, &t, &po_b), want);
    assert!(warm.misses.load(Relaxed) > misses);
    warm.evict("alice", Some("model"));
    assert_eq!(warm.held("alice").1, 0);
    assert!(warm.held("bob").1 > 0);
    // a tier too small for anything still answers correctly
    let tiny = Warm::new(16, 16, false);
    let po = PredictOpts { warm: Some(tiny.handle("t", "m")), ..Default::default() };
    assert_eq!(proba(&m, &mut IclWeights::Resident, &cb, &t, &po), want);
    assert_eq!(tiny.held("t"), (0, 0));
}

#[test]
fn half_size_weights_stay_close() {
    let (m, t, o) = setup();
    let want = proba(&m, &mut IclWeights::Resident, &bytes_bundle(fit_mem(&m, &t, &o, &[])), &t, &PredictOpts::default());
    let ckpt = std::fs::read(root().join("tests/data/tiny-clf.ckpt")).unwrap();
    let half = bytes_bundle(browsertabicl::ckpt::bundle_bytes(ckpt, true).unwrap());
    let hm = model(&half);
    let got = proba(&hm, &mut IclWeights::Resident, &bytes_bundle(fit_mem(&hm, &t, &o, &[])), &t, &PredictOpts::default());
    // only tensors of 4096 numbers or more are narrowed, and the tiny model has none: so nothing may change at
    // all (the narrowing itself is checked in tests/formats.rs)
    assert_eq!(got, want);
}

#[test]
fn members_fitted_apart_add_up_exactly() {
    let (m, t, o) = setup();
    let whole = bytes_bundle(fit_mem(&m, &t, &o, &[]));
    let clf = Classifier::open(&whole).unwrap();
    let want = clf.predict_proba(&m, &mut IclWeights::Resident, &t.xt, t.nt, &PredictOpts::default(), &Stats::default()).unwrap();
    let mut sums = vec![0.0f64; want.len()];
    for i in 0..o.n_estimators {
        let one = bytes_bundle(fit_mem(&m, &t, &ClassifierOpts { only: Some(vec![i]), ..o.clone() }, &[]));
        let c1 = Classifier::open(&one).unwrap();
        assert_eq!(c1.members, vec![clf.members[i].clone()]);
        let part = c1.logit_sums(&m, &mut IclWeights::Resident, &t.xt, t.nt, &PredictOpts::default(), &Stats::default()).unwrap();
        sums.iter_mut().zip(&part).for_each(|(a, b)| *a += b);
    }
    finish_proba(&mut sums, clf.classes, o.n_estimators, clf.temperature);
    assert_eq!(sums, want);
    assert_eq!(member_count(&t.x, t.n, t.h, clf.classes, &o), o.n_estimators);
}

#[test]
fn fitting_rejects_bad_input() {
    let (m, t, o) = setup();
    let fit = |x: &[f64], y: &[usize], n: usize, h: usize, o: &ClassifierOpts| fit_classifier_on(&m, x, y, n, h, Box::new(MemStore::new()), o, &[]);
    let err = |r: std::io::Result<()>| r.unwrap_err().to_string();
    assert!(err(fit(&t.x, &t.y, t.n + 1, t.h, &o)).contains("table is"));
    assert!(err(fit(&t.x, &t.y[1..], t.n, t.h, &o)).contains("targets"));
    assert!(fit(&[], &[], 0, t.h, &o).is_err());
    let mut gap = t.y.clone();
    gap.iter_mut().for_each(|c| *c = if *c == 1 { 2 } else { *c }); // class 1 never occurs
    if gap.contains(&2) {
        assert!(err(fit(&t.x, &gap, t.n, t.h, &o)).contains("never occurs"));
    }
    assert!(err(fit(&t.x, &t.y, t.n, t.h, &ClassifierOpts { only: Some(vec![99]), ..o.clone() })).contains("member 99"));
    // classifier weights cannot fit a regression, nor regression weights a classifier
    let r = fit_regressor_on(&m, &t.x, &vec![1.0; t.n], t.n, t.h, Box::new(MemStore::new()), &o, &[]);
    assert!(err(r).contains("classifier's"));
    let rm = model(&tiny_weights("reg"));
    assert!(err(fit_classifier_on(&rm, &t.x, &t.y, t.n, t.h, Box::new(MemStore::new()), &o, &[])).contains("regression model's"));
    let cb = bytes_bundle(fit_mem(&m, &t, &o, &[]));
    let clf = Classifier::open(&cb).unwrap();
    assert!(clf.predict_proba(&m, &mut IclWeights::Resident, &t.xt[1..], t.nt, &PredictOpts::default(), &Stats::default()).is_err());
}

#[test]
fn opening_the_wrong_or_a_damaged_bundle_is_an_error() {
    let (m, t, o) = setup();
    let clf_bytes = fit_mem(&m, &t, &o, &[]);
    let rm = model(&tiny_weights("reg"));
    let rt = table(&data("tiny-reg-b"));
    let out = MemStore::new();
    fit_regressor_on(&rm, &rt.x, &rt.yf, rt.n, rt.h, Box::new(out.clone()), &ClassifierOpts { n_estimators: 1, ..Default::default() }, &[]).unwrap();
    let reg_bytes = out.bytes();
    assert!(Classifier::open(&bytes_bundle(reg_bytes.clone())).is_err());
    assert!(Regressor::open(&bytes_bundle(clf_bytes.clone())).is_err());
    let reg_b = bytes_bundle(reg_bytes);
    let reg = Regressor::open(&reg_b).unwrap();
    assert!(reg.predict(&rm, &mut IclWeights::Resident, &rt.xt[1..], rt.nt, &[0.5], &PredictOpts::default(), &Stats::default()).is_err());
    // damage each stored table field in turn: an error every time, never a panic
    let cb = bytes_bundle(clf_bytes.clone());
    for name in ["table/meta", "table/keep", "table/members", "table/power"] {
        let e = cb.entry(name).unwrap().clone();
        for value in [-1i32, 1 << 20] {
            let mut bytes = clf_bytes.clone();
            bytes[e.offset as usize..e.offset as usize + 4].copy_from_slice(&value.to_le_bytes());
            let b = bytes_bundle(bytes);
            if let Ok(c) = Classifier::open(&b) {
                // a value that is still consistent (power flag) is allowed; prediction must still work
                assert_eq!(name, "table/power");
                c.predict_proba(&m, &mut IclWeights::Resident, &t.xt, t.nt, &PredictOpts::default(), &Stats::default()).unwrap();
            }
        }
    }
}

#[test]
fn regression_pipeline() {
    let rm = model(&tiny_weights("reg"));
    assert_eq!(rm.max_classes, 0);
    let t = table(&data("tiny-reg-a"));
    let o = ClassifierOpts { n_estimators: 2, ..Default::default() };
    let alphas = [0.0001, 0.1, 0.5, 0.9, 0.9999];
    let run = |o: &ClassifierOpts, po: &PredictOpts| {
        let out = MemStore::new();
        fit_regressor_on(&rm, &t.x, &t.yf, t.n, t.h, Box::new(out.clone()), o, &[("x", b"1")]).unwrap();
        let b = bytes_bundle(out.bytes());
        Regressor::open(&b).unwrap().predict(&rm, &mut IclWeights::Resident, &t.xt, t.nt, &alphas, po, &Stats::default()).unwrap()
    };
    let r = run(&o, &PredictOpts::default());
    for i in 0..t.nt {
        let q = &r.quantiles[i * alphas.len()..(i + 1) * alphas.len()];
        assert!(q.windows(2).all(|w| w[0] <= w[1]), "quantiles must rise: {q:?}");
        assert!((q[2] - r.median[i]).abs() < 1e-9, "the 0.5 quantile is the median");
        assert!(r.mean[i].is_finite() && r.variance[i].is_finite());
    }
    // the file form and the 16-bit cache
    let path = scratch("reg");
    fit_regressor(&rm, &t.x, &t.yf, t.n, t.h, &path, &o).unwrap();
    let fb = Bundle::open(&path).unwrap();
    let rf = Regressor::open(&fb).unwrap().predict(&rm, &mut IclWeights::Resident, &t.xt, t.nt, &alphas, &PredictOpts::default(), &Stats::default()).unwrap();
    assert_eq!(rf.mean, r.mean);
    let _ = std::fs::remove_file(path);
    let mut o16 = o.clone();
    o16.fit.kv = Kv::I16;
    let r16 = run(&o16, &PredictOpts { key_threads: 2, ..Default::default() });
    let spread = 4.0; // the generator's targets have a spread of about 4
    assert!(worst(&r16.mean, &r.mean) < 1e-3 * spread);
    // a constant target is allowed (its scale is taken as 1) and predicts that constant
    let out = MemStore::new();
    fit_regressor_on(&rm, &t.x, &vec![7.0; t.n], t.n, t.h, Box::new(out.clone()), &o, &[]).unwrap();
    let b = bytes_bundle(out.bytes());
    let c = Regressor::open(&b).unwrap();
    assert_eq!(c.target_scale, 1.0);
    assert_eq!(c.target_mean, 7.0);
    let bad = fit_regressor_on(&rm, &t.x, &t.yf[1..], t.n, t.h, Box::new(MemStore::new()), &o, &[]);
    assert!(bad.is_err());
}

#[test]
fn quantile_distribution() {
    let q: Vec<f32> = (1..=99).rev().map(|i| i as f32 * 0.1).collect(); // given out of order on purpose
    let d = QuantileDist::new(q);
    assert!(d.q.windows(2).all(|w| w[0] <= w[1]), "sorted");
    assert!((d.mean() - 5.0).abs() < 1e-4);
    assert!(d.variance() > 0.0);
    assert!((d.icdf(0.5) - 5.0).abs() < 1e-4, "{}", d.icdf(0.5));
    let mut last = f32::MIN;
    for a in [1e-7f32, 1e-4, 0.005, 0.01, 0.2, 0.5, 0.8, 0.99, 0.995, 0.9999, 1.0 - 1e-7] {
        let v = d.icdf(a);
        assert!(v >= last, "icdf must rise: {a} -> {v}");
        last = v;
    }
    assert_eq!(d.icdf(0.99), d.q[98], "exactly at the last level");
    let flat = QuantileDist::new(vec![3.0; 9]);
    assert_eq!(flat.icdf(0.5), 3.0);
}

#[test]
fn labels_and_probabilities() {
    let (y, labels) = encode_labels(&[30, -2, 30, 7, -2]);
    assert_eq!(labels, [-2, 7, 30]);
    assert_eq!(y, [2, 0, 2, 1, 0]);
    let mut s = vec![1000.0, 1000.0, 0.0, -5.0, 0.0, 5.0];
    finish_proba(&mut s, 3, 1, 1.0);
    assert!((s[0] - 0.5).abs() < 1e-12 && s[2] < 1e-300, "huge scores must not overflow");
    assert!((s[3..].iter().sum::<f64>() - 1.0).abs() < 1e-12 && s[5] > s[4] && s[4] > s[3]);
    // a higher temperature flattens
    let (mut a, mut b) = (vec![0.0, 2.0], vec![0.0, 2.0]);
    finish_proba(&mut a, 2, 1, 0.5);
    finish_proba(&mut b, 2, 1, 2.0);
    assert!(a[1] > b[1]);
}

#[test]
fn shared_store_is_usable_from_threads() {
    // MemStore is what a browser fits into; two fits into two stores at once must not interfere
    let (m, t, o) = setup();
    let m = Arc::new(m);
    let (a, b) = std::thread::scope(|s| {
        let h1 = s.spawn(|| fit_mem(&m, &t, &o, &[]));
        let h2 = s.spawn(|| fit_mem(&m, &t, &o, &[]));
        (h1.join().unwrap(), h2.join().unwrap())
    });
    assert_eq!(a, b);
}
