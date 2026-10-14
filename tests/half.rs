//! Half-size (16-bit) weights and one-member fitting, measured against the reference's numbers and against the
//! full-size weights. Skipped (with a note) if the fixtures have not been generated.
//! `cargo test --release --test half -- --nocapture` prints the measurements.

use browsertabicl::model::{Load, Model};
use browsertabicl::predict::{IclWeights, PredictOpts, Stats};
use browsertabicl::store::{Bundle, MemStore};
use browsertabicl::table::{encode_labels, finish_proba, fit_classifier_on, fit_regressor_on, Classifier, ClassifierOpts, Regressor};
use std::sync::Arc;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures");

fn weights(name: &str, half: bool) -> Option<Model> {
    let bytes = std::fs::read(format!("{DIR}/{name}.tirs")).ok()?;
    let full = bytes.len();
    let bytes = browsertabicl::ckpt::bundle_bytes(bytes, half).unwrap();
    if half {
        println!("{name}: {:.1} MB at full size, {:.1} MB with 16-bit floats", full as f64 / 1e6, bytes.len() as f64 / 1e6);
    }
    Some(Model::load(&Bundle::from_source(Box::new(Arc::new(bytes))).unwrap(), Load::Fit).unwrap())
}

fn f64s(fx: &Bundle, n: &str) -> Vec<f64> {
    bytemuck::pod_collect_to_vec(&fx.bytes(n).unwrap())
}

fn worst(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn fitted(f: impl FnOnce(Box<dyn browsertabicl::store::Storage>)) -> Bundle {
    let out = MemStore::new();
    f(Box::new(out.clone()));
    Bundle::from_source(Box::new(Arc::new(out.bytes()))).unwrap()
}

fn classify(m: &Model, fx: &Bundle, kv: browsertabicl::fit::Kv) -> (Vec<f64>, usize) {
    let (x, y, xt) = (f64s(fx, "x_train"), fx.i32s("y_train").unwrap(), f64s(fx, "x_test"));
    let (n, h) = (y.len(), x.len() / y.len());
    let (y, _) = encode_labels(&y.iter().map(|v| *v as i64).collect::<Vec<_>>());
    let members = (0..).take_while(|i| fx.has(&format!("member{i}_method"))).count();
    let mut o = ClassifierOpts { n_estimators: members, ..Default::default() };
    o.fit.kv = kv;
    let cb = fitted(|s| fit_classifier_on(m, &x, &y, n, h, s, &o, &[]).unwrap());
    let clf = Classifier::open(&cb).unwrap();
    (clf.predict_proba(m, &mut IclWeights::Resident, &xt, xt.len() / h, &PredictOpts::default(), &Stats::default()).unwrap(), clf.classes)
}

#[test]
fn half_weights_classifier() {
    use browsertabicl::fit::Kv;
    let (Some(full), Some(half)) = (weights("classifier-v2", false), weights("classifier-v2", true)) else {
        eprintln!("fixtures missing, skipping");
        return;
    };
    for name in ["table-a", "table-b", "many-a", "many-b"] {
        let Ok(fx) = Bundle::open(format!("{DIR}/{name}.tirs")) else { continue };
        let want = f64s(&fx, "proba");
        let (pf, c) = classify(&full, &fx, Kv::F32);
        let (ph, _) = classify(&half, &fx, Kv::F32);
        let (ph16, _) = classify(&half, &fx, Kv::I16);
        let arg = |p: &[f64]| -> Vec<usize> { p.chunks_exact(c).map(|r| (0..c).max_by(|a, b| r[*a].total_cmp(&r[*b])).unwrap()).collect() };
        let same = |p: &[f64]| arg(p).iter().zip(arg(&want)).filter(|(a, b)| *a == b).count();
        let rows = want.len() / c;
        println!(
            "{name} ({c} classes, {rows} test rows): worst probability difference from the reference: full weights {:.2e}; half weights {:.2e}; half weights + 16-bit fitted model {:.2e}. Half against full: {:.2e}. Same predicted class as the reference: {}/{rows} half, {}/{rows} half + 16-bit model",
            worst(&pf, &want), worst(&ph, &want), worst(&ph16, &want), worst(&ph, &pf), same(&ph), same(&ph16)
        );
        assert!(worst(&pf, &want) < 2e-4);
        assert!(worst(&ph, &want) < 2e-2, "half weights moved a probability by more than 0.02");
        assert!(worst(&ph16, &want) < 2e-2);
    }
}

#[test]
fn half_weights_regressor() {
    let (Some(full), Some(half)) = (weights("regressor-v2", false), weights("regressor-v2", true)) else {
        eprintln!("fixtures missing, skipping");
        return;
    };
    for (name, members) in [("reg-a", 4), ("reg-b", 1)] {
        let Ok(fx) = Bundle::open(format!("{DIR}/{name}.tirs")) else { continue };
        let (x, y, xt, alphas) = (f64s(&fx, "x_train"), f64s(&fx, "y_train"), f64s(&fx, "x_test"), f64s(&fx, "alphas"));
        let (n, h) = (y.len(), x.len() / y.len());
        let run = |m: &Model| {
            let cb = fitted(|s| fit_regressor_on(m, &x, &y, n, h, s, &ClassifierOpts { n_estimators: members, ..Default::default() }, &[]).unwrap());
            Regressor::open(&cb).unwrap().predict(m, &mut IclWeights::Resident, &xt, xt.len() / h, &alphas, &PredictOpts::default(), &Stats::default()).unwrap()
        };
        let (rf, rh) = (run(&full), run(&half));
        let ym = y.iter().sum::<f64>() / n as f64;
        let spread = (y.iter().map(|v| (v - ym) * (v - ym)).sum::<f64>() / n as f64).sqrt();
        for (kind, f, hv) in [("mean", &rf.mean, &rh.mean), ("median", &rf.median, &rh.median), ("quantiles", &rf.quantiles, &rh.quantiles)] {
            let want = f64s(&fx, kind);
            println!(
                "{name} {kind}: worst difference from the reference: full weights {:.2e}; half weights {:.2e} (= {:.3}% of the targets' spread, {spread:.1})",
                worst(f, &want),
                worst(hv, &want),
                100.0 * worst(hv, &want) / spread
            );
            assert!(worst(hv, &want) < 0.02 * spread, "{kind}");
        }
    }
}

/// Fitting and predicting one member at a time gives exactly what the whole ensemble gives in one go.
#[test]
fn members_one_at_a_time() {
    let Some(m) = weights("classifier-v2", false) else { return };
    let Ok(fx) = Bundle::open(format!("{DIR}/table-b.tirs")) else { return };
    let (x, y, xt) = (f64s(&fx, "x_train"), fx.i32s("y_train").unwrap(), f64s(&fx, "x_test"));
    let (n, h) = (y.len(), x.len() / y.len());
    let nt = xt.len() / h;
    let (y, _) = encode_labels(&y.iter().map(|v| *v as i64).collect::<Vec<_>>());
    let o = ClassifierOpts { n_estimators: 3, ..Default::default() };
    let whole = fitted(|s| fit_classifier_on(&m, &x, &y, n, h, s, &o, &[]).unwrap());
    let clf = Classifier::open(&whole).unwrap();
    let want = clf.predict_proba(&m, &mut IclWeights::Resident, &xt, nt, &PredictOpts::default(), &Stats::default()).unwrap();
    let mut sums = vec![0.0f64; want.len()];
    for i in 0..3 {
        let one = fitted(|s| fit_classifier_on(&m, &x, &y, n, h, s, &ClassifierOpts { only: Some(vec![i]), ..o.clone() }, &[]).unwrap());
        let c1 = Classifier::open(&one).unwrap();
        assert_eq!(c1.members.len(), 1);
        assert_eq!(c1.members[0], clf.members[i]);
        let part = c1.logit_sums(&m, &mut IclWeights::Resident, &xt, nt, &PredictOpts::default(), &Stats::default()).unwrap();
        sums.iter_mut().zip(&part).for_each(|(a, b)| *a += b);
    }
    finish_proba(&mut sums, clf.classes, 3, clf.temperature);
    assert_eq!(sums, want, "one member at a time must give identical probabilities");
    let o = ClassifierOpts { only: Some(vec![7]), ..o };
    assert!(fit_classifier_on(&m, &x, &y, n, h, Box::new(MemStore::new()), &o, &[]).is_err());
}
