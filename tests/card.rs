//! The graphics card against the CPU path: prediction with all three stages on the card, fitting on the card,
//! the regression model, more than 10 classes, the 16-bit cache, small and large batches, and the rules of the
//! graphics-memory tier under concurrency. Needs `--features gpu`; each test is skipped (with a note) when there
//! is no graphics adapter. Every comparison is card against CPU, so without the generated fixtures the tests use
//! the committed data instead: the 16-bit weights in docs/weights and the small tables in tests/data (the tests
//! on `table-a`, which is not committed, then skip).
#![cfg(feature = "gpu")]

use std::path::PathBuf;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, OnceLock};
use tabicl_anywhere::fit::{Accel, FitOpts, Kv};
use tabicl_anywhere::gpu::{Gpu, GpuWeights, VramTier};
use tabicl_anywhere::model::{Load, Model};
use tabicl_anywhere::predict::{CardAttend, IclWeights, PredictOpts, Stats};
use tabicl_anywhere::store::Bundle;
use tabicl_anywhere::table::{fit_classifier, fit_regressor, Classifier, ClassifierOpts, Regressor};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures");

/// One card for the whole run (opening one takes a moment), or `None` if this machine has none.
fn card() -> Option<Arc<Gpu>> {
    static CARD: OnceLock<Option<Arc<Gpu>>> = OnceLock::new();
    CARD.get_or_init(|| match Gpu::new() {
        Ok(g) => {
            println!("graphics card: {}", g.name);
            Some(Arc::new(g))
        }
        Err(e) => {
            eprintln!("no graphics card ({e}), skipping");
            None
        }
    })
    .clone()
}

/// The tests share one card; this keeps its working-memory budget (which some of them change) to one test at
/// a time, and puts it back afterwards.
fn exclusive() -> std::sync::MutexGuard<'static, ()> {
    static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    ONE.lock().unwrap_or_else(|e| e.into_inner())
}

struct Table {
    x: Vec<f64>,
    n: usize,
    h: usize,
    xt: Vec<f64>,
    nt: usize,
    y_classes: Vec<usize>,
    y_numbers: Vec<f64>,
}

/// A fixture table, its test rows repeated or cut to `test_rows`.
fn table(name: &str, test_rows: usize) -> Option<Table> {
    let committed = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");
    let fx = Bundle::open(format!("{DIR}/{name}.tirs")).or_else(|_| Bundle::open(format!("{committed}/{name}.tirs"))).ok()?;
    let f64s = |n: &str| -> Vec<f64> { bytemuck::pod_collect_to_vec(&fx.bytes(n).unwrap()) };
    let x = f64s("x_train");
    let (y_classes, y_numbers) = if fx.entry("y_train").unwrap().dtype.contains("int") {
        let y = fx.i32s("y_train").unwrap();
        (tabicl_anywhere::table::encode_labels(&y.iter().map(|v| *v as i64).collect::<Vec<_>>()).0, Vec::new())
    } else {
        (Vec::new(), f64s("y_train"))
    };
    let n = y_classes.len().max(y_numbers.len());
    let h = x.len() / n;
    let given = f64s("x_test");
    let xt: Vec<f64> = given.chunks_exact(h).cycle().take(test_rows).flatten().copied().collect();
    Some(Table { x, n, h, xt, nt: test_rows, y_classes, y_numbers })
}

/// The full-precision weights from fixtures/, or else the web page's 16-bit copy (joined from docs/weights).
fn weights(file: &str) -> Option<Model> {
    if let Ok(b) = Bundle::open(format!("{DIR}/{file}")) {
        return Model::load(&b, Load::Fit).ok();
    }
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/weights");
    let manifest = std::fs::read_to_string(dir.join(file.replace(".tirs", "-half.json"))).ok()?;
    let bytes = tabicl_anywhere::store::join_parts(&manifest, |name| std::fs::read(dir.join(name))).ok()?;
    Model::load(&Bundle::from_source(Box::new(Arc::new(bytes))).ok()?, Load::Fit).ok()
}

fn scratch(what: &str) -> PathBuf {
    std::env::temp_dir().join(format!("tabicl-anywhere-card-{what}-{}.tirs", std::process::id()))
}

fn worst(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// Load every in-context problem of every member of a fitted bundle into the tier for `tenant`.
fn preload_all(tier: &VramTier, tenant: &str, model: &str, bundle: &Bundle, gw: &GpuWeights) -> bool {
    let members = bundle.i32s("table/meta").unwrap()[3] as usize;
    let mut all = true;
    for i in 0..members {
        let prefix = format!("m{i}/");
        let nodes = tabicl_anywhere::predict::Cache::open(bundle, &prefix).unwrap().nodes.len();
        for node in 0..nodes {
            all &= tier.preload(tenant, model, bundle, gw, &prefix, node).unwrap();
        }
    }
    all
}

/// Class probabilities for `t.xt` from a fitted classifier, on the CPU (`accel` None) or through the card.
fn proba(m: &Model, bundle: &Bundle, t: &Table, accel: Option<CardAttend>, st: &Stats) -> Vec<f64> {
    let clf = Classifier::open(bundle).unwrap();
    let opts = PredictOpts { rows_per_pass: 512, accel, ..Default::default() };
    clf.predict_proba(m, &mut IclWeights::Resident, &t.xt, t.nt, &opts, st).unwrap()
}

/// Fit `name` on the CPU, then predict `test_rows` rows on the CPU and with all stages on the card.
fn classifier_on_card(name: &str, test_rows: usize, members: usize, kv: Kv) {
    let (Some(gpu), Some(m), Some(t)) = (card(), weights("classifier-v2.tirs"), table(name, test_rows)) else {
        return eprintln!("fixtures or card missing, skipping {name}");
    };
    let _one = exclusive();
    let path = scratch(&format!("{name}-{test_rows}"));
    let opts = ClassifierOpts { n_estimators: members, fit: FitOpts { kv, ..Default::default() }, ..Default::default() };
    fit_classifier(&m, &t.x, &t.y_classes, t.n, t.h, &path, &opts).unwrap();
    let bundle = Arc::new(Bundle::open(&path).unwrap());
    let cpu = proba(&m, &bundle, &t, None, &Stats::default());
    let gw = Arc::new(GpuWeights::new(&gpu, &m));
    let tier = VramTier::new(gpu.clone(), 4 << 30, 4 << 30);
    assert!(preload_all(&tier, "t", name, &bundle, &gw));
    let st = Stats::default();
    let on_card = proba(&m, &bundle, &t, Some(tier.handle("t", name, bundle.clone(), gw)), &st);
    let bad = worst(&cpu, &on_card);
    println!("{name}, {test_rows} rows, {members} members, {kv:?} cache: card against CPU, worst probability difference {bad:.2e}; {} layers on the card", st.card_layers.load(Relaxed));
    assert_eq!(tier.fallbacks.load(Relaxed), 0, "the card failed: {:?}", tier.last_error());
    assert!(st.card_layers.load(Relaxed) > 0, "nothing ran on the card");
    assert!(bad < 1e-3);
    assert_eq!(gpu.overdrawn(), 0, "a call borrowed more working memory than it said it would");
    let _ = std::fs::remove_file(path);
}

#[test]
fn card_predicts_like_cpu_one_row() {
    classifier_on_card("table-a", 1, 2, Kv::F32);
}

#[test]
fn card_predicts_like_cpu_many_rows() {
    classifier_on_card("table-b", 300, 2, Kv::F32);
}

#[test]
fn card_predicts_like_cpu_more_than_ten_classes() {
    classifier_on_card("many-a", 40, 1, Kv::F32);
}

#[test]
fn card_predicts_like_cpu_sixteen_bit_cache() {
    classifier_on_card("table-a", 64, 2, Kv::I16);
}

/// Fitting on the card writes a cache that predicts what the CPU's does.
fn fit_on_card_matches(name: &str, members: usize) {
    let (Some(gpu), Some(m), Some(t)) = (card(), weights("classifier-v2.tirs"), table(name, 50)) else {
        return eprintln!("fixtures or card missing, skipping {name}");
    };
    let _one = exclusive();
    let (pc, pg) = (scratch(&format!("{name}-cpu")), scratch(&format!("{name}-card")));
    let mut opts = ClassifierOpts { n_estimators: members, ..Default::default() };
    fit_classifier(&m, &t.x, &t.y_classes, t.n, t.h, &pc, &opts).unwrap();
    opts.fit.accel = Accel(Some(gpu.clone()));
    fit_classifier(&m, &t.x, &t.y_classes, t.n, t.h, &pg, &opts).unwrap();
    let (bc, bg) = (Bundle::open(&pc).unwrap(), Bundle::open(&pg).unwrap());
    let (a, b) = (proba(&m, &bc, &t, None, &Stats::default()), proba(&m, &bg, &t, None, &Stats::default()));
    let bad = worst(&a, &b);
    println!("{name}: fitted on the card against fitted on the CPU, worst probability difference {bad:.2e}");
    assert!(bad < 1e-3);
    assert_eq!(gpu.overdrawn(), 0);
    let _ = (std::fs::remove_file(pc), std::fs::remove_file(pg));
}

#[test]
fn card_fit_matches_cpu_fit() {
    fit_on_card_matches("table-b", 2);
}

#[test]
fn card_fit_matches_cpu_fit_more_than_ten_classes() {
    fit_on_card_matches("many-a", 1);
}

#[test]
fn regression_on_card() {
    let (Some(gpu), Some(m), Some(t)) = (card(), weights("regressor-v2.tirs"), table("reg-a", 200)) else {
        return eprintln!("fixtures or card missing, skipping regression");
    };
    let _one = exclusive();
    let (pc, pg) = (scratch("reg-cpu"), scratch("reg-card"));
    let mut opts = ClassifierOpts { n_estimators: 2, ..Default::default() };
    fit_regressor(&m, &t.x, &t.y_numbers, t.n, t.h, &pc, &opts).unwrap();
    opts.fit.accel = Accel(Some(gpu.clone()));
    fit_regressor(&m, &t.x, &t.y_numbers, t.n, t.h, &pg, &opts).unwrap();
    let alphas = [0.05, 0.5, 0.95];
    let predict = |b: &Bundle, accel: Option<CardAttend>, st: &Stats| {
        let r = Regressor::open(b).unwrap();
        let po = PredictOpts { rows_per_pass: 512, accel, ..Default::default() };
        r.predict(&m, &mut IclWeights::Resident, &t.xt, t.nt, &alphas, &po, st).unwrap()
    };
    let (bc, bg) = (Arc::new(Bundle::open(&pc).unwrap()), Bundle::open(&pg).unwrap());
    let cpu = predict(&bc, None, &Stats::default());
    let spread = (t.y_numbers.iter().map(|v| v * v).sum::<f64>() / t.n as f64).sqrt().max(1.0);
    // fitted on the card, predicted on the CPU
    let fitted = predict(&bg, None, &Stats::default());
    // fitted on the CPU, predicted on the card
    let gw = Arc::new(GpuWeights::new(&gpu, &m));
    let tier = VramTier::new(gpu.clone(), 4 << 30, 4 << 30);
    assert!(preload_all(&tier, "t", "reg", &bc, &gw));
    let st = Stats::default();
    let card_pred = predict(&bc, Some(tier.handle("t", "reg", bc.clone(), gw)), &st);
    assert_eq!(tier.fallbacks.load(Relaxed), 0, "the card failed: {:?}", tier.last_error());
    assert!(st.card_layers.load(Relaxed) > 0);
    for (what, other) in [("fitted on the card", &fitted), ("predicted on the card", &card_pred)] {
        let bad = worst(&cpu.mean, &other.mean).max(worst(&cpu.median, &other.median)).max(worst(&cpu.quantiles, &other.quantiles));
        println!("regression, {what}: worst difference in mean, median and quantiles {bad:.2e} (targets spread about {spread:.1})");
        assert!(bad < 1e-3 * spread, "{what}");
    }
    let _ = (std::fs::remove_file(pc), std::fs::remove_file(pg));
}

/// A table fitted once on the CPU, shared by the tier tests: (model, bundle, table, CPU probabilities).
fn fitted(name: &str, rows: usize) -> Option<(Model, Arc<Bundle>, Table, Vec<f64>, PathBuf)> {
    let (m, t) = (weights("classifier-v2.tirs")?, table(name, rows)?);
    let path = scratch(&format!("tier-{name}-{rows}"));
    fit_classifier(&m, &t.x, &t.y_classes, t.n, t.h, &path, &ClassifierOpts { n_estimators: 1, ..Default::default() }).unwrap();
    let bundle = Arc::new(Bundle::open(&path).unwrap());
    let cpu = proba(&m, &bundle, &t, None, &Stats::default());
    Some((m, bundle, t, cpu, path))
}

#[test]
fn working_memory_is_bounded_and_released() {
    let Some(gpu) = card() else { return };
    let Some((m, bundle, t, cpu, path)) = fitted("table-b", 400) else { return eprintln!("fixtures missing, skipping") };
    let _one = exclusive();
    let gw = Arc::new(GpuWeights::new(&gpu, &m));
    let tier = VramTier::new(gpu.clone(), 1 << 30, 1 << 30);
    assert!(preload_all(&tier, "t", "b", &bundle, &gw));
    gpu.release();
    // a budget that holds well under 400 rows' working set: the batch is cut into sub-batches
    let budget = 24 << 20;
    gpu.set_work_budget(budget);
    let st = Stats::default();
    let got = proba(&m, &bundle, &t, Some(tier.handle("t", "b", bundle.clone(), gw.clone())), &st);
    let (now, peak) = gpu.working();
    println!("400 rows with a {} MB working budget: at most {:.1} MB of working memory, {:.1} MB kept; worst difference {:.2e}", budget >> 20, peak as f64 / 1e6, now as f64 / 1e6, worst(&cpu, &got));
    assert!(st.card_layers.load(Relaxed) > 0 && tier.fallbacks.load(Relaxed) == 0, "{:?}", tier.last_error());
    assert!(worst(&cpu, &got) < 1e-3);
    assert!(peak <= budget, "working memory went over its budget");
    assert_eq!(gpu.overdrawn(), 0);
    // a budget too small for even one row's whole chain: the CPU answers that part (here the column and row
    // stages; the in-context stage alone still fits), nothing fails
    gpu.set_work_budget(64 << 10);
    let st = Stats::default();
    let got = proba(&m, &bundle, &t, Some(tier.handle("t", "b", bundle.clone(), gw)), &st);
    println!(
        "with a 64 KB budget: worst difference {:.2e}, {} layers on the card, {} answered by the CPU instead ({:?})",
        worst(&cpu, &got),
        st.card_layers.load(Relaxed),
        tier.fallbacks.load(Relaxed),
        tier.last_error()
    );
    assert!(worst(&cpu, &got) < 1e-3 && tier.fallbacks.load(Relaxed) > 0);
    gpu.release();
    assert_eq!(gpu.working().0, 0, "release leaves no working memory behind");
    gpu.set_work_budget(tabicl_anywhere::gpu::WORK_BUDGET);
    let _ = std::fs::remove_file(path);
}

#[test]
fn tenants_at_once_and_eviction_in_flight() {
    let Some(gpu) = card() else { return };
    let Some((m, bundle, t, cpu, path)) = fitted("table-a", 120) else { return eprintln!("fixtures missing, skipping") };
    let _one = exclusive();
    let gw = Arc::new(GpuWeights::new(&gpu, &m));
    let one = Gpu::model_bytes(&gw, &tabicl_anywhere::predict::Cache::open(&bundle, "m0/").unwrap(), 0);
    // room for two tenants' copies, each tenant allowed one
    let tier = VramTier::new(gpu.clone(), 2 * one, one);
    assert!(tier.preload("alice", "a", &bundle, &gw, "m0/", 0).unwrap());
    assert!(tier.preload("bob", "a", &bundle, &gw, "m0/", 0).unwrap());
    assert_eq!(tier.on_card("alice"), (2 * one, one));
    let stop = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|s| {
        // two tenants predicting at once, from different threads, while a third keeps evicting alice
        let runs: Vec<_> = ["alice", "bob"]
            .into_iter()
            .map(|who| {
                let (m, bundle, t, cpu, tier, gw) = (&m, &bundle, &t, &cpu, &tier, &gw);
                s.spawn(move || {
                    let mut on_card = 0;
                    for _ in 0..6 {
                        let st = Stats::default();
                        let got = proba(m, bundle, t, Some(tier.handle(who, "a", bundle.clone(), gw.clone())), &st);
                        assert!(worst(cpu, &got) < 1e-3, "{who}: wrong answer");
                        on_card += (st.card_layers.load(Relaxed) > 0) as usize;
                        let (all, mine) = tier.on_card(who);
                        assert!(all <= 2 * one && mine <= one, "{who}: over budget on the card");
                    }
                    on_card
                })
            })
            .collect();
        s.spawn(|| {
            while !stop.load(Relaxed) {
                tier.evict("alice", None);
                std::thread::sleep(std::time::Duration::from_millis(3));
            }
        });
        let counts: Vec<usize> = runs.into_iter().map(|r| r.join().unwrap()).collect();
        stop.store(true, Relaxed);
        println!("predictions answered on the card: alice {} of 6 (evicted all the while), bob {} of 6", counts[0], counts[1]);
        assert!(counts[1] > 0);
    });
    assert_eq!(tier.fallbacks.load(Relaxed), 0, "{:?}", tier.last_error());
    // background loads started by alice's predictions may still be finishing: let them, then evict everything
    std::thread::sleep(std::time::Duration::from_millis(500));
    tier.evict("alice", None);
    tier.evict("bob", None);
    assert_eq!(tier.on_card("alice"), (0, 0), "evicted models are released once no prediction uses them");
    assert_eq!(gpu.overdrawn(), 0);
    let _ = std::fs::remove_file(path);
}

#[test]
fn model_larger_than_tenant_budget_stays_on_cpu() {
    let Some(gpu) = card() else { return };
    let Some((m, bundle, t, cpu, path)) = fitted("table-a", 10) else { return eprintln!("fixtures missing, skipping") };
    let _one = exclusive();
    let gw = Arc::new(GpuWeights::new(&gpu, &m));
    let tier = VramTier::new(gpu.clone(), 1 << 30, 1 << 20);
    assert!(!tier.preload("carol", "a", &bundle, &gw, "m0/", 0).unwrap());
    for _ in 0..3 {
        let st = Stats::default();
        let got = proba(&m, &bundle, &t, Some(tier.handle("carol", "a", bundle.clone(), gw.clone())), &st);
        assert!(worst(&cpu, &got) < 1e-9 && st.card_layers.load(Relaxed) == 0);
    }
    assert_eq!(tier.on_card("carol"), (0, 0), "nothing was ever put on the card");
    assert_eq!(tier.fallbacks.load(Relaxed), 0);
    let _ = std::fs::remove_file(path);
}

/// The card's per-layer attention (`fit::Attend::attend`, the path fitting takes when it cannot hand the card a
/// whole in-context stage) against plain attention worked out here in 64 bits, on keys and values in the stored
/// chunk layout with a last chunk that is not full; and the card's speed measurement.
#[test]
fn per_layer_attention_matches_plain_attention() {
    use tabicl_anywhere::fit::Attend;
    let Some(gpu) = card() else { return };
    let _one = exclusive();
    let (rows, chunk, d, heads) = (300usize, 128usize, 512usize, 8usize);
    let hd = d / heads;
    let val = |i: usize, s: usize| (((i * 7919 + s * 104_729) % 2003) as f32 - 1001.0) * 1e-3;
    let q: Vec<f32> = (0..rows * d).map(|i| val(i, 1) * 0.2).collect();
    let (k, v): (Vec<f32>, Vec<f32>) = ((0..rows * d).map(|i| val(i, 2)).collect(), (0..rows * d).map(|i| val(i, 3)).collect());
    // chunks of `chunk` rows, each keys `[d][rc]` then values `[d][rc]`
    let mut kv = Vec::with_capacity(2 * rows * d);
    for r0 in (0..rows).step_by(chunk) {
        let rc = chunk.min(rows - r0);
        for t in [&k, &v] {
            for j in 0..d {
                kv.extend((0..rc).map(|r| t[(r0 + r) * d + j]));
            }
        }
    }
    let got = tabicl_anywhere::wait::block_on(gpu.attend(&q, &kv, rows, chunk, d, heads)).unwrap();
    let mut bad = 0.0f32;
    for r in (0..rows).step_by(37) {
        for h in 0..heads {
            let s: Vec<f64> = (0..rows).map(|j| (0..hd).map(|i| q[r * d + h * hd + i] as f64 * k[j * d + h * hd + i] as f64).sum()).collect();
            let top = s.iter().copied().fold(f64::MIN, f64::max);
            let w: Vec<f64> = s.iter().map(|x| (x - top).exp()).collect();
            let sum: f64 = w.iter().sum();
            for i in 0..hd {
                let want = (0..rows).map(|j| w[j] * v[j * d + h * hd + i] as f64).sum::<f64>() / sum;
                bad = bad.max((got[r * d + h * hd + i] - want as f32).abs());
            }
        }
    }
    println!("per-layer attention on the card against plain attention: worst difference {bad:.2e}");
    assert!(bad < 1e-5, "{bad:.2e}");
    let rate = tabicl_anywhere::wait::block_on(gpu.calibrate(20.0)).unwrap();
    println!("card attention speed: {:.1} billion multiply-adds a second", rate / 1e9);
    assert!(rate > 1e8, "{rate}");
}
