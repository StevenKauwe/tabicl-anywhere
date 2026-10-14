//! Command line for BrowserTabICL.
//!
//! Once, to turn a published checkpoint into this crate's weight file:
//!
//!   browsertabicl convert tabicl-classifier-v2-20260212.ckpt classifier-v2.tirs
//!   browsertabicl convert IN OUT --half          half-size weights (16-bit floats); IN is a .ckpt or a .tirs
//!   browsertabicl split FILE PREFIX [--max-mb 14]   cut into PREFIX.00.bin ... plus PREFIX.json (for web hosting)
//!   browsertabicl join PREFIX.json OUT           put them back together
//!
//! Everyday use, on CSV files with a header row:
//!
//!   browsertabicl fit WEIGHTS train.csv --target COLUMN model.tirs   [--estimators 8] [--kv f32|i16|i8] [--chunk ROWS]
//!   browsertabicl predict WEIGHTS model.tirs new.csv                 [--key KEYFILE] [--resident] [--key-threads N] [--stats]
//!        prints CSV: the predicted label, then one probability column per class
//!   browsertabicl keygen KEYFILE                                     write a new random 32-byte key (hex)
//!   browsertabicl seal model.tirs model.sealed KEYFILE               encrypt a fitted model; predict reads it with --key
//!
//! Timing on already-numeric tables (no preprocessing, one ensemble member):
//!
//!   browsertabicl bench-synth ROWS TEST FEATURES CLASSES table.tirs
//!   browsertabicl bench-fit WEIGHTS table.tirs cache.tirs            [--kv ...] [--chunk ROWS]
//!   browsertabicl bench-predict WEIGHTS cache.tirs table.tirs        [--rows N] [--one-at-a-time] [--resident] [--key KEYFILE]
//!                                                             [--key-threads N] [--io BYTES] [--stages] [--against OTHER]
//!
//! Everywhere: --threads N sets the size of the thread pool.
#![forbid(unsafe_code)]

use browsertabicl::fit::{fit_raw, raw_entries, FitOpts};
use browsertabicl::model::{Load, Model};
use browsertabicl::predict::{predict_raw, Cache, IclWeights, PredictOpts, Stats};
use browsertabicl::store::{Bundle, BundleWriter};
use std::time::Instant;

/// Peak and current resident memory of this process in MB, from the kernel's own accounting (Linux).
fn rss_mb() -> (f64, f64) {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let get = |k: &str| s.lines().find(|l| l.starts_with(k)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0) / 1024.0;
    (get("VmHWM:"), get("VmRSS:"))
}

const USAGE: &str = "usage: browsertabicl convert|split|join|fit|predict|keygen|seal|bench-...   (see the top of src/bin/browsertabicl.rs)";

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

/// Open a fitted bundle, through decryption if `--key KEYFILE` was given.
fn open_model(path: &str, args: &[String]) -> std::io::Result<Bundle> {
    match flag(args, "--key") {
        None => Bundle::open(path),
        Some(k) => Bundle::from_source(Box::new(browsertabicl::crypt::Sealed::open(std::fs::File::open(path)?, &browsertabicl::crypt::read_key(k)?)?)),
    }
}

fn kv_flag(args: &[String]) -> browsertabicl::fit::Kv {
    match flag(args, "--kv").as_deref() {
        None | Some("f32") => browsertabicl::fit::Kv::F32,
        Some("i16") => browsertabicl::fit::Kv::I16,
        Some("i8") => browsertabicl::fit::Kv::I8,
        Some(o) => panic!("--kv f32|i16|i8, not {o}"),
    }
}

/// `--gpu`: do the heavy part of fitting on the graphics card (needs a build with `--features gpu`).
fn accel(args: &[String]) -> std::io::Result<browsertabicl::fit::Accel> {
    if !args.iter().any(|a| a == "--gpu") {
        return Ok(Default::default());
    }
    #[cfg(feature = "gpu")]
    let card = {
        let g = browsertabicl::gpu::Gpu::new()?;
        eprintln!("using {}", g.name);
        Ok(browsertabicl::fit::Accel(Some(std::sync::Arc::new(g))))
    };
    #[cfg(not(feature = "gpu"))]
    let card = Err(std::io::Error::other("this build has no graphics-card support; rebuild with --features gpu"));
    card
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(t) = flag(&args, "--threads") {
        rayon::ThreadPoolBuilder::new().num_threads(t.parse().expect("--threads N")).build_global().unwrap();
    }
    match args.first().map(|s| s.as_str()) {
        Some("bench-synth") => {
            let n: Vec<usize> = args[1..5].iter().map(|a| a.parse().expect("a number")).collect();
            let (rows, test, feats, classes) = (n[0], n[1], n[2], n[3]);
            let mut s = 0x9E3779B97F4A7C15u64;
            let mut unit = move || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 24) as f32
            };
            let mut normal = || (0..12).map(|_| unit()).sum::<f32>() - 6.0;
            let w: Vec<f32> = (0..feats * classes).map(|_| normal()).collect();
            let all = rows + test;
            let x: Vec<f32> = (0..all * feats).map(|_| normal()).collect();
            let y: Vec<i32> = (0..all)
                .map(|r| {
                    let mut score = |c: usize| (0..feats).map(|f| x[r * feats + f] * w[f * classes + c]).sum::<f32>() + 2.0 * normal();
                    (0..classes).map(|c| (score(c), c)).fold((f32::MIN, 0), |a, b| if b.0 > a.0 { b } else { a }).1 as i32
                })
                .collect();
            let b = BundleWriter::create(
                &args[5],
                &[("x_train", "float32", vec![rows, feats]), ("y_train", "int32", vec![rows]), ("x_test", "float32", vec![test, feats]), ("y_test", "int32", vec![test])],
            )?;
            b.write_f32_at("x_train", 0, &x[..rows * feats])?;
            b.write_f32_at("x_test", 0, &x[rows * feats..])?;
            b.write_bytes("y_train", bytemuck::cast_slice(&y[..rows]))?;
            b.write_bytes("y_test", bytemuck::cast_slice(&y[rows..]))?;
            println!("wrote {}: {rows} training rows, {test} test rows, {feats} features, {classes} classes", args[5]);
        }
        Some("bench-fit") => {
            let t0 = Instant::now();
            let m = Model::load(&Bundle::open(&args[1])?, Load::Fit)?;
            let data = Bundle::open(&args[2])?;
            let (x, y) = (data.f32s("x_train")?, data.i32s("y_train")?);
            let (rows, feats) = (y.len(), x.len() / y.len());
            let classes = *y.iter().max().unwrap() as usize + 1;

            let mut opts = FitOpts { kv: kv_flag(&args), accel: accel(&args)?, ..Default::default() };
            if let Some(v) = flag(&args, "--mem") {
                opts.mem_budget = v.parse::<usize>().expect("--mem MEGABYTES") << 20;
            }
            let yf: Vec<f32> = y.iter().map(|v| *v as f32).collect();
            let plan = browsertabicl::fit::Plan::new(&m, &yf, classes);
            let ents = raw_entries(&m, "", rows, feats, &opts, &plan);
            let ents: Vec<(&str, &str, Vec<usize>)> = ents.iter().map(|(a, b, c)| (a.as_str(), b.as_str(), c.clone())).collect();
            let w = BundleWriter::create(&args[3], &ents)?;
            let t1 = Instant::now();
            fit_raw(&m, &x, &yf, rows, feats, classes, &w, "", &opts, &plan)?;
            let size = std::fs::metadata(&args[3])?.len() as f64 / 1e6;
            println!("fit {rows} x {feats}, {classes} classes: load {:.2} s, fit {:.2} s, cache {size:.0} MB, peak memory {:.0} MB", (t1 - t0).as_secs_f64(), t1.elapsed().as_secs_f64(), rss_mb().0);
        }
        Some("bench-predict") => {
            let t0 = Instant::now();
            let resident = args.iter().any(|a| a == "--resident");
            let wb = Bundle::open(&args[1])?;
            let m = Model::load(&wb, if resident { Load::Predict } else { Load::PredictStreamed })?;
            let mut w = if resident { IclWeights::Resident } else { IclWeights::Streamed { bundle: wb, buf: Vec::new() } };
            let cb = open_model(&args[2], &args)?;
            let cache = Cache::open(&cb, "")?;
            let data = Bundle::open(&args[3])?;
            let (x, y) = (data.f32s("x_test")?, data.i32s("y_test")?);
            let mut opts = PredictOpts::default();
            if let Some(v) = flag(&args, "--io") {
                opts.io_bytes = v.parse().expect("--io BYTES");
            }
            if let Some(v) = flag(&args, "--key-threads") {
                opts.key_threads = v.parse().expect("--key-threads N");
            }
            let n = flag(&args, "--rows").map(|v| v.parse().expect("--rows N")).unwrap_or(y.len()).min(y.len());
            let single = args.iter().any(|a| a == "--one-at-a-time");
            opts.rows_per_pass = if single { 1 } else { n };
            // --warm MB: keep up to that much of the cache in memory and predict twice, to show cold against warm
            let warm = flag(&args, "--warm").map(|v| browsertabicl::warm::Warm::new(v.parse::<usize>().expect("--warm MEGABYTES") << 20, usize::MAX, true));
            if let Some(tier) = &warm {
                opts.warm = Some(tier.handle("bench", &args[2]));
                let t = Instant::now();
                predict_raw(&m, &mut w, &cache, &x[..n * cache.features], n, &opts, &Stats::default())?;
                println!("cold pass (filling the warm tier): {:.1} ms; tier now holds {:.0} MB", t.elapsed().as_secs_f64() * 1e3, tier.held("bench").0 as f64 / 1e6);
            }
            // --card MB: put the model in graphics memory (up to that budget). The first prediction is answered by
            // the CPU while it loads; then the card takes over.
            #[cfg(feature = "gpu")]
            let mut card = None;
            #[cfg(feature = "gpu")]
            if let Some(mb) = flag(&args, "--card") {
                use browsertabicl::gpu::{Gpu, GpuWeights, VramTier};
                let gpu = std::sync::Arc::new(Gpu::new()?);
                if let Some(w) = flag(&args, "--work") {
                    gpu.set_work_budget(w.parse::<usize>().expect("--work MEGABYTES") << 20);
                }
                card = Some(gpu.clone());
                let budget = mb.parse::<usize>().expect("--card MEGABYTES") << 20;
                let tier = VramTier::new(gpu.clone(), budget, budget);
                assert!(resident, "--card needs --resident (the in-context weights are uploaded from memory)");
                let gw = std::sync::Arc::new(GpuWeights::new(&gpu, &m));
                let gw_bytes = gw.bytes;
                opts.accel = Some(tier.handle("bench", &args[2], std::sync::Arc::new(open_model(&args[2], &args)?), gw));
                let t = Instant::now();
                predict_raw(&m, &mut w, &cache, &x[..n * cache.features], n, &opts, &Stats::default())?;
                let first = t.elapsed().as_secs_f64() * 1e3;
                while tier.ready("bench") == 0 && t.elapsed().as_secs() < 60 {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                println!(
                    "{}: first prediction (CPU, streaming) {first:.1} ms; card holds {:.0} MB of fitted model (and {:.0} MB of weights) after {:.0} ms",
                    gpu.name,
                    tier.held("bench").0 as f64 / 1e6,
                    gw_bytes as f64 / 1e6,
                    t.elapsed().as_secs_f64() * 1e3
                );
                predict_raw(&m, &mut w, &cache, &x[..n * cache.features], n, &opts, &Stats::default())?;
                // warm the pipeline
            }
            let loaded = t0.elapsed().as_secs_f64();
            let (base_peak, _) = rss_mb();
            let st = Stats::default();
            // --repeat N: predict N times and report the fastest and the middle one (steady-state latency)
            if let Some(reps) = flag(&args, "--repeat") {
                let mut times: Vec<f64> = Vec::new();
                for _ in 0..reps.parse::<usize>().expect("--repeat N") {
                    let t = Instant::now();
                    predict_raw(&m, &mut w, &cache, &x[..n * cache.features], n, &opts, &Stats::default())?;
                    times.push(t.elapsed().as_secs_f64() * 1e3);
                }
                times.sort_by(f64::total_cmp);
                println!("repeated {} times: fastest {:.1} ms, middle {:.1} ms, slowest {:.1} ms", times.len(), times[0], times[times.len() / 2], times[times.len() - 1]);
                #[cfg(feature = "gpu")]
                if let Some(g) = &card {
                    let (now, peak) = g.working();
                    println!("card working memory: {} MB now, at most {} MB, budget {} MB", now >> 20, peak >> 20, g.work_budget() >> 20);
                }
            }
            let t1 = Instant::now();
            let logits = predict_raw(&m, &mut w, &cache, &x[..n * cache.features], n, &opts, &st)?;
            let took = t1.elapsed().as_secs_f64();
            let right = logits.chunks_exact(cache.classes).zip(&y).filter(|(l, y)| (0..cache.classes).max_by(|a, b| l[*a].total_cmp(&l[*b])).unwrap() as i32 == **y).count();
            let (peak, _) = rss_mb();
            if let Some(other) = flag(&args, "--against") {
                let ob = Bundle::open(&other)?;
                let oc = Cache::open(&ob, "")?;
                // the other cache is always read by the plain CPU path, so this also checks any accelerator in use
                let plain = PredictOpts { warm: None, accel: None, ..opts.clone() };
                let l2 = predict_raw(&m, &mut w, &oc, &x[..n * cache.features], n, &plain, &Stats::default())?;
                let worst = logits.iter().zip(&l2).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                let arg = |l: &[f32]| (0..l.len()).max_by(|a, b| l[*a].total_cmp(&l[*b])).unwrap();
                let same = logits.chunks_exact(cache.classes).zip(l2.chunks_exact(cache.classes)).filter(|(a, b)| arg(a) == arg(b)).count();
                println!("against {other}: worst logit difference {worst:.4}, same answer on {same}/{n} rows");
            }
            if args.iter().any(|a| a == "--stages") {
                println!("{st}");
            }
            println!(
                "predict {n} rows against {} x {} ({}): start-up {:.1} ms, prediction {:.1} ms total, {:.2} ms per row; {right}/{n} right; peak memory {peak:.1} MB (was {base_peak:.1} MB before predicting)",
                cache.rows, cache.features, if resident { "weights in memory" } else { "weights streamed" }, loaded * 1e3, took * 1e3, took * 1e3 / n as f64
            );
        }
        Some("fit") => {
            use browsertabicl::csv::{parse, Schema};
            use browsertabicl::table::{fit_classifier_with, ClassifierOpts};
            let target = flag(&args, "--target").expect("--target COLUMN");
            let pos: Vec<&String> = {
                let mut skip = false;
                args[1..]
                    .iter()
                    .filter(|a| {
                        if skip {
                            skip = false;
                            return false;
                        }
                        if a.starts_with("--") {
                            skip = true;
                            return false;
                        }
                        true
                    })
                    .collect()
            };
            let [weights, train, out] = pos[..] else { panic!("fit WEIGHTS train.csv --target COLUMN model.tirs") };
            let t0 = Instant::now();
            let m = Model::load(&Bundle::open(weights)?, Load::Fit)?;
            let table = parse(&std::fs::read_to_string(train)?);
            if m.max_classes == 0 {
                // regression weights: the target is a number
                let (schema, x, y) = Schema::learn_numeric(&table, &target)?;
                let (n, h) = (y.len(), schema.columns.len());
                let mut opts = ClassifierOpts::default();
                opts.fit.kv = kv_flag(&args);
                if let Some(v) = flag(&args, "--estimators") {
                    opts.n_estimators = v.parse().expect("--estimators N");
                }
                browsertabicl::table::fit_regressor_with(&m, &x, &y, n, h, out, &opts, &[("table/schema", schema.to_text().as_bytes())])?;
                eprintln!(
                    "fitted regression on {n} rows, {h} features, {} ensemble members in {:.1} s; {out} is {:.0} MB; peak memory {:.0} MB",
                    opts.n_estimators,
                    t0.elapsed().as_secs_f64(),
                    std::fs::metadata(out)?.len() as f64 / 1e6,
                    rss_mb().0
                );
                return Ok(());
            }
            let (schema, x, y) = Schema::learn(&table, &target)?;
            let (n, h) = (y.len(), schema.columns.len());
            let mut opts = ClassifierOpts::default();
            opts.fit.kv = kv_flag(&args);
            opts.fit.accel = accel(&args)?;
            if let Some(v) = flag(&args, "--estimators") {
                opts.n_estimators = v.parse().expect("--estimators N");
            }
            if let Some(v) = flag(&args, "--chunk") {
                opts.fit.chunk = v.parse().expect("--chunk ROWS");
            }
            fit_classifier_with(&m, &x, &y, n, h, out, &opts, &[("table/schema", schema.to_text().as_bytes())])?;
            eprintln!(
                "fitted {n} rows, {h} features, {} classes, {} ensemble members in {:.1} s; {out} is {:.0} MB; peak memory {:.0} MB",
                schema.labels.len(),
                opts.n_estimators,
                t0.elapsed().as_secs_f64(),
                std::fs::metadata(out)?.len() as f64 / 1e6,
                rss_mb().0
            );
        }
        Some("predict") => {
            use browsertabicl::csv::{parse, Schema};
            use browsertabicl::table::Classifier;
            let resident = args.iter().any(|a| a == "--resident");
            let wb = Bundle::open(&args[1])?;
            let m = Model::load(&wb, if resident { Load::Predict } else { Load::PredictStreamed })?;
            let mut w = if resident { IclWeights::Resident } else { IclWeights::Streamed { bundle: wb, buf: Vec::new() } };
            let cb = open_model(&args[2], &args)?;
            let schema = Schema::from_text(&String::from_utf8_lossy(&cb.bytes("table/schema")?));
            let x = schema.features(&parse(&std::fs::read_to_string(&args[3])?))?;
            let n = x.len() / schema.columns.len().max(1);
            let mut opts = PredictOpts { rows_per_pass: 256, ..Default::default() };
            if let Some(v) = flag(&args, "--key-threads") {
                opts.key_threads = v.parse().expect("--key-threads N");
            }
            let st = Stats::default();
            let t0 = Instant::now();
            if m.max_classes == 0 {
                let reg = browsertabicl::table::Regressor::open(&cb)?;
                let r = reg.predict(&m, &mut w, &x, n, &[0.05, 0.95], &opts, &st)?;
                let mut out = String::from("prediction,median,q05,q95");
                for i in 0..n {
                    out += &format!("\n{:.6},{:.6},{:.6},{:.6}", r.mean[i], r.median[i], r.quantiles[2 * i], r.quantiles[2 * i + 1]);
                }
                println!("{out}");
                if args.iter().any(|a| a == "--stats") {
                    eprintln!("{st}\n{:.1} ms in all, peak memory {:.1} MB", t0.elapsed().as_secs_f64() * 1e3, rss_mb().0);
                }
                return Ok(());
            }
            let clf = Classifier::open(&cb)?;
            let proba = clf.predict_proba(&m, &mut w, &x, n, &opts, &st)?;
            let mut out = String::from("prediction");
            schema.labels.iter().for_each(|l| out += &format!(",p({l})"));
            for row in proba.chunks_exact(clf.classes) {
                let best = (0..clf.classes).max_by(|a, b| row[*a].total_cmp(&row[*b])).unwrap();
                out += &format!("\n{}", schema.labels[best]);
                row.iter().for_each(|p| out += &format!(",{p:.6}"));
            }
            println!("{out}");
            if args.iter().any(|a| a == "--stats") {
                eprintln!("{st}\n{:.1} ms in all, peak memory {:.1} MB", t0.elapsed().as_secs_f64() * 1e3, rss_mb().0);
            }
        }
        Some("bench-kern") => {
            // how fast the inner loops run on this machine, in billions of multiply-adds per second
            use browsertabicl::kern::*;
            use browsertabicl::with_simd;
            let time = |name: &str, flops: f64, f: &mut dyn FnMut()| {
                let t = Instant::now();
                let mut n = 0u64;
                while t.elapsed().as_millis() < 200 {
                    for _ in 0..50 {
                        f();
                    }
                    n += 50;
                }
                let per = t.elapsed().as_secs_f64() / n as f64;
                println!("{name:44} {:8.2} us   {:6.1} G multiply-adds/s", per * 1e6, flops / per / 1e9);
            };
            for (o, i) in [(128usize, 128usize), (256, 128), (128, 256), (512, 512), (1024, 512), (512, 1024)] {
                let w: Vec<f32> = (0..o * i).map(|k| (k % 13) as f32 * 0.01).collect();
                let x: Vec<f32> = (0..i).map(|k| (k % 7) as f32 * 0.1).collect();
                let mut y = vec![0.0f32; o];
                let (w, x, y) = (&w[..], &x[..], &mut y[..]);
                time(&format!("matvec {o} x {i}"), (o * i) as f64, &mut || with_simd!(|simd| [w: &'a [f32], x: &'a [f32], y: &'a mut [f32]] -> () { matvec(simd, w, x, y) }));
            }
            for (rows, len) in [(16usize, 128usize), (64, 256)] {
                let k: Vec<f32> = (0..rows * len).map(|k| (k % 13) as f32 * 0.01).collect();
                let q: Vec<f32> = (0..rows).map(|k| (k % 7) as f32 * 0.1).collect();
                let mut sc = vec![0.0f32; len];
                let mut out = vec![0.0f32; rows];
                let (k, q, sc, out) = (&k[..], &q[..], &mut sc[..], &mut out[..]);
                time(
                    &format!("scores: weighted_sum {rows} rows x {len}"),
                    (rows * len) as f64,
                    &mut || with_simd!(|simd| [k: &'a [f32], q: &'a [f32], sc: &'a mut [f32]] -> () { weighted_sum(simd, q, k, sc) }),
                );
                time(&format!("values: matvec {rows} x {len}"), (rows * len) as f64, &mut || with_simd!(|simd| [k: &'a [f32], sc: &'a [f32], out: &'a mut [f32]] -> () { matvec(simd, k, sc, out) }));
                time(&format!("exp over {len}"), len as f64, &mut || with_simd!(|_simd| [sc: &'a mut [f32]] -> () { sc.iter_mut().for_each(|v| *v = -0.5); exp_shift_sum(sc, 0.0); }));
            }
            let mut g = vec![0.3f32; 256];
            let g = &mut g[..];
            time("gelu over 256", 256.0, &mut || with_simd!(|_simd| [g: &'a mut [f32]] -> () { g.iter_mut().for_each(|v| *v = 0.3); gelu(g) }));
        }
        Some("bench-attention") => {
            // browsertabicl bench-attention WEIGHTS cache.tirs table.tirs [--rows N]: how few training rows carry the attention
            use browsertabicl::predict::{attention_profile, embed, represent};
            let m = Model::load(&Bundle::open(&args[1])?, Load::Predict)?;
            let cb = Bundle::open(&args[2])?;
            let cache = Cache::open(&cb, "")?;
            let data = Bundle::open(&args[3])?;
            let x = data.f32s("x_test")?;
            let n = flag(&args, "--rows").map(|v| v.parse().expect("--rows N")).unwrap_or(32).min(x.len() / cache.features);
            let opts = PredictOpts::default();
            let mut tok = embed(&m, &cache, &x[..n * cache.features], n, &opts)?;
            let mut rep = represent(&m, &mut tok, n);
            let drops = [0.01f32, 0.001, 0.0001];
            let prof = attention_profile(&m, &cache, &mut rep, n, &drops)?;
            println!("share of the {} training rows needed to keep all but the given fraction of attention weight", cache.rows);
            println!("(average over {n} test rows and {} heads)\n layer   drop 1%   drop 0.1%   drop 0.01%", m.icl_heads);
            for (l, p) in prof.iter().enumerate() {
                println!("  {l:>3}   {:>6.2}%   {:>7.2}%   {:>8.2}%", p[0] * 100.0, p[1] * 100.0, p[2] * 100.0);
            }
            let avg = |i: usize| prof.iter().map(|p| p[i]).sum::<f64>() / prof.len() as f64 * 100.0;
            println!("  all   {:>6.2}%   {:>7.2}%   {:>8.2}%", avg(0), avg(1), avg(2));
        }
        Some("convert") => {
            // IN is a published checkpoint or an existing weight file; --half stores large tensors in 16 bits
            let half = args.iter().any(|a| a == "--half");
            let out = browsertabicl::ckpt::bundle_bytes(std::fs::read(&args[1])?, half)?;
            std::fs::write(&args[2], &out)?;
            eprintln!("wrote {}: {:.1} MB{}", args[2], out.len() as f64 / 1e6, if half { ", large tensors as 16-bit floats" } else { "" });
        }
        Some("split") => {
            // split FILE PREFIX [--max-mb 14]: PREFIX.00.bin, PREFIX.01.bin, ... and PREFIX.json listing them
            let bytes = std::fs::read(&args[1])?;
            let max = flag(&args, "--max-mb").map(|v| v.parse::<usize>().expect("--max-mb N")).unwrap_or(14) << 20;
            let base = std::path::Path::new(&args[2]).file_name().expect("PREFIX names a file").to_string_lossy().into_owned();
            let mut listed = Vec::new();
            for (i, part) in browsertabicl::store::split_parts(&bytes, max).iter().enumerate() {
                std::fs::write(format!("{}.{i:02}.bin", args[2]), part)?;
                listed.push((format!("{base}.{i:02}.bin"), part.len()));
            }
            std::fs::write(format!("{}.json", args[2]), browsertabicl::store::parts_manifest(bytes.len(), &listed))?;
            eprintln!("wrote {} parts of at most {} MB and {}.json", listed.len(), max >> 20, args[2]);
        }
        Some("join") => {
            // join PREFIX.json OUT: the inverse of split, checking every part's length and the total
            let text = std::fs::read_to_string(&args[1])?;
            let dir = std::path::Path::new(&args[1]).parent().unwrap_or(std::path::Path::new("."));
            let out = browsertabicl::store::join_parts(&text, |name| std::fs::read(dir.join(name)))?;
            std::fs::write(&args[2], &out)?;
            eprintln!("wrote {}: {} bytes", args[2], out.len());
        }
        Some("keygen") => {
            let key = browsertabicl::crypt::new_key()?;
            let hex: String = key.iter().map(|b| format!("{b:02x}")).collect();
            // readable by its owner only (where the platform has such permissions), and never over an existing key
            let mut f = std::fs::File::options();
            f.write(true).create_new(true);
            #[cfg(unix)]
            std::os::unix::fs::OpenOptionsExt::mode(&mut f, 0o600);
            std::io::Write::write_all(&mut f.open(&args[1])?, (hex + "\n").as_bytes())?;
            eprintln!("wrote a new key to {}; keep it safe: without it a sealed model cannot be read", args[1]);
        }
        Some("seal") => {
            browsertabicl::crypt::seal_file(&args[1], &args[2], &browsertabicl::crypt::read_key(&args[3])?)?;
            eprintln!("sealed {} into {}", args[1], args[2]);
        }
        None | Some("help" | "--help" | "-h") => eprintln!("{USAGE}"),
        Some(other) => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("unknown command '{other}'\n{USAGE}"))),
    }
    Ok(())
}
