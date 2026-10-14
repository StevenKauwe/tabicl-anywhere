//! The everyday interface: fit a classifier on a table of numbers and predict class probabilities for new rows,
//! doing what the reference's `TabICLClassifier` does around the network: preprocessing, an ensemble of members
//! that see the features in different orders with relabelled classes, and averaging their answers.
//!
//! Everything a fitted classifier is lives in one bundle: the preprocessing numbers, each member's settings, and
//! each member's cache (tensors prefixed `m0/`, `m1/`, ...), laid out in the order prediction reads them.

use crate::fit::{fit_raw_async, raw_entries, FitOpts, Plan};
use crate::model::Model;
use crate::predict::{predict_raw_async, Cache, IclWeights, PredictOpts, Stats};
use crate::prep::{ensemble, ColNorm, Member, Prep, COL_NORM_FIELDS};
use crate::store::{Bundle, BundleWriter, Storage};
use crate::wait::block_on;
use std::fs::File;
use std::io;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct ClassifierOpts {
    /// Ensemble members. Each one costs a full fit and, at prediction, a full trip through its own cache.
    pub n_estimators: usize,
    /// Normalisation variants to alternate between: `false` plain, `true` with the power transform.
    pub power_variants: Vec<bool>,
    pub outlier_threshold: f64,
    pub temperature: f64,
    pub seed: u32,
    pub fit: FitOpts,
    /// Fit only these members (indices into the ensemble of `n_estimators`), and write a bundle holding just
    /// them. Members do not depend on each other, so they can be fitted one each on separate workers and their
    /// answers combined afterwards (`Classifier::logit_sums`, `finish_proba`). `None`: all of them.
    pub only: Option<Vec<usize>>,
}

impl Default for ClassifierOpts {
    fn default() -> ClassifierOpts {
        ClassifierOpts { n_estimators: 8, power_variants: vec![false, true], outlier_threshold: 4.0, temperature: 0.9, seed: 42, fit: FitOpts::default(), only: None }
    }
}

/// Turn arbitrary integer labels into class indices `0..classes` (in increasing label order, as the reference's
/// label encoder does). Returns the indices and the sorted distinct labels, which map indices back.
pub fn encode_labels(labels: &[i64]) -> (Vec<usize>, Vec<i64>) {
    let mut classes = labels.to_vec();
    classes.sort_unstable();
    classes.dedup();
    (labels.iter().map(|l| classes.binary_search(l).unwrap()).collect(), classes)
}

fn bad(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

/// The members to fit: the whole ensemble, or the ones `opts.only` names.
fn chosen(all: Vec<Member>, opts: &ClassifierOpts) -> io::Result<Vec<Member>> {
    match &opts.only {
        None => Ok(all),
        Some(ix) => ix.iter().map(|i| all.get(*i).cloned().ok_or_else(|| bad(format!("member {i} asked for, but this ensemble has {}", all.len())))).collect(),
    }
}

/// How many members an ensemble of `n_estimators` really has for this table (a table with very few features and
/// classes has fewer distinct arrangements than were asked for).
pub fn member_count(x: &[f64], n: usize, h: usize, classes: usize, opts: &ClassifierOpts) -> usize {
    let (prep, _) = Prep::fit(x, n, h, &opts.power_variants, opts.outlier_threshold);
    ensemble(prep.keep.len(), classes.max(1), opts.power_variants.len(), opts.n_estimators.max(1), opts.seed).len()
}

/// Turn summed scores (`[rows][classes]`, from `Classifier::logit_sums`, added up over `members` members in
/// member order) into probabilities, in place.
pub fn finish_proba(sums: &mut [f64], classes: usize, members: usize, temperature: f64) {
    let inv = 1.0 / (members as f64 * temperature);
    for row in sums.chunks_exact_mut(classes) {
        let top = row.iter().fold(f64::MIN, |a, v| a.max(*v));
        row.iter_mut().for_each(|v| *v = ((*v - top) * inv).exp());
        let sum: f64 = row.iter().sum();
        row.iter_mut().for_each(|v| *v /= sum);
    }
}

/// Fit on `x` (`[n][h]`, 64-bit, NaN where a cell is missing) with class indices `y` (`0..classes`, every class
/// present), and write the fitted classifier to `path`.
pub fn fit_classifier(m: &Model, x: &[f64], y: &[usize], n: usize, h: usize, path: impl AsRef<Path>, opts: &ClassifierOpts) -> io::Result<()> {
    fit_classifier_with(m, x, y, n, h, path, opts, &[])
}

/// As `fit_classifier`, also storing the caller's own named byte blobs in the bundle (a table schema, say).
pub fn fit_classifier_with(m: &Model, x: &[f64], y: &[usize], n: usize, h: usize, path: impl AsRef<Path>, opts: &ClassifierOpts, extra: &[(&str, &[u8])]) -> io::Result<()> {
    let file = File::options().read(true).write(true).create(true).truncate(true).open(path)?;
    fit_classifier_on(m, x, y, n, h, Box::new(file), opts, extra)
}

/// As `fit_classifier_with`, writing to any storage: a file, or a `MemStore` where there are no files.
pub fn fit_classifier_on(m: &Model, x: &[f64], y: &[usize], n: usize, h: usize, store: Box<dyn Storage>, opts: &ClassifierOpts, extra: &[(&str, &[u8])]) -> io::Result<()> {
    block_on(fit_classifier_on_async(m, x, y, n, h, store, opts, extra))
}

/// `fit_classifier_on` in the form that can wait for an accelerator without blocking (see `wait`).
pub async fn fit_classifier_on_async(m: &Model, x: &[f64], y: &[usize], n: usize, h: usize, store: Box<dyn Storage>, opts: &ClassifierOpts, extra: &[(&str, &[u8])]) -> io::Result<()> {
    if x.len() != n * h || y.len() != n || n == 0 || h == 0 {
        return Err(bad(format!("table is {} values for {n} rows of {h} features, with {} targets", x.len(), y.len())));
    }
    let classes = y.iter().max().unwrap() + 1;
    if m.max_classes == 0 {
        return Err(bad("these weights are a regression model's; classification needs the classifier checkpoint".into()));
    }
    if let Some(c) = (0..classes).find(|c| !y.contains(c)) {
        return Err(bad(format!("class index {c} never occurs; class indices must be 0..classes with none unused")));
    }
    let (prep, tables) = Prep::fit(x, n, h, &opts.power_variants, opts.outlier_threshold);
    let k = prep.keep.len();
    let members = chosen(ensemble(k, classes, opts.power_variants.len(), opts.n_estimators.max(1), opts.seed), opts)?;
    let nv = prep.power.len();
    let mut ents: Vec<(String, String, Vec<usize>)> = vec![
        ("table/meta".into(), "int32".into(), vec![5]),
        ("table/temperature".into(), "float64".into(), vec![1]),
        ("table/impute".into(), "float64".into(), vec![h]),
        ("table/keep".into(), "int32".into(), vec![k]),
        ("table/power".into(), "int32".into(), vec![nv]),
        ("table/norms".into(), "float64".into(), vec![nv, k, COL_NORM_FIELDS]),
        ("table/members".into(), "int32".into(), vec![members.len(), 1 + k + classes]),
    ];
    for (name, data) in extra {
        ents.push((name.to_string(), "uint8".into(), vec![data.len()]));
    }
    let member_y = |mb: &Member| -> Vec<f32> { y.iter().map(|c| mb.classes[*c] as f32).collect() };
    let plans: Vec<Plan> = members.iter().map(|mb| Plan::new(m, &member_y(mb), classes)).collect();
    for (i, plan) in plans.iter().enumerate() {
        ents.extend(raw_entries(m, &format!("m{i}/"), n, k, &opts.fit, plan));
    }
    let refs: Vec<(&str, &str, Vec<usize>)> = ents.iter().map(|(a, b, c)| (a.as_str(), b.as_str(), c.clone())).collect();
    let w = BundleWriter::create_on(store, &refs)?;
    for (name, data) in extra {
        w.write_bytes(name, data)?;
    }
    let ints = |v: Vec<i32>| bytemuck::cast_slice::<i32, u8>(&v).to_vec();
    w.write_bytes("table/meta", &ints(vec![h as i32, k as i32, classes as i32, members.len() as i32, nv as i32]))?;
    w.write_bytes("table/temperature", bytemuck::cast_slice(&[opts.temperature]))?;
    w.write_bytes("table/impute", bytemuck::cast_slice(&prep.impute))?;
    w.write_bytes("table/keep", &ints(prep.keep.iter().map(|v| *v as i32).collect()))?;
    w.write_bytes("table/power", &ints(prep.power.iter().map(|v| *v as i32).collect()))?;
    let norms: Vec<f64> = prep.norms.iter().flatten().flat_map(|c| c.to_array()).collect();
    w.write_bytes("table/norms", bytemuck::cast_slice(&norms))?;
    let rows: Vec<i32> = members.iter().flat_map(|mb| std::iter::once(mb.norm).chain(mb.features.iter().copied()).chain(mb.classes.iter().copied()).map(|v| v as i32)).collect();
    w.write_bytes("table/members", &ints(rows))?;
    let mut xs = vec![0.0f32; n * k];
    let mut ys = vec![0.0f32; n];
    for (i, mb) in members.iter().enumerate() {
        let t = &tables[mb.norm];
        for r in 0..n {
            for (j, &src) in mb.features.iter().enumerate() {
                xs[r * k + j] = t[r * k + src] as f32;
            }
            ys[r] = mb.classes[y[r]] as f32;
        }
        fit_raw_async(m, &xs, &ys, n, k, classes, &w, &format!("m{i}/"), &opts.fit, &plans[i]).await?;
    }
    w.file.sync()
}

/// The preprocessing state and member rows stored by a fit, read back and checked for consistency, so a damaged
/// bundle (or a regression bundle opened as a classifier, or the other way round) is an error and not a panic.
/// `extra` is how many numbers follow each member's feature order (`classes` for a classifier, 1 for regression).
fn open_table(bundle: &Bundle, extra: impl Fn(usize) -> usize) -> io::Result<(Prep, Vec<Vec<i32>>, usize)> {
    let damaged = |what: &str| io::Error::new(io::ErrorKind::InvalidData, format!("fitted bundle is damaged or of the other kind: {what}"));
    let f64s = |name: &str| -> io::Result<Vec<f64>> { Ok(bytemuck::pod_collect_to_vec(&bundle.bytes(name)?)) };
    let meta = bundle.i32s("table/meta")?;
    if meta.len() != 5 || meta.iter().any(|v| *v < 0) {
        return Err(damaged("table/meta"));
    }
    let (h, k, classes, nm, nv) = (meta[0] as usize, meta[1] as usize, meta[2] as usize, meta[3] as usize, meta[4] as usize);
    let norms = f64s("table/norms")?;
    let impute = f64s("table/impute")?;
    let keep: Vec<usize> = bundle.i32s("table/keep")?.iter().map(|v| *v as usize).collect();
    let power: Vec<bool> = bundle.i32s("table/power")?.iter().map(|v| *v != 0).collect();
    if impute.len() != h || keep.len() != k || keep.iter().any(|j| *j >= h) || power.len() != nv || norms.len() != nv * k * COL_NORM_FIELDS {
        return Err(damaged("preprocessing"));
    }
    let prep = Prep { n_in: h, impute, keep, power, norms: (0..nv).map(|v| (0..k).map(|j| ColNorm::from_array(&norms[(v * k + j) * COL_NORM_FIELDS..])).collect()).collect() };
    let width = 1 + k + extra(classes);
    let rows = bundle.i32s("table/members")?;
    if nm == 0 || rows.len() != nm * width {
        return Err(damaged("members"));
    }
    let members: Vec<Vec<i32>> = rows.chunks_exact(width).map(|r| r.to_vec()).collect();
    for r in &members {
        let ok_norm = (r[0] as usize) < nv;
        let ok_feats = r[1..1 + k].iter().all(|v| *v >= 0 && (*v as usize) < k);
        let ok_classes = r[1 + k..].iter().all(|v| *v >= 0 && (*v as usize) < classes.max(1));
        if r[0] < 0 || !ok_norm || !ok_feats || !ok_classes {
            return Err(damaged("a member's settings"));
        }
    }
    Ok((prep, members, classes))
}

/// A fitted classifier, opened from its bundle. Holds only the small preprocessing state; caches stay in the bundle.
pub struct Classifier<'b> {
    pub bundle: &'b Bundle,
    pub prep: Prep,
    pub members: Vec<Member>,
    pub classes: usize,
    pub temperature: f64,
}

impl<'b> Classifier<'b> {
    pub fn open(bundle: &'b Bundle) -> io::Result<Classifier<'b>> {
        if bundle.has("table/target") {
            return Err(bad("this bundle is a fitted regression model, not a classifier".into()));
        }
        let (prep, rows, classes) = open_table(bundle, |c| c)?;
        if classes == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "fitted classifier has no classes"));
        }
        let k = prep.keep.len();
        let members =
            rows.iter().map(|r| Member { norm: r[0] as usize, features: r[1..1 + k].iter().map(|v| *v as usize).collect(), classes: r[1 + k..].iter().map(|v| *v as usize).collect() }).collect();
        let temperature: Vec<f64> = bytemuck::pod_collect_to_vec(&bundle.bytes("table/temperature")?);
        let temperature = *temperature.first().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "fitted classifier has no temperature"))?;
        Ok(Classifier { bundle, prep, members, classes, temperature })
    }

    /// Class probabilities `[n][classes]` for raw rows `x` (`[n][features as given to fit]`, NaN for missing).
    pub fn predict_proba(&self, m: &Model, w: &mut IclWeights, x: &[f64], n: usize, opts: &PredictOpts, st: &Stats) -> io::Result<Vec<f64>> {
        block_on(self.predict_proba_async(m, w, x, n, opts, st))
    }

    /// `predict_proba` in the form that can wait for an accelerator without blocking (see `wait`).
    pub async fn predict_proba_async(&self, m: &Model, w: &mut IclWeights, x: &[f64], n: usize, opts: &PredictOpts, st: &Stats) -> io::Result<Vec<f64>> {
        let mut sums = self.logit_sums_async(m, w, x, n, opts, st).await?;
        finish_proba(&mut sums, self.classes, self.members.len(), self.temperature);
        Ok(sums)
    }

    /// The members' raw scores added together, `[n][classes]` in class order, before they become probabilities.
    /// Sums from bundles holding different members of one ensemble add up (in member order) to exactly what the
    /// whole ensemble's bundle gives; `finish_proba` then does the rest.
    pub fn logit_sums(&self, m: &Model, w: &mut IclWeights, x: &[f64], n: usize, opts: &PredictOpts, st: &Stats) -> io::Result<Vec<f64>> {
        block_on(self.logit_sums_async(m, w, x, n, opts, st))
    }

    /// `logit_sums` in the form that can wait for an accelerator without blocking (see `wait`).
    pub async fn logit_sums_async(&self, m: &Model, w: &mut IclWeights, x: &[f64], n: usize, opts: &PredictOpts, st: &Stats) -> io::Result<Vec<f64>> {
        let (h, k, c) = (self.prep.n_in, self.prep.keep.len(), self.classes);
        if x.len() != n * h {
            return Err(bad(format!("{} values is not {n} rows of {h} features", x.len())));
        }
        // each normalisation variant of the rows, worked out once
        let mut variants = vec![vec![0.0f64; n * k]; self.prep.power.len()];
        for (v, t) in variants.iter_mut().enumerate() {
            for r in 0..n {
                self.prep.apply(v, &x[r * h..(r + 1) * h], &mut t[r * k..(r + 1) * k]);
            }
        }
        let mut avg = vec![0.0f64; n * c];
        let mut xs = vec![0.0f32; n * k];
        for (i, mb) in self.members.iter().enumerate() {
            let t = &variants[mb.norm];
            for r in 0..n {
                for (j, &src) in mb.features.iter().enumerate() {
                    xs[r * k + j] = t[r * k + src] as f32;
                }
            }
            let cache = Cache::open(self.bundle, &format!("m{i}/"))?;
            let logits = predict_raw_async(m, w, &cache, &xs, n, opts, st).await?;
            for r in 0..n {
                for cls in 0..c {
                    avg[r * c + cls] += logits[r * c + mb.classes[cls]] as f64;
                }
            }
        }
        Ok(avg)
    }
}

// ---- regression ----------------------------------------------------------------------------------------------

/// What `Regressor::predict` returns for each row.
#[derive(Clone, Debug, Default)]
pub struct Regression {
    /// Mean of the predicted distribution.
    pub mean: Vec<f64>,
    /// Its variance, in squared units of the standardised target as the reference reports it, shifted by the
    /// target's scale and mean exactly as the reference does (so compare like with like, not as a true variance).
    pub variance: Vec<f64>,
    pub median: Vec<f64>,
    /// `[rows][alphas]`: the requested quantiles.
    pub quantiles: Vec<f64>,
}

/// The reference's reading of a row of predicted quantiles (levels 1/1000 .. 999/1000) as a distribution:
/// sorted, straight lines between neighbours, exponential tails beyond the first and last.
pub struct QuantileDist {
    pub q: Vec<f32>,
    levels: Vec<f32>,
    tail_l: (f32, f32),
    tail_r: (f32, f32),
}

impl QuantileDist {
    pub fn new(mut q: Vec<f32>) -> QuantileDist {
        q.sort_by(f32::total_cmp);
        let n = q.len();
        let levels: Vec<f32> = (1..=n).map(|i| i as f32 / (n + 1) as f32).collect();
        let k = 20.min(n / 4).max(1);
        let slope = |xs: &[f32], ys: &[f32]| {
            let (mx, my) = (xs.iter().sum::<f32>() / k as f32, ys.iter().sum::<f32>() / k as f32);
            let cov = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum::<f32>() / k as f32;
            let var = xs.iter().map(|x| (x - mx) * (x - mx)).sum::<f32>() / k as f32;
            (cov / var.max(1e-6)).abs().clamp(0.01, 100.0)
        };
        let ln_l: Vec<f32> = levels[..k].iter().map(|a| a.max(1e-6).ln()).collect();
        let ln_r: Vec<f32> = levels[n - k..].iter().map(|a| (1.0 - a).max(1e-6).ln()).collect();
        let (bl, br) = (slope(&ln_l, &q[..k]), slope(&ln_r, &q[n - k..]));
        let tail_l = (bl, q[0] - bl * levels[0].max(1e-6).ln());
        let tail_r = (-br, q[n - 1] + br * (1.0 - levels[n - 1].min(1.0 - 1e-6)).ln());
        QuantileDist { q, levels, tail_l, tail_r }
    }

    pub fn mean(&self) -> f32 {
        self.q.iter().sum::<f32>() / self.q.len() as f32
    }

    /// Sample variance of the quantile values (divisor `n - 1`).
    pub fn variance(&self) -> f32 {
        let m = self.mean();
        self.q.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / (self.q.len() - 1) as f32
    }

    /// The value below which a fraction `alpha` of the distribution lies.
    pub fn icdf(&self, alpha: f32) -> f32 {
        let n = self.q.len();
        if alpha < self.levels[0] {
            return self.tail_l.0 * alpha.max(1e-6).ln() + self.tail_l.1;
        }
        if alpha > self.levels[n - 1] {
            return self.tail_r.0 * (1.0 - alpha).max(1e-6).ln() + self.tail_r.1;
        }
        if alpha >= self.levels[n - 1] {
            return self.q[n - 1];
        }
        // the segment whose left end is the last level <= alpha
        let i = self.levels[..n - 1].partition_point(|l| *l <= alpha).saturating_sub(1).min(n - 2);
        let t = ((alpha - self.levels[i]) / (self.levels[i + 1] - self.levels[i]).max(1e-6)).clamp(0.0, 1.0);
        self.q[i] + t * (self.q[i + 1] - self.q[i])
    }
}

/// Fit a regression model on `x` (`[n][h]`, NaN where missing) and numeric targets `y`; write it to `path`.
/// `m` must be a regression checkpoint.
pub fn fit_regressor(m: &Model, x: &[f64], y: &[f64], n: usize, h: usize, path: impl AsRef<Path>, opts: &ClassifierOpts) -> io::Result<()> {
    fit_regressor_with(m, x, y, n, h, path, opts, &[])
}

/// As `fit_regressor`, also storing the caller's own named byte blobs in the bundle.
pub fn fit_regressor_with(m: &Model, x: &[f64], y: &[f64], n: usize, h: usize, path: impl AsRef<Path>, opts: &ClassifierOpts, extra: &[(&str, &[u8])]) -> io::Result<()> {
    let file = File::options().read(true).write(true).create(true).truncate(true).open(path)?;
    fit_regressor_on(m, x, y, n, h, Box::new(file), opts, extra)
}

/// As `fit_regressor_with`, writing to any storage.
pub fn fit_regressor_on(m: &Model, x: &[f64], y: &[f64], n: usize, h: usize, store: Box<dyn Storage>, opts: &ClassifierOpts, extra: &[(&str, &[u8])]) -> io::Result<()> {
    block_on(fit_regressor_on_async(m, x, y, n, h, store, opts, extra))
}

/// `fit_regressor_on` in the form that can wait for an accelerator without blocking (see `wait`).
pub async fn fit_regressor_on_async(m: &Model, x: &[f64], y: &[f64], n: usize, h: usize, store: Box<dyn Storage>, opts: &ClassifierOpts, extra: &[(&str, &[u8])]) -> io::Result<()> {
    if m.max_classes != 0 {
        return Err(bad("these weights are a classifier's; regression needs the regression checkpoint".into()));
    }
    if x.len() != n * h || y.len() != n || n == 0 || h == 0 {
        return Err(bad(format!("table is {} values for {n} rows of {h} features, with {} targets", x.len(), y.len())));
    }
    // the reference narrows targets to 32 bits before standardising them
    let y32: Vec<f64> = y.iter().map(|v| *v as f32 as f64).collect();
    let ym = y32.iter().sum::<f64>() / n as f64;
    let var = y32.iter().map(|v| (v - ym) * (v - ym)).sum::<f64>() / n as f64;
    let ys = if var.sqrt() < 10.0 * f64::EPSILON { 1.0 } else { var.sqrt() };
    let (prep, tables) = Prep::fit(x, n, h, &opts.power_variants, opts.outlier_threshold);
    let k = prep.keep.len();
    let members = chosen(ensemble(k, 1, opts.power_variants.len(), opts.n_estimators.max(1), opts.seed), opts)?;
    let nv = prep.power.len();
    let mut ents: Vec<(String, String, Vec<usize>)> = vec![
        ("table/meta".into(), "int32".into(), vec![5]),
        ("table/target".into(), "float64".into(), vec![2]),
        ("table/impute".into(), "float64".into(), vec![h]),
        ("table/keep".into(), "int32".into(), vec![k]),
        ("table/power".into(), "int32".into(), vec![nv]),
        ("table/norms".into(), "float64".into(), vec![nv, k, COL_NORM_FIELDS]),
        ("table/members".into(), "int32".into(), vec![members.len(), 2 + k]),
    ];
    for (name, data) in extra {
        ents.push((name.to_string(), "uint8".into(), vec![data.len()]));
    }
    let yz: Vec<f32> = y32.iter().map(|v| ((v - ym) / ys) as f32).collect();
    let plan = Plan::new(m, &yz, m.outputs());
    for i in 0..members.len() {
        ents.extend(raw_entries(m, &format!("m{i}/"), n, k, &opts.fit, &plan));
    }
    let refs: Vec<(&str, &str, Vec<usize>)> = ents.iter().map(|(a, b, c)| (a.as_str(), b.as_str(), c.clone())).collect();
    let w = BundleWriter::create_on(store, &refs)?;
    let ints = |v: Vec<i32>| bytemuck::cast_slice::<i32, u8>(&v).to_vec();
    w.write_bytes("table/meta", &ints(vec![h as i32, k as i32, 0, members.len() as i32, nv as i32]))?;
    w.write_bytes("table/target", bytemuck::cast_slice(&[ym, ys]))?;
    for (name, data) in extra {
        w.write_bytes(name, data)?;
    }
    w.write_bytes("table/impute", bytemuck::cast_slice(&prep.impute))?;
    w.write_bytes("table/keep", &ints(prep.keep.iter().map(|v| *v as i32).collect()))?;
    w.write_bytes("table/power", &ints(prep.power.iter().map(|v| *v as i32).collect()))?;
    let norms: Vec<f64> = prep.norms.iter().flatten().flat_map(|c| c.to_array()).collect();
    w.write_bytes("table/norms", bytemuck::cast_slice(&norms))?;
    let rows: Vec<i32> = members.iter().flat_map(|mb| std::iter::once(mb.norm).chain(mb.features.iter().copied()).chain([0]).map(|v| v as i32)).collect();
    w.write_bytes("table/members", &ints(rows))?;
    let mut xs = vec![0.0f32; n * k];
    for (i, mb) in members.iter().enumerate() {
        let t = &tables[mb.norm];
        for r in 0..n {
            for (j, &src) in mb.features.iter().enumerate() {
                xs[r * k + j] = t[r * k + src] as f32;
            }
        }
        fit_raw_async(m, &xs, &yz, n, k, m.outputs(), &w, &format!("m{i}/"), &opts.fit, &plan).await?;
    }
    w.file.sync()
}

/// A fitted regression model, opened from its bundle.
pub struct Regressor<'b> {
    pub bundle: &'b Bundle,
    pub prep: Prep,
    pub members: Vec<Member>,
    pub target_mean: f64,
    pub target_scale: f64,
}

impl<'b> Regressor<'b> {
    pub fn open(bundle: &'b Bundle) -> io::Result<Regressor<'b>> {
        if !bundle.has("table/target") {
            return Err(bad("this bundle is not a fitted regression model (a classifier, perhaps)".into()));
        }
        let (prep, rows, _) = open_table(bundle, |_| 1)?;
        let k = prep.keep.len();
        let members = rows.iter().map(|r| Member { norm: r[0] as usize, features: r[1..1 + k].iter().map(|v| *v as usize).collect(), classes: vec![0] }).collect();
        let t: Vec<f64> = bytemuck::pod_collect_to_vec(&bundle.bytes("table/target")?);
        if t.len() != 2 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "fitted regression model: table/target is damaged"));
        }
        Ok(Regressor { bundle, prep, members, target_mean: t[0], target_scale: t[1] })
    }

    /// Mean, variance, median and the quantiles at `alphas` for raw rows `x` (`[n][features as given to fit]`).
    pub fn predict(&self, m: &Model, w: &mut IclWeights, x: &[f64], n: usize, alphas: &[f64], opts: &PredictOpts, st: &Stats) -> io::Result<Regression> {
        block_on(self.predict_async(m, w, x, n, alphas, opts, st))
    }

    /// `predict` in the form that can wait for an accelerator without blocking (see `wait`).
    pub async fn predict_async(&self, m: &Model, w: &mut IclWeights, x: &[f64], n: usize, alphas: &[f64], opts: &PredictOpts, st: &Stats) -> io::Result<Regression> {
        let (h, k) = (self.prep.n_in, self.prep.keep.len());
        if x.len() != n * h {
            return Err(bad(format!("{} values is not {n} rows of {h} features", x.len())));
        }
        let mut variants = vec![vec![0.0f64; n * k]; self.prep.power.len()];
        for (v, t) in variants.iter_mut().enumerate() {
            for r in 0..n {
                self.prep.apply(v, &x[r * h..(r + 1) * h], &mut t[r * k..(r + 1) * k]);
            }
        }
        let na = alphas.len();
        let mut out = Regression { mean: vec![0.0; n], variance: vec![0.0; n], median: vec![0.0; n], quantiles: vec![0.0; n * na] };
        let mut xs = vec![0.0f32; n * k];
        let back = |v: f32| v as f64 * self.target_scale + self.target_mean;
        for (i, mb) in self.members.iter().enumerate() {
            let t = &variants[mb.norm];
            for r in 0..n {
                for (j, &src) in mb.features.iter().enumerate() {
                    xs[r * k + j] = t[r * k + src] as f32;
                }
            }
            let cache = Cache::open(self.bundle, &format!("m{i}/"))?;
            let q = predict_raw_async(m, w, &cache, &xs, n, opts, st).await?;
            for (r, row) in q.chunks_exact(cache.classes).enumerate() {
                let d = QuantileDist::new(row.to_vec());
                out.mean[r] += back(d.mean());
                out.variance[r] += back(d.variance());
                out.median[r] += back(d.icdf(0.5));
                for (a, alpha) in alphas.iter().enumerate() {
                    out.quantiles[r * na + a] += back(d.icdf(*alpha as f32));
                }
            }
        }
        let inv = 1.0 / self.members.len() as f64;
        for v in out.mean.iter_mut().chain(&mut out.variance).chain(&mut out.median).chain(&mut out.quantiles) {
            *v *= inv;
        }
        Ok(out)
    }
}
