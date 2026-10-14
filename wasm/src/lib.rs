//! tabicl-anywhere for JavaScript: load weights, fit a table, predict, all on bytes held in memory.
//!
//! ```js
//! const weights = new Weights(new Uint8Array(await (await fetch("classifier-v2.tirs")).arrayBuffer()));
//! const fitted = weights.fitClassifier(x /* Float64Array, rows*features, NaN = missing */, y /* Uint32Array */, rows, features, 8, "i16");
//! const proba = weights.predictProba(fitted, xNew, newRows);   // Float64Array, newRows * classes
//! ```
//!
//! `Weights` accepts this project's weight file (full or half size) or a published `.ckpt` checkpoint as downloaded.
//! A fitted model is plain bytes (`Uint8Array`): keep it in IndexedDB, a file, or a server.
//!
//! WebAssembly has one thread per instance here. To use several processor cores, run several Web Workers, each
//! with its own `Weights`, and give each ONE ensemble member: `fitClassifierMember` / `fitRegressorMember` return
//! a small fitted model holding just that member, `predictSums` / `predictRegression` answer for it, and
//! `combineProba` (or a plain average, for regression) puts the members' answers together. The result is the same,
//! bit for bit, as fitting the whole ensemble in one call.
//!
//! Graphics card: `Card.open()` (a Promise; rejects where the browser offers no WebGPU) gives a card, and the
//! `...Card` methods of `Weights` fit and predict one member on it, returning Promises. They run the same
//! shaders as the native build. Members go one after another on one card, from one worker. If the card fails
//! part-way (out of memory, lost), the Promise rejects; the caller then uses the processor methods instead.
//!
//! Tables: `Table` reads CSV text by the same rules as the command line (text columns become categories, empty
//! cells are missing) and hands back numbers plus a schema to keep with the fitted model.

use std::rc::Rc;
use std::sync::Arc;
use tabicl_anywhere::csv::{describe, parse, Schema};
use tabicl_anywhere::fit::{FitOpts, Kv};
use tabicl_anywhere::gpu::{Gpu, VramTier};
use tabicl_anywhere::model::{Load, Model};
use tabicl_anywhere::predict::{IclWeights, PredictOpts, Stats};
use tabicl_anywhere::store::{Bundle, MemStore};
use tabicl_anywhere::table::{finish_proba, fit_classifier_on, fit_classifier_on_async, fit_regressor_on, fit_regressor_on_async, member_count, Classifier, ClassifierOpts, Regressor};
use wasm_bindgen::prelude::*;

fn err(e: std::io::Error) -> JsError {
    JsError::new(&e.to_string())
}

#[wasm_bindgen]
pub struct Weights {
    model: Rc<Model>,
}

/// A graphics card, through the browser's WebGPU: the device, its tier for fitted models (one tenant: this
/// page) and its working memory.
#[wasm_bindgen]
pub struct Card {
    gpu: Arc<Gpu>,
    tier: Arc<VramTier>,
}

fn js(e: std::io::Error) -> JsValue {
    JsError::new(&e.to_string()).into()
}

/// A plain object `{ key: value, ... }` for JavaScript.
fn object(fields: &[(&str, JsValue)]) -> JsValue {
    let o = js_sys::Object::new();
    for (k, v) in fields {
        let _ = js_sys::Reflect::set(&o, &JsValue::from_str(k), v);
    }
    o.into()
}

#[wasm_bindgen]
impl Card {
    /// Ask the browser for its most capable graphics card and prepare it. `work_mb`: working memory it may use;
    /// `models_mb`: graphics memory for fitted models kept between predictions.
    pub async fn open(work_mb: f64, models_mb: f64) -> Result<Card, JsError> {
        let gpu = Arc::new(Gpu::new_async().await.map_err(err)?);
        gpu.set_work_budget((work_mb.max(1.0) * 1048576.0) as usize);
        let budget = (models_mb.max(0.0) * 1048576.0) as usize;
        let tier = VramTier::new(gpu.clone(), budget, budget);
        Ok(Card { gpu, tier })
    }

    /// Change how much working memory the card may use, in MB.
    #[wasm_bindgen(js_name = setWorkMb)]
    pub fn set_work_mb(&self, mb: f64) {
        self.gpu.set_work_budget((mb.max(0.0) * 1048576.0) as usize);
    }

    /// The adapter as the browser describes it.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.gpu.name.clone()
    }

    /// True if the browser offers only a program standing in for a card (correct, but slow).
    #[wasm_bindgen(getter)]
    pub fn software(&self) -> bool {
        self.gpu.software
    }

    /// Multiply-adds per second in the card's attention, measured for about `ms` milliseconds (a Promise).
    pub fn calibrate(&self, ms: f64) -> js_sys::Promise {
        let gpu = self.gpu.clone();
        wasm_bindgen_futures::future_to_promise(async move { Ok(JsValue::from_f64(gpu.calibrate(ms).await.map_err(js)?)) })
    }

    /// `{ working, workingPeak, models }`: bytes of graphics memory in use now.
    pub fn memory(&self) -> JsValue {
        let (now, peak) = self.gpu.working();
        object(&[("working", JsValue::from_f64(now as f64)), ("workingPeak", JsValue::from_f64(peak as f64)), ("models", JsValue::from_f64(self.tier.on_card("page").0 as f64))])
    }

    /// Release everything on the card that is not in use: fitted models, working buffers, weights.
    pub fn release(&self) {
        self.tier.evict("page", None);
        self.gpu.release();
    }
}

fn opts(estimators: usize, kv: &str) -> Result<ClassifierOpts, JsError> {
    let kv = match kv {
        "f32" | "" => Kv::F32,
        "i16" => Kv::I16,
        "i8" => Kv::I8,
        other => return Err(JsError::new(&format!("cache format must be f32, i16 or i8, not {other}"))),
    };
    // there is no disk to spill to here: keep everything in memory
    Ok(ClassifierOpts { n_estimators: estimators.max(1), fit: FitOpts { kv, mem_budget: usize::MAX, ..Default::default() }, ..Default::default() })
}

fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o += "\\\"",
            '\\' => o += "\\\\",
            '\n' => o += "\\n",
            '\r' => o += "\\r",
            '\t' => o += "\\t",
            c if (c as u32) < 0x20 => o += &format!("\\u{:04x}", c as u32),
            c => o.push(c),
        }
    }
    o + "\""
}

/// Turn a checkpoint or weight file into a weight file, optionally half size (16-bit floats).
#[wasm_bindgen(js_name = toBundle)]
pub fn to_bundle(bytes: Vec<u8>, half: bool) -> Result<Vec<u8>, JsError> {
    tabicl_anywhere::ckpt::bundle_bytes(bytes, half).map_err(err)
}

/// How fast this machine runs the inner loop, in multiply-adds per second, measured for about `ms` milliseconds.
/// Needs no weights, so a page can call it at once and use it to estimate how long a fit will take.
#[wasm_bindgen]
pub fn calibrate(ms: f64) -> f64 {
    tabicl_anywhere::nn::calibrate(ms)
}

/// Bytes of memory this WebAssembly instance holds right now. It only ever grows (up to 4 GB), so after an
/// operation this is the most that operation needed.
#[wasm_bindgen(js_name = memoryBytes)]
pub fn memory_bytes() -> f64 {
    #[cfg(target_arch = "wasm32")]
    return core::arch::wasm32::memory_size(0) as f64 * 65536.0;
    #[cfg(not(target_arch = "wasm32"))]
    0.0
}

/// Describe the columns of CSV text, as JSON:
/// `{"rows": N, "columns": [{"name", "numeric", "distinct", "missing", "examples": [..]}]}`.
#[wasm_bindgen(js_name = describeCsv)]
pub fn describe_csv(text: &str) -> Result<String, JsError> {
    let rows = parse(text);
    let cols = describe(&rows).map_err(err)?;
    let list: Vec<String> = cols
        .iter()
        .map(|c| {
            format!(
                "{{\"name\":{},\"numeric\":{},\"distinct\":{},\"missing\":{},\"examples\":[{}]}}",
                json_str(&c.name),
                c.numeric,
                c.distinct,
                c.missing,
                c.examples.iter().map(|e| json_str(e)).collect::<Vec<_>>().join(",")
            )
        })
        .collect();
    Ok(format!("{{\"rows\":{},\"columns\":[{}]}}", rows.len().saturating_sub(1), list.join(",")))
}

/// A training table read from CSV text: features as numbers, the target, and the schema that turns later CSV
/// rows into the same numbers.
#[wasm_bindgen]
pub struct Table {
    schema: Schema,
    x: Vec<f64>,
    classes: Vec<u32>,
    numbers: Vec<f64>,
}

#[wasm_bindgen]
impl Table {
    /// Read `csv` (header row first). `regression`: the target is a number to predict; otherwise a class.
    #[wasm_bindgen(constructor)]
    pub fn new(csv: &str, target: &str, regression: bool) -> Result<Table, JsError> {
        let rows = parse(csv);
        if regression {
            let (schema, x, numbers) = Schema::learn_numeric(&rows, target).map_err(err)?;
            Ok(Table { schema, x, classes: Vec::new(), numbers })
        } else {
            let (schema, x, y) = Schema::learn(&rows, target).map_err(err)?;
            Ok(Table { schema, x, classes: y.iter().map(|v| *v as u32).collect(), numbers: Vec::new() })
        }
    }
    #[wasm_bindgen(getter)]
    pub fn rows(&self) -> usize {
        self.classes.len().max(self.numbers.len())
    }
    #[wasm_bindgen(getter)]
    pub fn features(&self) -> usize {
        self.schema.columns.len()
    }
    /// Number of classes (0 for regression).
    #[wasm_bindgen(getter)]
    pub fn classes(&self) -> usize {
        self.schema.labels.len()
    }
    /// The schema as text: store it with the fitted model and pass it to `featuresFromCsv` when predicting.
    #[wasm_bindgen(getter)]
    pub fn schema(&self) -> String {
        self.schema.to_text()
    }
    /// Class labels in class-index order, as a JSON array of strings.
    #[wasm_bindgen(getter)]
    pub fn labels(&self) -> String {
        format!("[{}]", self.schema.labels.iter().map(|l| json_str(l)).collect::<Vec<_>>().join(","))
    }
    /// `rows * features` numbers, row by row, NaN where a cell is missing.
    #[wasm_bindgen(getter)]
    pub fn x(&self) -> Vec<f64> {
        self.x.clone()
    }
    #[wasm_bindgen(getter, js_name = yClasses)]
    pub fn y_classes(&self) -> Vec<u32> {
        self.classes.clone()
    }
    #[wasm_bindgen(getter, js_name = yNumbers)]
    pub fn y_numbers(&self) -> Vec<f64> {
        self.numbers.clone()
    }
    /// How many ensemble members a request for `estimators` really gives on this table (a table with very few
    /// columns has fewer distinct arrangements).
    pub fn members(&self, estimators: usize) -> usize {
        let o = ClassifierOpts { n_estimators: estimators.max(1), ..Default::default() };
        member_count(&self.x, self.rows(), self.features(), self.classes().max(1), &o)
    }
}

/// Features (`rows * columns`, row by row) of new CSV rows, by column name, using a stored schema.
#[wasm_bindgen(js_name = featuresFromCsv)]
pub fn features_from_csv(schema: &str, csv: &str) -> Result<Vec<f64>, JsError> {
    Schema::from_text(schema).features(&parse(csv)).map_err(err)
}

/// The temperature class scores are divided by before they become probabilities.
#[wasm_bindgen(js_name = defaultTemperature)]
pub fn default_temperature() -> f64 {
    ClassifierOpts::default().temperature
}

/// Probabilities (`rows * classes`) from the members' score sums added together in member order.
#[wasm_bindgen(js_name = combineProba)]
pub fn combine_proba(sums: &[f64], classes: usize, members: usize, temperature: f64) -> Vec<f64> {
    let mut p = sums.to_vec();
    finish_proba(&mut p, classes, members, temperature);
    p
}

#[wasm_bindgen]
impl Weights {
    /// From the bytes of a weight bundle, or of a published PyTorch checkpoint.
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: Vec<u8>) -> Result<Weights, JsError> {
        let bytes = tabicl_anywhere::ckpt::bundle_bytes(bytes, false).map_err(err)?;
        let bundle = Bundle::from_source(Box::new(Arc::new(bytes))).map_err(err)?;
        Ok(Weights { model: Rc::new(Model::load(&bundle, Load::Fit).map_err(err)?) })
    }

    /// True for the regression checkpoint, false for the classifier.
    #[wasm_bindgen(getter, js_name = isRegression)]
    pub fn is_regression(&self) -> bool {
        self.model.max_classes == 0
    }

    /// Fit a classifier. `x` is `rows * features` numbers row by row (NaN where missing); `y` holds class indices
    /// `0..classes`. Returns the fitted model as bytes.
    #[wasm_bindgen(js_name = fitClassifier)]
    pub fn fit_classifier(&self, x: &[f64], y: &[u32], rows: usize, features: usize, estimators: usize, kv: &str) -> Result<Vec<u8>, JsError> {
        let y: Vec<usize> = y.iter().map(|v| *v as usize).collect();
        let out = MemStore::new();
        fit_classifier_on(&self.model, x, &y, rows, features, Box::new(out.clone()), &opts(estimators, kv)?, &[]).map_err(err)?;
        Ok(out.bytes())
    }

    /// Fit ONE member (`member`, counting from 0) of an ensemble of `estimators`. Returns a fitted model holding
    /// just that member. `schema` (may be empty) is stored inside it as `table/schema`.
    #[wasm_bindgen(js_name = fitClassifierMember)]
    pub fn fit_classifier_member(&self, x: &[f64], y: &[u32], rows: usize, features: usize, estimators: usize, member: usize, kv: &str, schema: &str) -> Result<Vec<u8>, JsError> {
        let y: Vec<usize> = y.iter().map(|v| *v as usize).collect();
        let out = MemStore::new();
        let o = ClassifierOpts { only: Some(vec![member]), ..opts(estimators, kv)? };
        let extra: &[(&str, &[u8])] = if schema.is_empty() { &[] } else { &[("table/schema", schema.as_bytes())] };
        fit_classifier_on(&self.model, x, &y, rows, features, Box::new(out.clone()), &o, extra).map_err(err)?;
        Ok(out.bytes())
    }

    /// Fit a regression model (needs the regression weights). Returns the fitted model as bytes.
    #[wasm_bindgen(js_name = fitRegressor)]
    pub fn fit_regressor(&self, x: &[f64], y: &[f64], rows: usize, features: usize, estimators: usize, kv: &str) -> Result<Vec<u8>, JsError> {
        let out = MemStore::new();
        fit_regressor_on(&self.model, x, y, rows, features, Box::new(out.clone()), &opts(estimators, kv)?, &[]).map_err(err)?;
        Ok(out.bytes())
    }

    /// As `fitClassifierMember`, for regression.
    #[wasm_bindgen(js_name = fitRegressorMember)]
    pub fn fit_regressor_member(&self, x: &[f64], y: &[f64], rows: usize, features: usize, estimators: usize, member: usize, kv: &str, schema: &str) -> Result<Vec<u8>, JsError> {
        let out = MemStore::new();
        let o = ClassifierOpts { only: Some(vec![member]), ..opts(estimators, kv)? };
        let extra: &[(&str, &[u8])] = if schema.is_empty() { &[] } else { &[("table/schema", schema.as_bytes())] };
        fit_regressor_on(&self.model, x, y, rows, features, Box::new(out.clone()), &o, extra).map_err(err)?;
        Ok(out.bytes())
    }

    /// Class probabilities, `rows * classes` numbers row by row, for new rows `x`.
    #[wasm_bindgen(js_name = predictProba)]
    pub fn predict_proba(&self, fitted: Vec<u8>, x: &[f64], rows: usize) -> Result<Vec<f64>, JsError> {
        let bundle = Bundle::from_source(Box::new(Arc::new(fitted))).map_err(err)?;
        let clf = Classifier::open(&bundle).map_err(err)?;
        let po = PredictOpts { rows_per_pass: 256, ..Default::default() };
        clf.predict_proba(&self.model, &mut IclWeights::Resident, x, rows, &po, &Stats::default()).map_err(err)
    }

    /// The score sums (`rows * classes`) of the members held in `fitted`, not yet probabilities. Add the arrays
    /// from all members' fitted models together, in member order, and pass the total to `combineProba`.
    #[wasm_bindgen(js_name = predictSums)]
    pub fn predict_sums(&self, fitted: Vec<u8>, x: &[f64], rows: usize) -> Result<Vec<f64>, JsError> {
        let bundle = Bundle::from_source(Box::new(Arc::new(fitted))).map_err(err)?;
        let clf = Classifier::open(&bundle).map_err(err)?;
        let po = PredictOpts { rows_per_pass: 256, ..Default::default() };
        clf.logit_sums(&self.model, &mut IclWeights::Resident, x, rows, &po, &Stats::default()).map_err(err)
    }

    /// Regression predictions: four numbers per row (mean, median, 5% and 95% quantiles). For a fitted model
    /// holding one member, average these over the members (add in member order, then multiply by 1 / members).
    #[wasm_bindgen(js_name = predictRegression)]
    pub fn predict_regression(&self, fitted: Vec<u8>, x: &[f64], rows: usize) -> Result<Vec<f64>, JsError> {
        let bundle = Bundle::from_source(Box::new(Arc::new(fitted))).map_err(err)?;
        let reg = Regressor::open(&bundle).map_err(err)?;
        let po = PredictOpts { rows_per_pass: 256, ..Default::default() };
        let r = reg.predict(&self.model, &mut IclWeights::Resident, x, rows, &[0.05, 0.95], &po, &Stats::default()).map_err(err)?;
        Ok((0..rows).flat_map(|i| [r.mean[i], r.median[i], r.quantiles[2 * i], r.quantiles[2 * i + 1]]).collect())
    }

    /// `fitClassifierMember` on the graphics card: a Promise of the fitted member's bytes (the same format, so
    /// it predicts on either the card or the processor).
    #[wasm_bindgen(js_name = fitClassifierMemberCard)]
    pub fn fit_classifier_member_card(
        &self, card: &Card, x: Vec<f64>, y: Vec<u32>, rows: usize, features: usize, estimators: usize, member: usize, kv: String, schema: String,
    ) -> Result<js_sys::Promise, JsError> {
        let (model, gpu) = (self.model.clone(), card.gpu.clone());
        let mut o = ClassifierOpts { only: Some(vec![member]), ..opts(estimators, &kv)? };
        let before = gpu.stages.load(std::sync::atomic::Ordering::Relaxed);
        o.fit.accel = tabicl_anywhere::fit::Accel(Some(gpu.clone()));
        Ok(wasm_bindgen_futures::future_to_promise(async move {
            let y: Vec<usize> = y.iter().map(|v| *v as usize).collect();
            let out = MemStore::new();
            let extra: &[(&str, &[u8])] = if schema.is_empty() { &[] } else { &[("table/schema", schema.as_bytes())] };
            fit_classifier_on_async(&model, &x, &y, rows, features, Box::new(out.clone()), &o, extra).await.map_err(js)?;
            if gpu.stages.load(std::sync::atomic::Ordering::Relaxed) == before {
                // every stage was declined (the card's working memory is too small for this table): the fit ran
                // on this worker's one processor thread, which the processor workers would have done sooner
                return Err(JsError::new("graphics card: its working memory is too small for this table, so it did none of the work").into());
            }
            Ok(js_sys::Uint8Array::from(&out.bytes()[..]).into())
        }))
    }

    /// `fitRegressorMember` on the graphics card (a Promise of the fitted member's bytes).
    #[wasm_bindgen(js_name = fitRegressorMemberCard)]
    pub fn fit_regressor_member_card(
        &self, card: &Card, x: Vec<f64>, y: Vec<f64>, rows: usize, features: usize, estimators: usize, member: usize, kv: String, schema: String,
    ) -> Result<js_sys::Promise, JsError> {
        let (model, gpu) = (self.model.clone(), card.gpu.clone());
        let mut o = ClassifierOpts { only: Some(vec![member]), ..opts(estimators, &kv)? };
        let before = gpu.stages.load(std::sync::atomic::Ordering::Relaxed);
        o.fit.accel = tabicl_anywhere::fit::Accel(Some(gpu.clone()));
        Ok(wasm_bindgen_futures::future_to_promise(async move {
            let out = MemStore::new();
            let extra: &[(&str, &[u8])] = if schema.is_empty() { &[] } else { &[("table/schema", schema.as_bytes())] };
            fit_regressor_on_async(&model, &x, &y, rows, features, Box::new(out.clone()), &o, extra).await.map_err(js)?;
            if gpu.stages.load(std::sync::atomic::Ordering::Relaxed) == before {
                // every stage was declined (the card's working memory is too small for this table): the fit ran
                // on this worker's one processor thread, which the processor workers would have done sooner
                return Err(JsError::new("graphics card: its working memory is too small for this table, so it did none of the work").into());
            }
            Ok(js_sys::Uint8Array::from(&out.bytes()[..]).into())
        }))
    }

    /// `predictSums` (classifier) or `predictRegression` with the fitted member on the graphics card. `key`
    /// names the member, so that it stays on the card for the next prediction. A Promise of
    /// `{ out, cardLayers, fallbacks, problem }`: `cardLayers` is how many network layers the card ran (0: the
    /// processor answered, for instance because the member does not fit the card's budget), `problem` why the
    /// card gave up on any part, if it did.
    #[wasm_bindgen(js_name = predictCard)]
    pub fn predict_card(&self, card: &Card, key: String, fitted: Vec<u8>, x: Vec<f64>, rows: usize) -> js_sys::Promise {
        let (model, gpu, tier) = (self.model.clone(), card.gpu.clone(), card.tier.clone());
        wasm_bindgen_futures::future_to_promise(async move {
            let bundle = Arc::new(Bundle::from_source(Box::new(Arc::new(fitted))).map_err(js)?);
            let gw = gpu.weights(&model);
            let before = tier.fallbacks.load(std::sync::atomic::Ordering::Relaxed);
            // every in-context problem of every member in this bundle goes on the card now, if it fits
            let members = bundle.i32s("table/meta").map_err(js)?[3] as usize;
            for i in 0..members {
                let prefix = format!("m{i}/");
                let nodes = tabicl_anywhere::predict::Cache::open(&bundle, &prefix).map_err(js)?.nodes.len();
                for node in 0..nodes {
                    tier.preload("page", &key, &bundle, &gw, &prefix, node).map_err(js)?;
                }
            }
            // the card cuts large batches itself; one pass over the rows is quickest
            let po = PredictOpts { rows_per_pass: rows.max(1), accel: Some(tier.handle("page", &key, bundle.clone(), gw)), ..Default::default() };
            let st = Stats::default();
            let out = if model.max_classes == 0 {
                let reg = Regressor::open(&bundle).map_err(js)?;
                let r = reg.predict_async(&model, &mut IclWeights::Resident, &x, rows, &[0.05, 0.95], &po, &st).await.map_err(js)?;
                (0..rows).flat_map(|i| [r.mean[i], r.median[i], r.quantiles[2 * i], r.quantiles[2 * i + 1]]).collect::<Vec<f64>>()
            } else {
                let clf = Classifier::open(&bundle).map_err(js)?;
                clf.logit_sums_async(&model, &mut IclWeights::Resident, &x, rows, &po, &st).await.map_err(js)?
            };
            let fallbacks = tier.fallbacks.load(std::sync::atomic::Ordering::Relaxed) - before;
            let problem = if fallbacks > 0 { tier.last_error().map_or(JsValue::NULL, |e| JsValue::from_str(&e)) } else { JsValue::NULL };
            Ok(object(&[
                ("out", js_sys::Float64Array::from(&out[..]).into()),
                ("cardLayers", JsValue::from_f64(st.card_layers.load(std::sync::atomic::Ordering::Relaxed) as f64)),
                ("fallbacks", JsValue::from_f64(fallbacks as f64)),
                ("problem", problem),
            ]))
        })
    }

    /// Forget a fitted member kept on the card (by the `key` given to `predictCard`).
    #[wasm_bindgen(js_name = forgetCard)]
    pub fn forget_card(&self, card: &Card, key: &str) {
        card.tier.evict("page", Some(key));
    }
}
