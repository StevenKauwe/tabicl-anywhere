//! BrowserTabICL for JavaScript: load weights, fit a table, predict, all on bytes held in memory.
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
//! Tables: `Table` reads CSV text by the same rules as the command line (text columns become categories, empty
//! cells are missing) and hands back numbers plus a schema to keep with the fitted model.

use browsertabicl::csv::{describe, parse, Schema};
use browsertabicl::fit::{FitOpts, Kv};
use browsertabicl::model::{Load, Model};
use browsertabicl::predict::{IclWeights, PredictOpts, Stats};
use browsertabicl::store::{Bundle, MemStore};
use browsertabicl::table::{finish_proba, fit_classifier_on, fit_regressor_on, member_count, Classifier, ClassifierOpts, Regressor};
use std::sync::Arc;
use wasm_bindgen::prelude::*;

fn err(e: std::io::Error) -> JsError {
    JsError::new(&e.to_string())
}

#[wasm_bindgen]
pub struct Weights {
    model: Model,
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
    browsertabicl::ckpt::bundle_bytes(bytes, half).map_err(err)
}

/// How fast this machine runs the inner loop, in multiply-adds per second, measured for about `ms` milliseconds.
/// Needs no weights, so a page can call it at once and use it to estimate how long a fit will take.
#[wasm_bindgen]
pub fn calibrate(ms: f64) -> f64 {
    browsertabicl::nn::calibrate(ms)
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
        .map(|c| format!("{{\"name\":{},\"numeric\":{},\"distinct\":{},\"missing\":{},\"examples\":[{}]}}", json_str(&c.name), c.numeric, c.distinct, c.missing, c.examples.iter().map(|e| json_str(e)).collect::<Vec<_>>().join(",")))
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
        let bytes = browsertabicl::ckpt::bundle_bytes(bytes, false).map_err(err)?;
        let bundle = Bundle::from_source(Box::new(Arc::new(bytes))).map_err(err)?;
        Ok(Weights { model: Model::load(&bundle, Load::Fit).map_err(err)? })
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
}
