//! Fitting: run the training table through the network once and write to disk everything prediction will need
//! about it, so that prediction never looks at the training table again.
//!
//! What gets written, per ensemble member (names carry the member's prefix):
//!
//! * `col_kv`    `[groups][digits][blocks][2][e][inducing]`: for every feature group and column block, the keys and
//!   values of the inducing points after they have looked at that column's training cells.
//! * `tree`      the in-context problems (one, unless there are more classes than the model's native limit).
//! * `n{j}_scale` `[layers][d]`: what a query is multiplied by in each in-context layer of problem `j`.
//! * `n{j}_kv{l}` `[2 * rows * d]` in chunks of `chunk` rows, each chunk `[K: d][rows in chunk]` then `[V: ...]`:
//!   keys and values of the training rows in in-context layer `l`.
//!
//! Memory while fitting: one column's cells per thread (`rows * e` floats) and the row representations
//! (`rows * d` floats). The two large intermediates, the per-cell embeddings (`rows * groups * e`) and one layer's
//! keys and values (`2 * rows * d`), stay in memory only while each fits `FitOpts::mem_budget`; beyond that the
//! first goes to a scratch file and the second is read back from the cache being written.

use crate::kern::*;
use crate::model::Model;
use crate::nn::*;
use crate::store::{file_read_at, file_write_at, BundleWriter, ReadAt};
use crate::wait::{block_on, Later};
use crate::with_simd;
use rayon::prelude::*;
use std::fs::File;
use std::io;
use std::path::PathBuf;

/// Something that can do a whole layer's attention at once, in place of the CPU loops: a graphics card, say.
/// Given every row's query (`[rows][d]`, already scaled) and the layer's keys and values in cache layout (chunks
/// of `[K: d][rows in chunk]`, `[V: ...]`), it returns each row's attention output `[rows][d]` (before the
/// output projection).
///
/// It may also offer the two stages before that, for tables with a single label digit. Each says how much it
/// takes at a time (0: not offered, the default, and the CPU does the stage):
///
/// * `columns` does the column stage for `groups` feature groups at once. `cells` holds their training cells,
///   `[rows][groups][e]`, already through the input layer and with the target's embedding added; it is replaced
///   by the cells after the stage. `ind_q` holds each block's inducing-point queries (`[inducing][e]`, scaled),
///   and `rec` receives what `col_kv` stores for these groups: `[groups][blocks][2][e][inducing]`.
/// * `rows` does the row stage for `n` rows: `cells` is `[n][features][e]`, `reps` receives `[n][d]`.
///
/// The methods that do work return futures (`wait::Later`), because a graphics card in a browser can only be
/// waited for that way; an accelerator that can simply block returns futures that are ready at once.
pub trait Attend: Send + Sync {
    fn attend<'a>(&'a self, q: &'a [f32], kv: &'a [f32], rows: usize, chunk: usize, d: usize, heads: usize) -> Later<'a, io::Result<Vec<f32>>>;

    /// Whether `attend` can take a layer of `rows` rows at all (an accelerator with bounded working memory says
    /// no to tables that would not fit it; the CPU then does that attention).
    fn attend_fits(&self, _rows: usize, _d: usize) -> bool {
        true
    }

    fn column_batch(&self, _m: &Model, _rows: usize) -> usize {
        0
    }
    fn columns<'a>(&'a self, _m: &'a Model, _cells: &'a mut [f32], _rows: usize, _groups: usize, _ind_q: &'a [Vec<f32>], _rec: &'a mut [f32]) -> Later<'a, io::Result<()>> {
        unreachable!("column_batch said the column stage is not offered")
    }
    fn row_batch(&self, _m: &Model, _features: usize) -> usize {
        0
    }
    fn rows<'a>(&'a self, _m: &'a Model, _cells: &'a [f32], _n: usize, _reps: &'a mut [f32]) -> Later<'a, io::Result<()>> {
        unreachable!("row_batch said the row stage is not offered")
    }

    /// The whole in-context stage of one problem: `h` is `[rows][d]` (row representations with the target's
    /// embedding added), `scales` what each layer multiplies its queries by (`[layers][d]`). `sink` is called
    /// once per layer, in order, with that layer's keys and values, each `[rows][d]`. `false` (the default) if
    /// this is not offered for a table of this size; nothing has been done then.
    fn in_context<'a>(&'a self, _m: &'a Model, _h: &'a [f32], _rows: usize, _scales: &'a [f32], _sink: &'a mut dyn FnMut(usize, &[f32], &[f32]) -> io::Result<()>) -> Later<'a, io::Result<bool>> {
        crate::wait::ready(Ok(false))
    }
}

/// An optional accelerator for fitting (see `Attend`).
#[derive(Clone, Default)]
pub struct Accel(pub Option<std::sync::Arc<dyn Attend>>);

impl std::fmt::Debug for Accel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_some() { "accelerated" } else { "cpu" })
    }
}

#[derive(Clone, Debug)]
pub struct FitOpts {
    /// Training rows per stored chunk of keys and values. Prediction holds one chunk at a time.
    pub chunk: usize,
    /// Above this many bytes, each of the two large intermediates lives on disk instead of in memory.
    pub mem_budget: usize,
    /// Where the scratch file goes (deleted afterwards).
    pub scratch: Option<PathBuf>,
    /// How the in-context keys and values (nearly all of the cache) are stored.
    pub kv: Kv,
    /// Hand each in-context layer's attention to an accelerator (used only while a layer fits `mem_budget`).
    pub accel: Accel,
}

/// Number format of the stored in-context keys and values. Reading the cache is most of the time a single
/// prediction takes, so fewer bytes is faster as well as smaller.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kv {
    /// 4 bytes per number, exactly what fitting computed.
    F32,
    /// 2 bytes: each coordinate of each chunk is scaled to fill a 16-bit integer (about 4.5 digits kept).
    I16,
    /// 1 byte: the same with 8-bit integers (about 2 digits kept). Check accuracy on your data before relying on it.
    I8,
}

impl Kv {
    pub fn width(self) -> usize {
        match self {
            Kv::F32 => 4,
            Kv::I16 => 2,
            Kv::I8 => 1,
        }
    }
    pub fn from_code(c: i32) -> Kv {
        [Kv::F32, Kv::I16, Kv::I8][c as usize]
    }
    /// Bytes of one stored chunk of `rc` rows: for the integer formats, `2 d` scales come first.
    pub fn record(self, d: usize, rc: usize) -> usize {
        2 * d * rc * self.width() + if self == Kv::F32 { 0 } else { 8 * d }
    }
}

impl Default for FitOpts {
    fn default() -> FitOpts {
        FitOpts { chunk: 256, mem_budget: 256 << 20, scratch: None, kv: Kv::F32, accel: Accel::default() }
    }
}

/// One in-context problem. Usually there is exactly one, over all rows. With more classes than the model's
/// native limit the classes are split into a tree: an inner node tells groups of classes apart, a leaf tells the
/// classes of one group apart, and each node sees only the training rows of its own classes.
#[derive(Clone, Debug)]
pub struct Node {
    /// Training rows this node uses (`None`: all of them, in order).
    pub rows: Option<Vec<usize>>,
    /// Their targets as this node sees them: group index, class index within the leaf, or the regression target.
    pub y: Vec<f32>,
    /// Outputs to read from the decoder at this node.
    pub k: usize,
    pub leaf: bool,
    /// For a leaf: the class each output stands for. For an inner node: the node index of each group.
    pub map: Vec<usize>,
}

/// How a table will be fitted: label digits for the column stage and the tree of in-context problems.
#[derive(Clone, Debug)]
pub struct Plan {
    /// Mixed-radix bases of the class label (one entry, unused, unless there are more classes than native).
    pub bases: Vec<usize>,
    pub nodes: Vec<Node>,
}

impl Plan {
    /// `classes` is the number of classes, or for a regression model the number of outputs wanted.
    pub fn new(m: &Model, y: &[f32], classes: usize) -> Plan {
        let max = m.max_classes;
        if max == 0 || classes <= max {
            let map = (0..classes).collect();
            return Plan { bases: vec![classes.max(1)], nodes: vec![Node { rows: None, y: y.to_vec(), k: classes, leaf: true, map }] };
        }
        // balanced bases: D digits, each at most `max`, whose product covers the classes
        let digits = ((classes as f64).ln() / (max as f64).ln()).ceil() as usize;
        let k = ((classes as f64).powf(1.0 / digits as f64).ceil() as usize).min(max);
        let mut bases = vec![k; digits];
        let mut product = k.pow(digits as u32);
        for b in bases.iter_mut() {
            if product >= classes {
                break;
            }
            if *b < max {
                product = product / *b * (*b + 1);
                *b += 1;
            }
        }
        let mut nodes = Vec::new();
        let all: Vec<usize> = (0..y.len()).collect();
        Plan::grow(max, y, all, &mut nodes);
        Plan { bases, nodes }
    }

    fn grow(max: usize, y: &[f32], rows: Vec<usize>, nodes: &mut Vec<Node>) -> usize {
        let mut classes: Vec<usize> = rows.iter().map(|r| y[*r] as usize).collect();
        classes.sort_unstable();
        classes.dedup();
        let at = nodes.len();
        let local = |r: &usize| classes.binary_search(&(y[*r] as usize)).unwrap();
        if classes.len() <= max {
            let ys = rows.iter().map(|r| local(r) as f32).collect();
            nodes.push(Node { rows: Some(rows), y: ys, k: classes.len(), leaf: true, map: classes });
            return at;
        }
        let groups = classes.len().div_ceil(max).min(max);
        let mut group_of = vec![0usize; classes.len()];
        let (mut pos, mut left) = (0, classes.len());
        for g in 0..groups {
            let size = left.div_ceil(groups - g);
            group_of[pos..pos + size].fill(g);
            pos += size;
            left -= size;
        }
        let ys: Vec<f32> = rows.iter().map(|r| group_of[local(r)] as f32).collect();
        nodes.push(Node { rows: Some(rows.clone()), y: ys.clone(), k: groups, leaf: false, map: Vec::new() });
        for g in 0..groups {
            let sub: Vec<usize> = rows.iter().zip(&ys).filter(|(_, yg)| **yg as usize == g).map(|(r, _)| *r).collect();
            let child = Plan::grow(max, y, sub, nodes);
            nodes[at].map.push(child);
        }
        at
    }

    /// The tree as integers: per node `leaf, rows, k, map[0..k]`.
    fn tree(&self, total_rows: usize) -> Vec<i32> {
        self.nodes.iter().flat_map(|n| [n.leaf as i32, n.rows.as_ref().map_or(total_rows, |r| r.len()) as i32, n.k as i32].into_iter().chain(n.map.iter().map(|v| *v as i32))).collect()
    }
}

/// The tensors `fit_raw` will fill, to be passed to `BundleWriter::create` (with anything else the caller stores).
pub fn raw_entries(m: &Model, prefix: &str, rows: usize, features: usize, opts: &FitOpts, plan: &Plan) -> Vec<(String, String, Vec<usize>)> {
    let f = |n: &str, dims: Vec<usize>| (format!("{prefix}{n}"), "float32".to_string(), dims);
    let chunk = opts.chunk.max(8);
    let mut v = vec![
        (format!("{prefix}meta"), "int32".to_string(), vec![7]),
        (format!("{prefix}tree"), "int32".to_string(), vec![plan.tree(rows).len()]),
        f("col_kv", vec![features, plan.bases.len(), m.col.len(), 2, m.e, m.inds]),
    ];
    for (ni, node) in plan.nodes.iter().enumerate() {
        let nr = node.rows.as_ref().map_or(rows, |r| r.len());
        let kv_bytes = (nr / chunk) * opts.kv.record(m.d, chunk) + if nr % chunk > 0 { opts.kv.record(m.d, nr % chunk) } else { 0 };
        v.push(f(&format!("n{ni}_scale"), vec![m.icl_layers, m.d]));
        for l in 0..m.icl_layers {
            v.push((format!("{prefix}n{ni}_kv{l}"), "uint8".to_string(), vec![kv_bytes]));
        }
    }
    v
}

/// `rows * groups * e` floats, group-major, in memory or in a file.
enum Spill {
    Mem(Vec<f32>),
    Disk(File, PathBuf),
}

impl Spill {
    fn write(&self, at: usize, data: &[f32]) -> io::Result<()> {
        match self {
            // the in-memory case is written through `mem_parts` instead, which needs no lock
            Spill::Mem(_) => unreachable!(),
            Spill::Disk(f, _) => file_write_at(f, bytemuck::cast_slice(data), 4 * at as u64),
        }
    }
    fn read(&self, at: usize, out: &mut [f32]) -> io::Result<()> {
        match self {
            Spill::Mem(v) => {
                out.copy_from_slice(&v[at..at + out.len()]);
                Ok(())
            }
            Spill::Disk(f, _) => file_read_at(f, bytemuck::cast_slice_mut(out), 4 * at as u64),
        }
    }
}

impl Drop for Spill {
    fn drop(&mut self) {
        if let Spill::Disk(_, p) = self {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// Fit one table that is already numeric and normalised: `x` is `[rows][features]`; `y` holds class indices
/// `0..classes` (as floats) for a classifier, standardised targets for a regression model. `classes` is the number
/// of outputs prediction should return (classes, or the model's quantile count). Fills the tensors named by `raw_entries(m, prefix, ..)` in `out`.
pub fn fit_raw(m: &Model, x: &[f32], y: &[f32], rows: usize, features: usize, classes: usize, out: &BundleWriter, prefix: &str, opts: &FitOpts, plan: &Plan) -> io::Result<()> {
    block_on(fit_raw_async(m, x, y, rows, features, classes, out, prefix, opts, plan))
}

/// `fit_raw` in the form that can wait for an accelerator without blocking (see `wait`): the one place the
/// fitting pass is written.
pub async fn fit_raw_async(m: &Model, x: &[f32], y: &[f32], rows: usize, features: usize, classes: usize, out: &BundleWriter, prefix: &str, opts: &FitOpts, plan: &Plan) -> io::Result<()> {
    assert_eq!(x.len(), rows * features);
    assert_eq!(y.len(), rows);
    assert!(m.icl.len() == m.icl_layers && m.col[0].a1.is_some(), "fitting needs the model loaded with Load::Fit");
    let (e, d, inds, nb) = (m.e, m.d, m.inds, m.col.len());
    let chunk = opts.chunk.max(8);
    out.write_bytes(&format!("{prefix}meta"), bytemuck::cast_slice(&[rows as i32, features as i32, classes as i32, chunk as i32, opts.kv as i32, plan.bases.len() as i32, plan.nodes.len() as i32]))?;
    out.write_bytes(&format!("{prefix}tree"), bytemuck::cast_slice(&plan.tree(rows)))?;
    let bases = &plan.bases[..];
    let digits = bases.len();

    let clock = web_time::Instant::now();
    let trace = std::env::var_os("BROWSERTABICL_TRACE").is_some();
    // ---- column stage -------------------------------------------------------------------------------------
    // The inducing points' queries are the same for every column: work them out once per block.
    let ind_q: Vec<Vec<f32>> = m
        .col
        .iter()
        .map(|isab| {
            let a1 = isab.a1.as_ref().unwrap();
            let scale = a1.query_scale(rows);
            let mut q = vec![0.0f32; inds * e];
            let mut ws = Ws::new(e, a1.heads, a1.ff, 1);
            for (qi, xi) in q.chunks_exact_mut(e).zip(isab.ind.chunks_exact(e)) {
                query(pulp::Scalar::new(), a1.query_w(), &scale, xi, &mut ws.t, &mut ws.h64, &mut ws.mm, qi);
            }
            q
        })
        .collect();
    let a2_scale = vec![1.0 / (m.col[0].a2.hd() as f32).sqrt(); e];
    let rec_len = digits * nb * 2 * e * inds;
    let emb_len = rows * features * e;
    let mut spill = if emb_len * 4 <= opts.mem_budget {
        Spill::Mem(vec![0.0f32; emb_len])
    } else {
        let p = opts.scratch.clone().unwrap_or_else(std::env::temp_dir).join(format!("BrowserTabICL-emb-{}-{prefix:.8}.tmp", std::process::id()).replace('/', "_"));
        let f = File::options().read(true).write(true).create(true).truncate(true).open(&p)?;
        f.set_len(4 * emb_len as u64)?;
        Spill::Disk(f, p)
    };
    let col_name = format!("{prefix}col_kv");
    let col_one = |g: usize, src: &mut [f32]| -> io::Result<()> {
        let mut rec = vec![0.0f32; rec_len];
        {
            let (rec, ind_q, a2_scale) = (&mut rec[..], &ind_q[..], &a2_scale[..]);
            with_simd!(|simd| [m: &'a Model, x: &'a [f32], y: &'a [f32], g: usize, features: usize, chunk: usize, src: &'a mut [f32], rec: &'a mut [f32], ind_q: &'a [Vec<f32>], a2_scale: &'a [f32], bases: &'a [usize]] -> () {
                let (e, inds) = (m.e, m.inds);
                let rows = y.len();
                let mut vals = [0.0f32; 8];
                let vals = &mut vals[..m.group];
                let digits = bases.len();
                // with several label digits the column is embedded once per digit and the results averaged
                let (mut plain, mut sum) = if digits > 1 { (vec![0.0f32; rows * e], vec![0.0f32; rows * e]) } else { (Vec::new(), Vec::new()) };
                for r in 0..rows {
                    let cell = &mut src[r * e..(r + 1) * e];
                    group_values(&x[r * features..(r + 1) * features], g, m.group, vals);
                    if vals.iter().all(|v| *v == -100.0) {
                        cell.fill(-100.0);
                    } else {
                        linear(simd, &m.in_w, &m.in_b, vals, cell);
                    }
                }
                if digits > 1 {
                    plain.copy_from_slice(src);
                }
                let a1 = m.col[0].a1.as_ref().unwrap();
                let mut ws = Ws::new(e, a1.heads, a1.ff, chunk.max(inds));
                let mut softs: Vec<Soft> = (0..inds).map(|_| Soft::new(a1.heads, e)).collect();
                let (mut kt, mut vt) = (vec![0.0f32; e * chunk], vec![0.0f32; e * chunk]);
                let mut kv = vec![0.0f32; 2 * e];
                let mut hidden = vec![0.0f32; inds * e];
                for dg in 0..digits {
                if dg > 0 {
                    src.copy_from_slice(&plain);
                }
                let divisor: usize = bases[dg + 1..].iter().product();
                for r in 0..rows {
                    let yv = if digits > 1 { ((y[r] as usize / divisor) % bases[dg]) as f32 } else { y[r] };
                    m.add_y(&m.col_y, yv, &mut src[r * e..(r + 1) * e]);
                }
                let rec = &mut rec[dg * m.col.len() * 2 * e * inds..(dg + 1) * m.col.len() * 2 * e * inds];
                for (bi, isab) in m.col.iter().enumerate() {
                    let a1 = isab.a1.as_ref().unwrap();
                    // inducing points look at this column's training cells, a chunk of rows at a time
                    softs.iter_mut().for_each(|s| s.reset());
                    for r0 in (0..rows).step_by(chunk) {
                        let rc = chunk.min(rows - r0);
                        for r in 0..rc {
                            layer_norm(simd, &src[(r0 + r) * e..(r0 + r + 1) * e], &a1.n1w, &a1.n1b, &mut ws.t);
                            linear(simd, &a1.in_w[e * e..], &a1.in_b[e..], &ws.t, &mut kv);
                            for j in 0..e {
                                kt[j * rc + r] = kv[j];
                                vt[j * rc + r] = kv[e + j];
                            }
                        }
                        for (i, s) in softs.iter_mut().enumerate() {
                            s.update(simd, &ind_q[bi][i * e..(i + 1) * e], &kt[..e * rc], &vt[..e * rc], rc, &mut ws.sc, &mut ws.tmp);
                        }
                    }
                    for (i, s) in softs.iter_mut().enumerate() {
                        s.finish();
                        let hid = &mut hidden[i * e..(i + 1) * e];
                        linear(simd, &a1.out_w, &a1.out_b, &s.acc, hid);
                        for j in 0..e {
                            hid[j] += isab.ind[i * e + j];
                        }
                        feed_forward(simd, a1.ff1_w(), a1.ff2_w(), hid, &mut ws.t, &mut ws.h);
                    }
                    // their keys and values for the second attention: this is what prediction reuses
                    let a2 = &isab.a2;
                    let (rk, rv) = rec[bi * 2 * e * inds..(bi + 1) * 2 * e * inds].split_at_mut(e * inds);
                    for i in 0..inds {
                        layer_norm(simd, &hidden[i * e..(i + 1) * e], &a2.n1w, &a2.n1b, &mut ws.t);
                        linear(simd, &a2.in_w[e * e..], &a2.in_b[e..], &ws.t, &mut kv);
                        for j in 0..e {
                            rk[j * inds + i] = kv[j];
                            rv[j * inds + i] = kv[e + j];
                        }
                    }
                    for cell in src.chunks_exact_mut(e) {
                        cached_block(simd, a2, a2_scale, rk, rv, inds, cell, &mut ws);
                    }
                }
                if digits > 1 {
                    for (a, b) in sum.iter_mut().zip(src.iter()) {
                        *a += *b;
                    }
                }
                }
                if digits > 1 {
                    let inv = 1.0 / digits as f32;
                    for (a, b) in src.iter_mut().zip(&sum) {
                        *a = *b * inv;
                    }
                }
            });
        }
        out.write_f32_at(&col_name, g * rec_len, &rec)
    };
    // how many feature groups the accelerator takes at a time, if it does this stage at all
    let col_batch = match &opts.accel.0 {
        Some(acc) if digits == 1 => acc.column_batch(m, rows),
        _ => 0,
    };
    if let (Some(acc), true) = (&opts.accel.0, col_batch > 0) {
        for g0 in (0..features).step_by(col_batch) {
            let gn = col_batch.min(features - g0);
            // these groups' cells, row by row: the input layer and the target's embedding
            let mut cells = vec![0.0f32; rows * gn * e];
            cells.par_chunks_mut(gn * e).enumerate().for_each(|(r, row)| {
                let mut vals = [0.0f32; 8];
                let vals = &mut vals[..m.group];
                for (gi, cell) in row.chunks_exact_mut(e).enumerate() {
                    group_values(&x[r * features..(r + 1) * features], g0 + gi, m.group, vals);
                    if vals.iter().all(|v| *v == -100.0) {
                        cell.fill(-100.0);
                    } else {
                        linear(pulp::Scalar::new(), &m.in_w, &m.in_b, vals, cell);
                    }
                    m.add_y(&m.col_y, y[r], cell);
                }
            });
            let mut rec = vec![0.0f32; gn * rec_len];
            acc.columns(m, &mut cells, rows, gn, &ind_q, &mut rec).await?;
            out.write_f32_at(&col_name, g0 * rec_len, &rec)?;
            // back to one long run per group
            let by_group = |gi: usize, dst: &mut [f32]| {
                for (r, cell) in dst.chunks_exact_mut(e).enumerate() {
                    cell.copy_from_slice(&cells[(r * gn + gi) * e..(r * gn + gi + 1) * e]);
                }
            };
            match &mut spill {
                Spill::Mem(v) => v[g0 * rows * e..(g0 + gn) * rows * e].par_chunks_mut(rows * e).enumerate().for_each(|(gi, dst)| by_group(gi, dst)),
                Spill::Disk(..) => {
                    let mut one = vec![0.0f32; rows * e];
                    for gi in 0..gn {
                        by_group(gi, &mut one);
                        spill.write((g0 + gi) * rows * e, &one)?;
                    }
                }
            }
        }
    } else {
        match &mut spill {
            Spill::Mem(v) => v.par_chunks_mut(rows * e).enumerate().try_for_each(|(g, src)| col_one(g, src))?,
            Spill::Disk(..) => {
                let sp = &spill;
                (0..features).into_par_iter().try_for_each_init(
                    || vec![0.0f32; rows * e],
                    |src, g| {
                        col_one(g, src)?;
                        sp.write(g * rows * e, src)
                    },
                )?
            }
        }
    }

    if trace {
        eprintln!("fit: column stage done at {:.2} s", clock.elapsed().as_secs_f64());
    }
    // ---- row stage ----------------------------------------------------------------------------------------
    let tokens = m.cls + features;
    let rope = Rope::new(&m.rope_freqs, tokens);
    let rope = &rope;
    let mut reps_all = vec![0.0f32; rows * d];
    let row_batch = opts.accel.0.as_ref().map_or(0, |acc| acc.row_batch(m, features));
    if let (Some(acc), true) = (&opts.accel.0, row_batch > 0) {
        for (bi, reps) in reps_all.chunks_mut(row_batch * d).enumerate() {
            let (r0, n) = (bi * row_batch, reps.len() / d);
            // these rows' cells, row by row
            let mut cells = vec![0.0f32; n * features * e];
            match &spill {
                Spill::Mem(v) => cells.par_chunks_mut(features * e).enumerate().for_each(|(r, row)| {
                    for (g, cell) in row.chunks_exact_mut(e).enumerate() {
                        cell.copy_from_slice(&v[(g * rows + r0 + r) * e..(g * rows + r0 + r + 1) * e]);
                    }
                }),
                Spill::Disk(..) => {
                    let mut one = vec![0.0f32; n * e];
                    for g in 0..features {
                        spill.read((g * rows + r0) * e, &mut one)?;
                        for (r, cell) in one.chunks_exact(e).enumerate() {
                            cells[(r * features + g) * e..(r * features + g + 1) * e].copy_from_slice(cell);
                        }
                    }
                }
            }
            acc.rows(m, &cells, n, reps).await?;
        }
    } else {
        let row_chunk = 64;
        reps_all.par_chunks_mut(row_chunk * d).enumerate().try_for_each_init(
            || (RowWs::new(m, tokens), vec![0.0f32; tokens * e], vec![0.0f32; row_chunk * features * e]),
            |(ws, tok, buf), (ci, reps)| -> io::Result<()> {
                let r0 = ci * row_chunk;
                let rc = reps.len() / d;
                for g in 0..features {
                    spill.read(g * rows * e + r0 * e, &mut buf[g * rc * e..(g + 1) * rc * e])?;
                }
                with_simd!(|simd| [m: &'a Model, rope: &'a Rope, ws: &'a mut RowWs, tok: &'a mut [f32], buf: &'a [f32], reps: &'a mut [f32], rc: usize, tokens: usize] -> () {
                    let (e, d) = (m.e, m.d);
                    let features = tokens - m.cls;
                    for r in 0..rc {
                        tok[..m.cls * e].copy_from_slice(&m.cls_tokens);
                        for g in 0..features {
                            tok[(m.cls + g) * e..(m.cls + g + 1) * e].copy_from_slice(&buf[(g * rc + r) * e..(g * rc + r + 1) * e]);
                        }
                        let rep = &mut reps[r * d..(r + 1) * d];
                        row_forward(simd, m, rope, tok, tokens, ws, rep);
                    }
                });
                Ok(())
            },
        )?;
    }
    drop(spill);

    if trace {
        eprintln!("fit: row stage done at {:.2} s", clock.elapsed().as_secs_f64());
    }
    // ---- in-context stage ---------------------------------------------------------------------------------
    let all_rows = rows;
    // one chunk of keys and values in its stored form
    let pack = |c: &[f32]| -> Vec<u8> {
        if opts.kv == Kv::F32 {
            return bytemuck::cast_slice(c).to_vec();
        }
        // each long row (one coordinate over a chunk's rows) gets its own scale, so it fills the integer range
        let rc = c.len() / (2 * d);
        let top = if opts.kv == Kv::I16 { 32767.0f32 } else { 127.0 };
        let mut scales = vec![0.0f32; 2 * d];
        let mut rec = vec![0u8; opts.kv.record(d, rc)];
        let body = &mut rec[8 * d..];
        for (j, row) in c.chunks_exact(rc).enumerate() {
            let big = row.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let s = if big > 0.0 { big / top } else { 1.0 };
            scales[j] = s;
            let inv = 1.0 / s;
            if opts.kv == Kv::I16 {
                for (o, v) in body[2 * j * rc..2 * (j + 1) * rc].chunks_exact_mut(2).zip(row) {
                    o.copy_from_slice(&((v * inv).round() as i16).to_le_bytes());
                }
            } else {
                for (o, v) in body[j * rc..(j + 1) * rc].iter_mut().zip(row) {
                    *o = (v * inv).round() as i8 as u8;
                }
            }
        }
        rec[..8 * d].copy_from_slice(bytemuck::cast_slice(&scales));
        rec
    };
    for (ni, node) in plan.nodes.iter().enumerate() {
        let rows = node.rows.as_ref().map_or(all_rows, |r| r.len());
        let mut hcur = match &node.rows {
            None => std::mem::take(&mut reps_all),
            Some(r) => r.iter().flat_map(|src| reps_all[src * d..(src + 1) * d].iter().copied()).collect(),
        };
        for (h, y) in hcur.chunks_exact_mut(d).zip(&node.y) {
            m.add_y(&m.icl_y, *y, h);
        }
        // One layer's keys and values for every row: held in memory if that fits the budget, otherwise read back,
        // a chunk at a time, from where they were just written.
        let in_mem = 2 * rows * d * 4 <= opts.mem_budget;
        let mut kv = vec![0.0f32; if in_mem { 2 * rows * d } else { 0 }];
        let side = if in_mem || opts.kv == Kv::F32 {
            None
        } else {
            // integer storage and too big for memory: keep an exact copy of the layer in a scratch file
            let p = opts.scratch.clone().unwrap_or_else(std::env::temp_dir).join(format!("BrowserTabICL-kv-{}-{}.tmp", std::process::id(), prefix.replace('/', "_")));
            let f = File::options().read(true).write(true).create(true).truncate(true).open(&p)?;
            f.set_len(8 * (rows * d) as u64)?;
            Some(Spill::Disk(f, p))
        };
        let full = opts.kv.record(d, chunk);
        let mut scales = vec![0.0f32; m.icl_layers * d];
        if let Some(acc) = &opts.accel.0 {
            for (l, b) in m.icl.iter().enumerate() {
                scales[l * d..(l + 1) * d].copy_from_slice(&b.query_scale(rows));
            }
            // the accelerator runs the layers; each layer's keys and values come back row by row and are stored
            // here in chunks, coordinate by coordinate
            let mut store = |l: usize, k: &[f32], v: &[f32]| -> io::Result<()> {
                let name = format!("{prefix}n{ni}_kv{l}");
                k.par_chunks(chunk * d).zip(v.par_chunks(chunk * d)).enumerate().try_for_each(|(ci, (kc, vc))| {
                    let rc = kc.len() / d;
                    let mut kvc = vec![0.0f32; 2 * d * rc];
                    let (kt, vt) = kvc.split_at_mut(d * rc);
                    for r in 0..rc {
                        for j in 0..d {
                            kt[j * rc + r] = kc[r * d + j];
                            vt[j * rc + r] = vc[r * d + j];
                        }
                    }
                    out.write_bytes_at(&name, ci * full, &pack(&kvc))
                })
            };
            if acc.in_context(m, &hcur, rows, &scales, &mut store).await? {
                out.write_f32_at(&format!("{prefix}n{ni}_scale"), 0, &scales)?;
                continue;
            }
        }
        for (l, b) in m.icl.iter().enumerate() {
            let scale = b.query_scale(rows);
            scales[l * d..(l + 1) * d].copy_from_slice(&scale);
            let scale = &scale[..];
            let name = format!("{prefix}n{ni}_kv{l}");
            // keys and values of every training row, chunk by chunk
            let fill = |ws: &mut Ws, two: &mut [f32], kvc: &mut [f32], hc: &[f32]| {
                with_simd!(|simd| [b: &'a crate::model::Block, ws: &'a mut Ws, two: &'a mut [f32], kvc: &'a mut [f32], hc: &'a [f32]] -> () {
                    let d = b.d;
                    let rc = hc.len() / d;
                    let (kt, vt) = kvc.split_at_mut(d * rc);
                    for r in 0..rc {
                        layer_norm(simd, &hc[r * d..(r + 1) * d], &b.n1w, &b.n1b, &mut ws.t);
                        linear(simd, &b.in_w[d * d..], &b.in_b[d..], &ws.t, two);
                        for j in 0..d {
                            kt[j * rc + r] = two[j];
                            vt[j * rc + r] = two[d + j];
                        }
                    }
                });
            };
            let new = || (Ws::new(d, b.heads, b.ff, 1), vec![0.0f32; 2 * d], vec![0.0f32; if in_mem { 0 } else { 2 * d * chunk }]);
            if in_mem {
                kv.par_chunks_mut(2 * d * chunk).zip(hcur.par_chunks(chunk * d)).enumerate().try_for_each_init(new, |(ws, two, _), (ci, (kvc, hc))| {
                    fill(ws, two, kvc, hc);
                    out.write_bytes_at(&name, ci * full, &pack(kvc))
                })?;
            } else {
                hcur.par_chunks(chunk * d).enumerate().try_for_each_init(new, |(ws, two, buf), (ci, hc)| {
                    let kvc = &mut buf[..2 * hc.len()];
                    fill(ws, two, kvc, hc);
                    if let Some(sp) = &side {
                        sp.write(ci * 2 * d * chunk, kvc)?;
                    }
                    out.write_bytes_at(&name, ci * full, &pack(kvc))
                })?;
            }
            if l + 1 == m.icl.len() {
                break; // the training rows' own outputs of the last layer are never used
            }
            if let (Some(acc), true) = (&opts.accel.0, in_mem && opts.accel.0.as_ref().is_some_and(|a| a.attend_fits(rows, d))) {
                // the accelerator does the attention for all rows at once; the CPU does the rest of the block
                let mut q = vec![0.0f32; rows * d];
                q.par_chunks_mut(d).zip(hcur.par_chunks(d)).for_each_init(
                    || Ws::new(d, b.heads, b.ff, 1),
                    |ws, (qr, hr)| {
                        with_simd!(|simd| [b: &'a crate::model::Block, scale: &'a [f32], ws: &'a mut Ws, qr: &'a mut [f32], hr: &'a [f32]] -> () {
                            query(simd, b.query_w(), scale, hr, &mut ws.t, &mut ws.h64, &mut ws.mm, qr)
                        });
                    },
                );
                if trace {
                    eprintln!("fit: layer {l} queries ready at {:.2} s", clock.elapsed().as_secs_f64());
                }
                let t = web_time::Instant::now();
                let att = acc.attend(&q, &kv, rows, chunk, d, b.heads).await?;
                if trace {
                    eprintln!("fit: layer {l} attention on the accelerator took {:.2} s", t.elapsed().as_secs_f64());
                }
                hcur.par_chunks_mut(d).zip(att.par_chunks(d)).for_each_init(
                    || Ws::new(d, b.heads, b.ff, 1),
                    |ws, (x, a)| {
                        with_simd!(|simd| [b: &'a crate::model::Block, ws: &'a mut Ws, x: &'a mut [f32], a: &'a [f32]] -> () {
                            linear(simd, &b.out_w, &b.out_b, a, &mut ws.t);
                            for j in 0..x.len() {
                                x[j] += ws.t[j];
                            }
                            feed_forward(simd, b.ff1_w(), b.ff2_w(), x, &mut ws.t, &mut ws.h);
                        });
                    },
                );
                continue;
            }
            // every training row attends to all training rows; a batch of queries shares each chunk while it is hot
            let qb = 32;
            let kv = &kv[..];
            let (back, base): (Option<&dyn ReadAt>, u64) = match &side {
                _ if in_mem => (None, 0),
                Some(Spill::Disk(f, _)) => (Some(f), 0),
                _ => (Some(&*out.file), out.entry(&name).offset),
            };
            hcur.par_chunks_mut(qb * d).try_for_each_init(
            || (Ws::new(d, b.heads, b.ff, chunk), (0..qb).map(|_| Soft::new(b.heads, d)).collect::<Vec<_>>(), vec![0.0f32; qb * d], vec![0.0f32; if in_mem { 0 } else { 2 * d * chunk }]),
            |(ws, softs, qc, buf), hc| -> io::Result<()> {
                with_simd!(|simd| [b: &'a crate::model::Block, scale: &'a [f32], kv: &'a [f32], back: Option<&'a dyn ReadAt>, base: u64, rows: usize, chunk: usize, ws: &'a mut Ws, softs: &'a mut [Soft], hc: &'a mut [f32], qc: &'a mut [f32], buf: &'a mut [f32]] -> io::Result<()> {
                    let d = b.d;
                    let n = hc.len() / d;
                    for i in 0..n {
                        query(simd, b.query_w(), scale, &hc[i * d..(i + 1) * d], &mut ws.t, &mut ws.h64, &mut ws.mm, &mut qc[i * d..(i + 1) * d]);
                    }
                    softs.iter_mut().for_each(|s| s.reset());
                    for (ci, r0) in (0..rows).step_by(chunk).enumerate() {
                        let rc = chunk.min(rows - r0);
                        let kvc: &[f32] = match back {
                            None => &kv[2 * d * r0..2 * d * (r0 + rc)],
                            Some(f) => {
                                f.read_exact_at(bytemuck::cast_slice_mut(&mut buf[..2 * d * rc]), base + (8 * d * chunk * ci) as u64)?;
                                &buf[..2 * d * rc]
                            }
                        };
                        let (kt, vt) = kvc.split_at(d * rc);
                        for i in 0..n {
                            softs[i].update(simd, &qc[i * d..(i + 1) * d], kt, vt, rc, &mut ws.sc, &mut ws.tmp);
                        }
                    }
                    for i in 0..n {
                        softs[i].finish();
                        let x = &mut hc[i * d..(i + 1) * d];
                        linear(simd, &b.out_w, &b.out_b, &softs[i].acc, &mut ws.t);
                        for j in 0..d {
                            x[j] += ws.t[j];
                        }
                        feed_forward(simd, b.ff1_w(), b.ff2_w(), x, &mut ws.t, &mut ws.h);
                    }
                    Ok(())
                })
            },
        )?;
        }
        out.write_f32_at(&format!("{prefix}n{ni}_scale"), 0, &scales)?;
    }
    if trace {
        eprintln!("fit: in-context stage done at {:.2} s", clock.elapsed().as_secs_f64());
    }
    Ok(())
}
