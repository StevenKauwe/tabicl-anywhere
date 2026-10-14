//! Prediction: test rows go through the network while the fitted cache is read from its source front to back,
//! once per pass, a bounded piece at a time. Nothing about the training table is ever held whole.
//!
//! Order of reads in one pass (and of the tensors in the bundle): `col_kv` group by group, `icl_scale`, then
//! `icl_kv0`, `icl_kv1`, ... chunk by chunk. All rows of a pass share each piece, so a batch of rows costs one
//! trip through the cache, not one per row.

use crate::fit::Kv;
use crate::kern::*;
use crate::model::Model;
use crate::nn::*;
use crate::store::Bundle;
use crate::wait::{block_on, Later};
use crate::with_simd;
use rayon::prelude::*;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use web_time::Instant;

/// Counters a caller can read after (or during) prediction: where the time went and how much was read.
/// Times are nanoseconds; the three marked "thread time" are summed over threads, the rest are wall-clock.
/// One `Stats` can be shared by any number of calls and threads; export it to your own metrics as you like.
#[derive(Default, Debug)]
pub struct Stats {
    pub rows: AtomicU64,
    pub passes: AtomicU64,
    pub columns_ns: AtomicU64,
    pub rows_ns: AtomicU64,
    pub in_context_ns: AtomicU64,
    /// Thread time reading in-context weights (zero when they are held in memory).
    pub weights_read_ns: AtomicU64,
    /// Thread time waiting for cache bytes from the source.
    pub cache_read_ns: AtomicU64,
    pub cache_bytes: AtomicU64,
    /// Thread time attending over the cached training rows.
    pub attention_ns: AtomicU64,
    /// In-context layers whose attention ran on an accelerator instead.
    pub card_layers: AtomicU64,
}

impl Stats {
    pub fn add(&self, field: &AtomicU64, since: Instant) {
        field.fetch_add(since.elapsed().as_nanos() as u64, Relaxed);
    }
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ms = |a: &AtomicU64| a.load(Relaxed) as f64 / 1e6;
        write!(
            f,
            "{} rows in {} passes: columns {:.1} ms, rows {:.1} ms, in-context {:.1} ms (thread time: weights read {:.1}, cache read {:.1} for {:.0} MB, attention {:.1}; {} layers on the accelerator)",
            self.rows.load(Relaxed), self.passes.load(Relaxed), ms(&self.columns_ns), ms(&self.rows_ns), ms(&self.in_context_ns), ms(&self.weights_read_ns), ms(&self.cache_read_ns), self.cache_bytes.load(Relaxed) as f64 / 1e6, ms(&self.attention_ns), self.card_layers.load(Relaxed)
        )
    }
}

/// Something that holds a fitted model's keys and values itself (a graphics card's memory, say) and can do a
/// layer's attention there. `attend` returns `None` when it is not ready (still loading, or over budget), and
/// the CPU path streams as usual; so the first prediction never waits for it.
///
/// The methods return futures (`wait::Later`): a graphics card in a browser can only be waited for that way.
pub trait AttendCached: Send + Sync {
    /// Attention outputs `[rows][d]` for queries `q` (`[rows][d]`, scaled) against problem `node` of the member
    /// whose tensors start with `prefix`, in-context layer `layer`.
    fn attend<'a>(&'a self, prefix: &'a str, node: usize, layer: usize, q: &'a [f32], d: usize, heads: usize) -> Later<'a, io::Result<Option<Vec<f32>>>>;

    /// The whole in-context stage at once, if the accelerator can: row representations `h` (`[rows][d]`) to the
    /// first `out_n` decoder outputs per row. `None` falls back to layer-by-layer.
    fn stage<'a>(&'a self, _prefix: &'a str, _node: usize, _h: &'a [f32], _d: usize, _out_n: usize) -> Later<'a, io::Result<Option<Vec<f32>>>> {
        crate::wait::ready(Ok(None))
    }

    /// All three stages at once for already-normalised rows `x` (`[rows][features]`), for a member with a single
    /// in-context problem. `None` falls back to the CPU for the first two stages.
    fn rows<'a>(&'a self, _prefix: &'a str, _x: &'a [f32], _out_n: usize) -> Later<'a, io::Result<Option<Vec<f32>>>> {
        crate::wait::ready(Ok(None))
    }
}

/// An optional accelerator for prediction (see `AttendCached`).
#[derive(Clone)]
pub struct CardAttend(pub std::sync::Arc<dyn AttendCached>);

impl std::fmt::Debug for CardAttend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("accelerated")
    }
}

#[derive(Clone, Debug)]
pub struct PredictOpts {
    /// Roughly how many bytes of cache to ask the source for at a time. Small keeps memory down; raise it (to
    /// tens of MB) when the source is slow per request, such as object storage.
    pub io_bytes: usize,
    /// Test rows handled per trip through the cache. Each row costs about `(cls + features) * e * 4` bytes.
    pub rows_per_pass: usize,
    /// Split the training rows into this many ranges and read
    /// them at once, one thread each. Each range holds its own `io_bytes` buffer, so this trades memory for
    /// latency. 1 keeps to a single reader.
    pub key_threads: usize,
    /// Keep the pieces of cache that go past in memory, and use them in place next time (see `warm`).
    pub warm: Option<crate::warm::WarmHandle>,
    /// Do the in-context attention on an accelerator once it holds the model (CPU streaming until then).
    pub accel: Option<CardAttend>,
}

impl Default for PredictOpts {
    fn default() -> PredictOpts {
        PredictOpts { io_bytes: 1 << 20, rows_per_pass: 256, key_threads: 1, warm: None, accel: None }
    }
}

/// A fitted member inside a bundle.
pub struct Cache<'b> {
    pub bundle: &'b Bundle,
    pub prefix: String,
    pub rows: usize,
    pub features: usize,
    pub classes: usize,
    pub chunk: usize,
    pub kv: Kv,
    /// Label digits the column stage was fitted with (1 unless there are more classes than native).
    pub digits: usize,
    pub nodes: Vec<NodeInfo>,
}

/// One in-context problem of a fitted member (see `fit::Node`).
#[derive(Clone, Debug)]
pub struct NodeInfo {
    pub leaf: bool,
    pub rows: usize,
    pub k: usize,
    pub map: Vec<usize>,
}

impl<'b> Cache<'b> {
    pub fn open(bundle: &'b Bundle, prefix: &str) -> io::Result<Cache<'b>> {
        let meta = bundle.i32s(&format!("{prefix}meta"))?;
        let tree = bundle.i32s(&format!("{prefix}tree"))?;
        let mut nodes = Vec::new();
        let mut at = 0;
        for _ in 0..meta[6] {
            let k = tree[at + 2] as usize;
            let leaf = tree[at] != 0;
            nodes.push(NodeInfo { leaf, rows: tree[at + 1] as usize, k, map: tree[at + 3..at + 3 + k].iter().map(|v| *v as usize).collect() });
            at += 3 + k;
        }
        Ok(Cache {
            bundle,
            prefix: prefix.to_string(),
            rows: meta[0] as usize,
            features: meta[1] as usize,
            classes: meta[2] as usize,
            chunk: meta[3] as usize,
            kv: Kv::from_code(meta[4]),
            digits: meta[5] as usize,
            nodes,
        })
    }
}

/// Where the in-context stage's weights come from during prediction.
pub enum IclWeights {
    /// Held in the `Model` (loaded with `Load::Predict`), shared by all threads.
    Resident,
    /// Read from the weight bundle a tensor group at a time into one reusable buffer (at most about 2 MB).
    Streamed { bundle: Bundle, buf: Vec<f32> },
}

impl IclWeights {
    /// Read the named tensors of layer `l` one after another into the buffer; returns their lengths.
    fn pull(&mut self, l: usize, names: &[&str], take: &[Option<usize>], st: &Stats) -> io::Result<Vec<usize>> {
        let IclWeights::Streamed { bundle, buf } = self else { unreachable!() };
        let mut lens = Vec::new();
        let mut at = 0;
        for (n, t) in names.iter().zip(take) {
            let name = format!("icl_predictor.tf_icl.blocks.{l}.{n}");
            let t0 = Instant::now();
            if !bundle.has(&name) && n.ends_with(".bias") {
                // a layer norm trained without bias: same length as the weight just read
                let len = *lens.last().unwrap();
                if buf.len() < at + len {
                    buf.resize(at + len, 0.0);
                }
                buf[at..at + len].fill(0.0);
                lens.push(len);
                at += len;
                continue;
            }
            let len = t.unwrap_or(bundle.entry(&name)?.len());
            if buf.len() < at + len {
                buf.resize(at + len, 0.0);
            }
            bundle.read_f32_at(&name, 0, &mut buf[at..at + len])?;
            st.add(&st.weights_read_ns, t0);
            lens.push(len);
            at += len;
        }
        Ok(lens)
    }

    fn parts<'a>(&'a self, lens: &[usize]) -> Vec<&'a [f32]> {
        let IclWeights::Streamed { buf, .. } = self else { unreachable!() };
        let mut at = 0;
        lens.iter()
            .map(|l| {
                at += l;
                &buf[at - l..at]
            })
            .collect()
    }

    fn query<'a>(&'a mut self, m: &'a Model, l: usize, st: &Stats) -> io::Result<QueryW<'a>> {
        if let IclWeights::Resident = self {
            return Ok(m.icl[l].query_w());
        }
        let d = m.d;
        let names = [
            "norm1.weight",
            "norm1.bias",
            "attn.in_proj_weight",
            "attn.in_proj_bias",
            "attn.ssmax_layer.query_mlp.0.weight",
            "attn.ssmax_layer.query_mlp.0.bias",
            "attn.ssmax_layer.query_mlp.2.weight",
            "attn.ssmax_layer.query_mlp.2.bias",
        ];
        let lens = self.pull(l, &names, &[None, None, Some(d * d), Some(d), None, None, None, None], st)?;
        let p = self.parts(&lens);
        Ok(QueryW { heads: m.icl_heads, n1w: p[0], n1b: p[1], qw: p[2], qb: p[3], ss: Some((p[4], p[5], p[6], p[7])) })
    }

    fn out<'a>(&'a mut self, m: &'a Model, l: usize, st: &Stats) -> io::Result<OutW<'a>> {
        if let IclWeights::Resident = self {
            return Ok(m.icl[l].out_w());
        }
        let lens = self.pull(l, &["attn.out_proj.weight", "attn.out_proj.bias"], &[None, None], st)?;
        let p = self.parts(&lens);
        Ok(OutW { w: p[0], b: p[1] })
    }

    fn ff1<'a>(&'a mut self, m: &'a Model, l: usize, st: &Stats) -> io::Result<Ff1W<'a>> {
        if let IclWeights::Resident = self {
            return Ok(m.icl[l].ff1_w());
        }
        let lens = self.pull(l, &["norm2.weight", "norm2.bias", "linear1.weight", "linear1.bias"], &[None; 4], st)?;
        let p = self.parts(&lens);
        Ok(Ff1W { n2w: p[0], n2b: p[1], l1w: p[2], l1b: p[3] })
    }

    fn ff2<'a>(&'a mut self, m: &'a Model, l: usize, st: &Stats) -> io::Result<Ff2W<'a>> {
        if let IclWeights::Resident = self {
            return Ok(m.icl[l].ff2_w());
        }
        let lens = self.pull(l, &["linear2.weight", "linear2.bias"], &[None; 2], st)?;
        let p = self.parts(&lens);
        Ok(Ff2W { l2w: p[0], l2b: p[1] })
    }
}

/// One stored chunk of keys and values, viewed in place: floats, or integers with their `2 d` scales.
#[derive(Clone, Copy)]
enum Chunk<'a> {
    F32(&'a [f32], &'a [f32]),
    I16(&'a [i16], &'a [i16], &'a [f32]),
    I8(&'a [i8], &'a [i8], &'a [f32]),
}

/// Run `f(scratch, item)` over the items, on the thread pool unless there is only one.
fn each<T: Send, W, I: Fn() -> W + Sync + Send, F: Fn(&mut W, T) + Sync + Send>(items: Vec<T>, init: I, f: F) {
    if items.len() <= 1 {
        let mut w = init();
        items.into_iter().for_each(|t| f(&mut w, t));
    } else {
        items.into_par_iter().for_each_init(init, f);
    }
}

/// Column stage for `n` test rows (`x` is `[n][features]`): returns each row's tokens `[n][cls + features][e]`,
/// class tokens first.
pub fn embed(m: &Model, c: &Cache, x: &[f32], n: usize, opts: &PredictOpts) -> io::Result<Vec<f32>> {
    let (e, inds, nb, feats) = (m.e, m.inds, m.col.len(), c.features);
    assert_eq!(x.len(), n * feats);
    let tokens = m.cls + feats;
    let mut tok = vec![0.0f32; n * tokens * e];
    for t in tok.chunks_exact_mut(tokens * e) {
        t[..m.cls * e].copy_from_slice(&m.cls_tokens);
    }
    let digits = c.digits;
    let rec_len = digits * nb * 2 * e * inds;
    let per_read = (opts.io_bytes / (4 * rec_len)).clamp(1, feats);
    let mut recs = vec![0.0f32; per_read * rec_len];
    let scale = vec![1.0 / (m.col[0].a2.hd() as f32).sqrt(); e];
    let name = format!("{}col_kv", c.prefix);
    for g0 in (0..feats).step_by(per_read) {
        let gc = per_read.min(feats - g0);
        c.bundle.read_f32_at(&name, g0 * rec_len, &mut recs[..gc * rec_len])?;
        let (recs, scale) = (&recs[..gc * rec_len], &scale[..]);
        let items: Vec<(&mut [f32], &[f32])> = tok.chunks_exact_mut(tokens * e).zip(x.chunks_exact(feats)).collect();
        each(
            items,
            || Ws::new(e, m.col_heads, m.col[0].a2.ff, inds),
            |ws, (t, xr)| {
                with_simd!(|simd| [m: &'a Model, ws: &'a mut Ws, t: &'a mut [f32], xr: &'a [f32], recs: &'a [f32], scale: &'a [f32], g0: usize, gc: usize, digits: usize] -> () {
                    let (e, inds) = (m.e, m.inds);
                    let rec_len = recs.len() / gc;
                    let mut vals = [0.0f32; 8];
                    let vals = &mut vals[..m.group];
                    for gi in 0..gc {
                        let g = g0 + gi;
                        let cell = &mut t[(m.cls + g) * e..(m.cls + g + 1) * e];
                        group_values(xr, g, m.group, vals);
                        if vals.iter().all(|v| *v == -100.0) {
                            cell.fill(-100.0);
                        } else {
                            linear(simd, &m.in_w, &m.in_b, vals, cell);
                        }
                        let per = 2 * e * inds;
                        let nb = m.col.len();
                        if digits == 1 {
                            for (bi, isab) in m.col.iter().enumerate() {
                                let (rk, rv) = recs[gi * rec_len + bi * per..gi * rec_len + (bi + 1) * per].split_at(e * inds);
                                cached_block(simd, &isab.a2, scale, rk, rv, inds, cell, ws);
                            }
                        } else {
                            // once per label digit, then the average (as at fit time)
                            let mut plain = [0.0f32; 512];
                            let mut sum = [0.0f32; 512];
                            let mut one = [0.0f32; 512];
                            plain[..e].copy_from_slice(cell);
                            for dg in 0..digits {
                                one[..e].copy_from_slice(&plain[..e]);
                                for (bi, isab) in m.col.iter().enumerate() {
                                    let at = gi * rec_len + (dg * nb + bi) * per;
                                    let (rk, rv) = recs[at..at + per].split_at(e * inds);
                                    cached_block(simd, &isab.a2, scale, rk, rv, inds, &mut one[..e], ws);
                                }
                                for i in 0..e {
                                    sum[i] += one[i];
                                }
                            }
                            for i in 0..e {
                                cell[i] = sum[i] / digits as f32;
                            }
                        }
                    }
                });
            },
        );
    }
    Ok(tok)
}

/// Row stage: tokens `[n][tokens][e]` (consumed as scratch) to row representations `[n][d]`.
pub fn represent(m: &Model, tok: &mut [f32], n: usize) -> Vec<f32> {
    let tokens = tok.len() / (n.max(1) * m.e);
    let rope = Rope::new(&m.rope_freqs, tokens);
    let mut rep = vec![0.0f32; n * m.d];
    let items: Vec<(&mut [f32], &mut [f32])> = tok.chunks_exact_mut(tokens * m.e).zip(rep.chunks_exact_mut(m.d)).collect();
    let rope = &rope;
    each(
        items,
        || RowWs::new(m, tokens),
        |ws, (t, r)| {
            with_simd!(|simd| [m: &'a Model, rope: &'a Rope, ws: &'a mut RowWs, t: &'a mut [f32], r: &'a mut [f32], tokens: usize] -> () {
                row_forward(simd, m, rope, t, tokens, ws, r)
            });
        },
    );
    rep
}

/// In-context stage for problem `node` of the member: row representations `[n][d]` (consumed) to that problem's
/// outputs `[n][k]`.
pub fn in_context(m: &Model, w: &mut IclWeights, c: &Cache, node: usize, h: &mut [f32], n: usize, opts: &PredictOpts, st: &Stats) -> io::Result<Vec<f32>> {
    block_on(in_context_async(m, w, c, node, h, n, opts, st))
}

/// `in_context` in the form that can wait for an accelerator without blocking (see `wait`).
pub async fn in_context_async(m: &Model, w: &mut IclWeights, c: &Cache<'_>, node: usize, h: &mut [f32], n: usize, opts: &PredictOpts, st: &Stats) -> io::Result<Vec<f32>> {
    let c =
        &Cache { bundle: c.bundle, prefix: c.prefix.clone(), rows: c.nodes[node].rows, features: c.features, classes: c.nodes[node].k, chunk: c.chunk, kv: c.kv, digits: c.digits, nodes: Vec::new() };
    let (d, heads, chunk) = (m.d, m.icl_heads, c.chunk);
    let ff = m.dec0b.len();
    if let Some(a) = &opts.accel {
        if let Some(out) = a.0.stage(&c.prefix, node, h, d, c.classes).await? {
            st.card_layers.fetch_add(m.icl_layers as u64, Relaxed);
            return Ok(out);
        }
    }
    let scales = c.bundle.f32s(&format!("{}n{node}_scale", c.prefix))?;
    let mut q = vec![0.0f32; n * d];
    let full = c.kv.record(d, chunk);
    let chunks_per_read = (opts.io_bytes / full).max(1);
    let n_chunks = c.rows.div_ceil(chunk);
    let reads: Vec<usize> = (0..n_chunks).step_by(chunks_per_read).collect();
    let threads = rayon::current_num_threads();
    let parts_n = opts.key_threads.clamp(1, threads).min(reads.len());
    /// One reader's share of the training rows: its buffers and its running averages for every test row.
    struct Part {
        /// Bytes as read (kept as u32 so they can be viewed as floats or integers in place).
        raw: Vec<u32>,
        softs: Vec<Soft>,
        ws: Ws,
    }
    let mut parts: Vec<Part> = (0..parts_n)
        .map(|_| Part { raw: vec![0u32; (chunks_per_read.min(n_chunks) * full).div_ceil(4)], softs: (0..n).map(|_| Soft::new(heads, d)).collect(), ws: Ws::new(d, heads, ff, chunk) })
        .collect();
    let mut qk = vec![0.0f32; if c.kv == Kv::F32 { 0 } else { parts_n * n * d }];
    let mut mid = vec![0.0f32; n * ff];
    let new_ws = || Ws::new(d, heads, ff, chunk);
    for l in 0..m.icl_layers {
        let scale = &scales[l * d..(l + 1) * d];
        let qw = w.query(m, l, st)?;
        each(q.chunks_exact_mut(d).zip(h.chunks_exact(d)).collect(), new_ws, |ws, (qr, hr)| {
            with_simd!(|simd| [qw: QueryW<'a>, scale: &'a [f32], ws: &'a mut Ws, qr: &'a mut [f32], hr: &'a [f32]] -> () {
                query(simd, qw, scale, hr, &mut ws.t, &mut ws.h64, &mut ws.mm, qr)
            });
        });
        let name = format!("{}n{node}_kv{l}", c.prefix);
        let q = &q[..];
        let on_card = match &opts.accel {
            Some(a) => a.0.attend(&c.prefix, node, l, q, d, heads).await?,
            None => None,
        };
        if on_card.is_none() {
            let per = reads.len().div_ceil(parts_n);
            let kvf = c.kv;
            let rows_total = c.rows;
            let run = |part: &mut Part, qk: &mut [f32], mine: &[usize], spread: bool| -> io::Result<()> {
                let Part { raw, softs, ws } = part;
                softs.iter_mut().for_each(|s| s.reset());
                for &c0 in mine {
                    let cn = chunks_per_read.min(n_chunks - c0);
                    let last_rows = (rows_total - (c0 + cn - 1) * chunk).min(chunk);
                    let bytes = (cn - 1) * full + kvf.record(d, last_rows);
                    let t = Instant::now();
                    let load = |words: &mut [u32]| -> io::Result<()> {
                        c.bundle.read_bytes_at(&name, c0 * full, &mut bytemuck::cast_slice_mut::<u32, u8>(words)[..bytes])?;
                        st.cache_bytes.fetch_add(bytes as u64, Relaxed);
                        Ok(())
                    };
                    // from the warm tier in place if there is one, otherwise into this reader's own buffer
                    let held = match &opts.warm {
                        Some(w) => Some(w.get(&name, (c0 * full) as u64, || {
                            let mut words = vec![0u32; bytes.div_ceil(4)];
                            load(&mut words)?;
                            Ok(words)
                        })?),
                        None => {
                            load(&mut raw[..])?;
                            None
                        }
                    };
                    let raw: &[u32] = held.as_ref().map_or(&raw[..], |p| &p.words[..]);
                    st.add(&st.cache_read_ns, t);
                    let t = Instant::now();
                    for ci in 0..cn {
                        let rc = if ci + 1 == cn { last_rows } else { chunk };
                        let rec = &raw[ci * full / 4..(ci * full + kvf.record(d, rc)) / 4];
                        // the chunk in place, as stored: `[K: d][rc]`, `[V: d][rc]`, after the scales if integers
                        let chunk_ref = if kvf == Kv::F32 {
                            let all: &[f32] = bytemuck::cast_slice(rec);
                            let (kt, vt) = all.split_at(d * rc);
                            Chunk::F32(kt, vt)
                        } else {
                            let (sc, body) = rec.split_at(2 * d);
                            let body: &[u8] = bytemuck::cast_slice(body);
                            let sc: &[f32] = bytemuck::cast_slice(sc);
                            if kvf == Kv::I16 {
                                let all: &[i16] = bytemuck::cast_slice(body);
                                let (kt, vt) = all.split_at(d * rc);
                                Chunk::I16(kt, vt, sc)
                            } else {
                                let all: &[i8] = bytemuck::cast_slice(body);
                                let (kt, vt) = all.split_at(d * rc);
                                Chunk::I8(kt, vt, sc)
                            }
                        };
                        let one = |ws: &mut Ws, ((s, qr), qs): ((&mut Soft, &[f32]), &mut [f32])| {
                            with_simd!(|simd| [ws: &'a mut Ws, s: &'a mut Soft, qr: &'a [f32], qs: &'a mut [f32], chunk_ref: Chunk<'a>, rc: usize] -> () {
                                let d = qr.len();
                                match chunk_ref {
                                    Chunk::F32(kt, vt) => s.update(simd, qr, kt, vt, rc, &mut ws.sc, &mut ws.tmp),
                                    Chunk::I16(kt, vt, sc) => {
                                        for i in 0..d {
                                            qs[i] = qr[i] * sc[i];
                                        }
                                        s.update_int(simd, qs, kt, vt, &sc[d..], rc, &mut ws.sc, &mut ws.tmp)
                                    }
                                    Chunk::I8(kt, vt, sc) => {
                                        for i in 0..d {
                                            qs[i] = qr[i] * sc[i];
                                        }
                                        s.update_int(simd, qs, kt, vt, &sc[d..], rc, &mut ws.sc, &mut ws.tmp)
                                    }
                                }
                            });
                        };
                        // with float storage `qk` is empty; hand each row an empty scratch then
                        let qd = if kvf == Kv::F32 { 0 } else { d };
                        let mut empty = Vec::new();
                        let qss: Vec<&mut [f32]> = if qd == 0 {
                            empty.resize_with(softs.len(), || -> &mut [f32] { &mut [] });
                            empty
                        } else {
                            qk.chunks_exact_mut(d).collect()
                        };
                        let items: Vec<_> = softs.iter_mut().zip(q.chunks_exact(d)).zip(qss).collect();
                        if spread {
                            each(items, new_ws, one);
                        } else {
                            items.into_iter().for_each(|it| one(ws, it));
                        }
                    }
                    st.add(&st.attention_ns, t);
                }
                Ok(())
            };
            if parts_n == 1 {
                run(&mut parts[0], &mut qk, &reads, true)?;
            } else {
                let qd = (n * d).max(1);
                let mut none: Vec<Vec<f32>> = (0..parts_n).map(|_| Vec::new()).collect();
                let qks: Vec<&mut [f32]> = if c.kv == Kv::F32 { none.iter_mut().map(|v| &mut v[..]).collect() } else { qk.chunks_mut(qd).collect() };
                parts.par_iter_mut().zip(reads.par_chunks(per)).zip(qks).try_for_each(|((part, mine), qk)| run(part, qk, mine, false))?;
                let (first, rest) = parts.split_at_mut(1);
                for p in rest.iter() {
                    first[0].softs.iter_mut().zip(&p.softs).for_each(|(a, b)| a.merge(b));
                }
            }
        }
        let softs = &mut parts[0].softs;
        if let Some(att) = &on_card {
            st.card_layers.fetch_add(1, Relaxed);
            for (s, a) in softs.iter_mut().zip(att.chunks_exact(d)) {
                s.acc.copy_from_slice(a);
                s.s.fill(1.0);
            }
        }
        let ow = w.out(m, l, st)?;
        each(softs.iter_mut().zip(h.chunks_exact_mut(d)).collect(), new_ws, |ws, (s, hr)| {
            with_simd!(|simd| [ow: OutW<'a>, ws: &'a mut Ws, s: &'a mut Soft, hr: &'a mut [f32]] -> () {
                s.finish();
                linear(simd, ow.w, ow.b, &s.acc, &mut ws.t);
                for i in 0..hr.len() {
                    hr[i] += ws.t[i];
                }
            });
        });
        let f1 = w.ff1(m, l, st)?;
        each(mid.chunks_exact_mut(ff).zip(h.chunks_exact(d)).collect(), new_ws, |ws, (mr, hr)| {
            with_simd!(|simd| [f1: Ff1W<'a>, ws: &'a mut Ws, mr: &'a mut [f32], hr: &'a [f32]] -> () {
                layer_norm(simd, hr, f1.n2w, f1.n2b, &mut ws.t);
                linear(simd, f1.l1w, f1.l1b, &ws.t, mr);
                gelu(mr);
            });
        });
        let f2 = w.ff2(m, l, st)?;
        each(mid.chunks_exact(ff).zip(h.chunks_exact_mut(d)).collect(), new_ws, |ws, (mr, hr)| {
            with_simd!(|simd| [f2: Ff2W<'a>, ws: &'a mut Ws, mr: &'a [f32], hr: &'a mut [f32]] -> () {
                linear(simd, f2.l2w, f2.l2b, mr, &mut ws.t);
                for i in 0..hr.len() {
                    hr[i] += ws.t[i];
                }
            });
        });
    }
    let mut logits = vec![0.0f32; n * c.classes];
    let classes = c.classes;
    each(
        logits.chunks_exact_mut(classes).zip(h.chunks_exact(d)).collect(),
        || (new_ws(), vec![0.0f32; m.dec2b.len()]),
        |(ws, all), (lr, hr)| {
            with_simd!(|simd| [m: &'a Model, ws: &'a mut Ws, all: &'a mut [f32], lr: &'a mut [f32], hr: &'a [f32]] -> () {
                layer_norm(simd, hr, &m.lnw, &m.lnb, &mut ws.t);
                linear(simd, &m.dec0w, &m.dec0b, &ws.t, &mut ws.h);
                gelu(&mut ws.h);
                linear(simd, &m.dec2w, &m.dec2b, &ws.h, all);
                lr.copy_from_slice(&all[..lr.len()]);
            });
        },
    );
    Ok(logits)
}

/// One in-context layer's keys and values for problem `node`, whole and row by row (`[rows][d]` each), widened
/// to floats whatever the stored format. For loading a model into an accelerator's memory.
pub fn layer_rows(c: &Cache, node: usize, l: usize, d: usize) -> io::Result<(Vec<f32>, Vec<f32>)> {
    let (rows, chunk, kv) = (c.nodes[node].rows, c.chunk, c.kv);
    let name = format!("{}n{node}_kv{l}", c.prefix);
    let full = kv.record(d, chunk);
    let (mut k, mut v) = (vec![0.0f32; rows * d], vec![0.0f32; rows * d]);
    let mut raw = vec![0u32; full.div_ceil(4)];
    for (ci, r0) in (0..rows).step_by(chunk).enumerate() {
        let rc = chunk.min(rows - r0);
        let bytes = kv.record(d, rc);
        c.bundle.read_bytes_at(&name, ci * full, &mut bytemuck::cast_slice_mut::<u32, u8>(&mut raw)[..bytes])?;
        let rec = &raw[..bytes / 4];
        let mut put = |j: usize, r: usize, val: f32| {
            if j < d {
                k[(r0 + r) * d + j] = val;
            } else {
                v[(r0 + r) * d + j - d] = val;
            }
        };
        if kv == Kv::F32 {
            let all: &[f32] = bytemuck::cast_slice(rec);
            for j in 0..2 * d {
                for r in 0..rc {
                    put(j, r, all[j * rc + r]);
                }
            }
        } else {
            let (sc, body) = rec.split_at(2 * d);
            let sc: &[f32] = bytemuck::cast_slice(sc);
            let body: &[u8] = bytemuck::cast_slice(body);
            for j in 0..2 * d {
                for r in 0..rc {
                    let i = j * rc + r;
                    let int = if kv == Kv::I16 { bytemuck::cast_slice::<u8, i16>(body)[i] as f32 } else { body[i] as i8 as f32 };
                    put(j, r, int * sc[j]);
                }
            }
        }
    }
    Ok((k, v))
}

/// Logits `[n][classes]` for `n` already-normalised test rows `x` (`[n][features]`).
pub fn predict_raw(m: &Model, w: &mut IclWeights, c: &Cache, x: &[f32], n: usize, opts: &PredictOpts, st: &Stats) -> io::Result<Vec<f32>> {
    block_on(predict_raw_async(m, w, c, x, n, opts, st))
}

/// `predict_raw` in the form that can wait for an accelerator without blocking (see `wait`): the one place the
/// prediction pass is written.
pub async fn predict_raw_async(m: &Model, w: &mut IclWeights, c: &Cache<'_>, x: &[f32], n: usize, opts: &PredictOpts, st: &Stats) -> io::Result<Vec<f32>> {
    let mut out = Vec::with_capacity(n * c.classes);
    for xs in x.chunks(opts.rows_per_pass.max(1) * c.features) {
        let k = xs.len() / c.features;
        if let (Some(a), 1, 1) = (&opts.accel, c.nodes.len(), c.digits) {
            let t = Instant::now();
            if let Some(logits) = a.0.rows(&c.prefix, xs, c.classes).await? {
                out.extend(logits);
                st.add(&st.in_context_ns, t);
                st.card_layers.fetch_add(m.icl_layers as u64, Relaxed);
                st.rows.fetch_add(k as u64, Relaxed);
                st.passes.fetch_add(1, Relaxed);
                continue;
            }
        }
        let t = Instant::now();
        let mut tok = embed(m, c, xs, k, opts)?;
        st.add(&st.columns_ns, t);
        let t = Instant::now();
        let mut rep = represent(m, &mut tok, k);
        drop(tok);
        st.add(&st.rows_ns, t);
        let t = Instant::now();
        if c.nodes.len() == 1 {
            out.extend(in_context_async(m, w, c, 0, &mut rep, k, opts, st).await?);
        } else {
            // the tree of problems: each node's probabilities, multiplied down the tree, then back to scores
            // the way the reference does it (temperature 0.9 inside, 0.9 * ln(p + 1e-6) out)
            let mut probs = Vec::new();
            for ni in 0..c.nodes.len() {
                let mut p = in_context_async(m, w, c, ni, &mut rep.clone(), k, opts, st).await?;
                for row in p.chunks_exact_mut(c.nodes[ni].k) {
                    let top = row.iter().fold(f32::MIN, |a, v| a.max(*v));
                    row.iter_mut().for_each(|v| *v = ((*v - top) / 0.9).exp());
                    let sum: f32 = row.iter().sum();
                    row.iter_mut().for_each(|v| *v /= sum);
                }
                probs.push(p);
            }
            fn walk(c: &Cache, probs: &[Vec<f32>], ni: usize, r: usize, weight: f32, out: &mut [f32]) {
                let node = &c.nodes[ni];
                let p = &probs[ni][r * node.k..(r + 1) * node.k];
                for (j, target) in node.map.iter().enumerate() {
                    if node.leaf {
                        out[*target] += weight * p[j];
                    } else {
                        walk(c, probs, *target, r, weight * p[j], out);
                    }
                }
            }
            for r in 0..k {
                let mut row = vec![0.0f32; c.classes];
                walk(c, &probs, 0, r, 1.0, &mut row);
                out.extend(row.iter().map(|p| 0.9 * (p + 1e-6).ln()));
            }
        }
        st.add(&st.in_context_ns, t);
        st.rows.fetch_add(k as u64, Relaxed);
        st.passes.fetch_add(1, Relaxed);
    }
    Ok(out)
}

/// How concentrated the in-context attention is: for each layer, the share of training rows that carry all but
/// `drop` of the attention weight, averaged over test rows and heads (`[layers][drops]`), found by doing the
/// attention exactly and sorting the weights. A measurement tool, not part of prediction.
pub fn attention_profile(m: &Model, c: &Cache, h: &mut [f32], n: usize, drops: &[f32]) -> io::Result<Vec<Vec<f64>>> {
    assert!(m.icl.len() == m.icl_layers, "needs the in-context weights in memory");
    let (d, heads) = (m.d, m.icl_heads);
    let hd = d / heads;
    let rows = c.nodes[0].rows;
    let scales = c.bundle.f32s(&format!("{}n0_scale", c.prefix))?;
    let mut out = Vec::new();
    for (l, b) in m.icl.iter().enumerate() {
        let (k, v) = layer_rows(c, 0, l, d)?;
        let scale = &scales[l * d..(l + 1) * d];
        let per_row: Vec<Vec<f64>> = h
            .par_chunks_mut(d)
            .map(|x| {
                let mut ws = Ws::new(d, heads, b.ff, 1);
                let mut q = vec![0.0f32; d];
                query(pulp::Scalar::new(), b.query_w(), scale, x, &mut ws.t, &mut ws.h64, &mut ws.mm, &mut q);
                let mut att = vec![0.0f32; d];
                let mut need = vec![0.0f64; drops.len()];
                let mut wts = vec![0.0f32; rows];
                for hh in 0..heads {
                    let qh = &q[hh * hd..(hh + 1) * hd];
                    for r in 0..rows {
                        wts[r] = dot(pulp::Scalar::new(), qh, &k[r * d + hh * hd..r * d + (hh + 1) * hd]);
                    }
                    let top = wts.iter().fold(f32::MIN, |a, s| a.max(*s));
                    let mut sum = 0.0f64;
                    for w in wts.iter_mut() {
                        *w = (*w - top).exp();
                        sum += *w as f64;
                    }
                    for r in 0..rows {
                        let w = wts[r] / sum as f32;
                        for j in 0..hd {
                            att[hh * hd + j] += w * v[r * d + hh * hd + j];
                        }
                    }
                    let mut sorted = wts.clone();
                    sorted.sort_by(|a, b| b.total_cmp(a));
                    for (di, drop) in drops.iter().enumerate() {
                        let (mut acc, mut count) = (0.0f64, 0usize);
                        while count < rows && acc < (1.0 - *drop as f64) * sum {
                            acc += sorted[count] as f64;
                            count += 1;
                        }
                        need[di] += count as f64 / rows as f64 / heads as f64;
                    }
                }
                linear(pulp::Scalar::new(), &b.out_w, &b.out_b, &att, &mut ws.t);
                for i in 0..d {
                    x[i] += ws.t[i];
                }
                feed_forward(pulp::Scalar::new(), b.ff1_w(), b.ff2_w(), x, &mut ws.t, &mut ws.h);
                need
            })
            .collect();
        out.push((0..drops.len()).map(|di| per_row.iter().map(|r| r[di]).sum::<f64>() / n as f64).collect());
    }
    Ok(out)
}
