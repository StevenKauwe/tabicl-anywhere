//! Network pieces shared by fitting and prediction, each generic over the SIMD level.
//!
//! Keys and values are always held "dimension-major": for a set of `r` keys the layout is `[d][r]`, one long row
//! per coordinate, heads following one another. Scores for a query are then a weighted sum of long rows and the
//! attended value is a set of long dot products, so the inner loops run along `r` whatever the head width.

use crate::kern::*;
use crate::model::{Block, Model};
use pulp::Simd;
use std::sync::OnceLock;

pub fn arch() -> pulp::Arch {
    static ARCH: OnceLock<pulp::Arch> = OnceLock::new();
    *ARCH.get_or_init(pulp::Arch::new)
}

/// Run a block of code compiled for the CPU's best instruction set. Captured variables are listed with their
/// types (use the lifetime `'a` for borrows); inside, `$simd` is the token the kernels take.
#[macro_export]
macro_rules! with_simd {
    (|$simd:ident| [$($v:ident : $t:ty),* $(,)?] -> $out:ty $body:block) => {{
        struct Job<'a> { $($v: $t,)* _p: core::marker::PhantomData<&'a ()> }
        impl<'a> pulp::WithSimd for Job<'a> {
            type Output = $out;
            #[inline(always)]
            fn with_simd<S: pulp::Simd>(self, $simd: S) -> $out {
                let Job { $($v,)* .. } = self;
                $body
            }
        }
        $crate::nn::arch().dispatch(Job { $($v,)* _p: core::marker::PhantomData })
    }};
}

/// The weights that turn a token into its attention query.
#[derive(Clone, Copy)]
pub struct QueryW<'a> {
    pub heads: usize,
    pub n1w: &'a [f32],
    pub n1b: &'a [f32],
    pub qw: &'a [f32],
    pub qb: &'a [f32],
    /// Query-aware scalable softmax: `(l0 weight, l0 bias, l2 weight, l2 bias)`.
    pub ss: Option<(&'a [f32], &'a [f32], &'a [f32], &'a [f32])>,
}

#[derive(Clone, Copy)]
pub struct OutW<'a> {
    pub w: &'a [f32],
    pub b: &'a [f32],
}

#[derive(Clone, Copy)]
pub struct Ff1W<'a> {
    pub n2w: &'a [f32],
    pub n2b: &'a [f32],
    pub l1w: &'a [f32],
    pub l1b: &'a [f32],
}

#[derive(Clone, Copy)]
pub struct Ff2W<'a> {
    pub l2w: &'a [f32],
    pub l2b: &'a [f32],
}

impl Block {
    pub fn query_w(&self) -> QueryW<'_> {
        let d = self.d;
        QueryW { heads: self.heads, n1w: &self.n1w, n1b: &self.n1b, qw: &self.in_w[..d * d], qb: &self.in_b[..d], ss: self.ss.as_ref().map(|s| (&s.q0w[..], &s.q0b[..], &s.q2w[..], &s.q2b[..])) }
    }
    pub fn out_w(&self) -> OutW<'_> {
        OutW { w: &self.out_w, b: &self.out_b }
    }
    pub fn ff1_w(&self) -> Ff1W<'_> {
        Ff1W { n2w: &self.n2w, n2b: &self.n2b, l1w: &self.l1w, l1b: &self.l1b }
    }
    pub fn ff2_w(&self) -> Ff2W<'_> {
        Ff2W { l2w: &self.l2w, l2b: &self.l2b }
    }

    /// What every query of this block is multiplied by when there are `n` keys: the scalable softmax's
    /// length-dependent part (if the block has one) times the usual `1 / sqrt(head width)`.
    pub fn query_scale(&self, n: usize) -> Vec<f32> {
        let inv = 1.0 / (self.hd() as f32).sqrt();
        let Some(ss) = &self.ss else { return vec![inv; self.d] };
        let logn = (n.max(1) as f32).ln();
        let mut hid: Vec<f32> = ss.b0w.iter().zip(&ss.b0b).map(|(w, b)| w * logn + b).collect();
        gelu(&mut hid);
        let mut out = vec![0.0f32; self.d];
        linear(pulp::Scalar::new(), &ss.b2w, &ss.b2b, &hid, &mut out);
        out.iter_mut().for_each(|v| *v *= inv);
        out
    }
}

/// Scratch for working on one token.
pub struct Ws {
    pub t: Vec<f32>,
    pub q: Vec<f32>,
    pub h: Vec<f32>,
    pub h64: Vec<f32>,
    pub mm: Vec<f32>,
    pub sc: Vec<f32>,
    pub tmp: Vec<f32>,
    pub soft: Soft,
}

impl Ws {
    pub fn new(d: usize, heads: usize, ff: usize, max_keys: usize) -> Ws {
        Ws { t: vec![0.0; d], q: vec![0.0; d], h: vec![0.0; ff], h64: vec![0.0; 64], mm: vec![0.0; d / heads], sc: vec![0.0; max_keys], tmp: vec![0.0; d / heads], soft: Soft::new(heads, d) }
    }
}

/// `q = scale * (Wq norm1(x) + bq)`, with the query-aware modulation applied if the block has it.
#[inline(always)]
pub fn query<S: Simd>(simd: S, w: QueryW, scale: &[f32], x: &[f32], t: &mut [f32], h64: &mut [f32], mm: &mut [f32], q: &mut [f32]) {
    let d = x.len();
    let hd = d / w.heads;
    layer_norm(simd, x, w.n1w, w.n1b, t);
    linear(simd, w.qw, w.qb, t, q);
    match w.ss {
        None => {
            for i in 0..d {
                q[i] *= scale[i];
            }
        }
        Some((q0w, q0b, q2w, q2b)) => {
            for (qh, sh) in q.chunks_exact_mut(hd).zip(scale.chunks_exact(hd)) {
                linear(simd, q0w, q0b, qh, h64);
                gelu(h64);
                linear(simd, q2w, q2b, h64, mm);
                for i in 0..hd {
                    qh[i] *= sh[i] * (1.0 + tanh1(mm[i]));
                }
            }
        }
    }
}

/// `x += l2(gelu(l1(norm2 x)))`.
#[inline(always)]
pub fn feed_forward<S: Simd>(simd: S, a: Ff1W, b: Ff2W, x: &mut [f32], t: &mut [f32], h: &mut [f32]) {
    layer_norm(simd, x, a.n2w, a.n2b, t);
    linear(simd, a.l1w, a.l1b, t, h);
    gelu(h);
    linear(simd, b.l2w, b.l2b, h, t);
    for i in 0..x.len() {
        x[i] += t[i];
    }
}

/// A softmax-weighted average that accepts its keys in pieces: per head it keeps the largest score seen so far,
/// the sum of weights relative to it, and the weighted sum of values. Memory does not grow with the number of keys.
pub struct Soft {
    pub m: Vec<f32>,
    pub s: Vec<f32>,
    pub acc: Vec<f32>,
}

impl Soft {
    pub fn new(heads: usize, d: usize) -> Soft {
        Soft { m: vec![f32::NEG_INFINITY; heads], s: vec![0.0; heads], acc: vec![0.0; d] }
    }

    pub fn reset(&mut self) {
        self.m.fill(f32::NEG_INFINITY);
        self.s.fill(0.0);
        self.acc.fill(0.0);
    }

    /// Take in `r` more keys and values, laid out `[d][r]`. `sc` needs room for `r` floats, `tmp` for a head.
    #[inline(always)]
    pub fn update<S: Simd>(&mut self, simd: S, q: &[f32], kt: &[f32], vt: &[f32], r: usize, sc: &mut [f32], tmp: &mut [f32]) {
        self.update_scaled(simd, q, kt, vt, None, r, sc, tmp)
    }

    /// As `update`, for values stored as integers: `vs[j]` is what coordinate `j` of the values must be multiplied
    /// by. (Keys stored that way need nothing here: the caller multiplies the query by the keys' scales instead.)
    #[inline(always)]
    pub fn update_scaled<S: Simd>(&mut self, simd: S, q: &[f32], kt: &[f32], vt: &[f32], vs: Option<&[f32]>, r: usize, sc: &mut [f32], tmp: &mut [f32]) {
        let heads = self.m.len();
        let hd = self.acc.len() / heads;
        let sc = &mut sc[..r];
        let tmp = &mut tmp[..hd];
        for h in 0..heads {
            let span = h * hd * r..(h + 1) * hd * r;
            sc.fill(0.0);
            weighted_sum(simd, &q[h * hd..(h + 1) * hd], &kt[span.clone()], sc);
            let top = max(simd, sc);
            let acc = &mut self.acc[h * hd..(h + 1) * hd];
            if top > self.m[h] {
                if self.s[h] > 0.0 {
                    let f = exp1(self.m[h] - top);
                    self.s[h] *= f;
                    scale(simd, f, acc);
                }
                self.m[h] = top;
            }
            self.s[h] += exp_shift_sum(sc, self.m[h]);
            matvec(simd, &vt[span], sc, tmp);
            match vs {
                None => {
                    for i in 0..hd {
                        acc[i] += tmp[i];
                    }
                }
                Some(vs) => {
                    for i in 0..hd {
                        acc[i] += tmp[i] * vs[h * hd + i];
                    }
                }
            }
        }
    }

    /// As `update`, for keys and values stored as integers. `q` must already be multiplied by the keys'
    /// per-coordinate scales; `vs` holds the values' scales.
    #[inline(always)]
    pub fn update_int<S: Simd, T: Copy + Into<f32>>(&mut self, simd: S, q: &[f32], kt: &[T], vt: &[T], vs: &[f32], r: usize, sc: &mut [f32], tmp: &mut [f32]) {
        let heads = self.m.len();
        let hd = self.acc.len() / heads;
        let sc = &mut sc[..r];
        let tmp = &mut tmp[..hd];
        for h in 0..heads {
            let span = h * hd * r..(h + 1) * hd * r;
            sc.fill(0.0);
            weighted_sum_int(&q[h * hd..(h + 1) * hd], &kt[span.clone()], sc);
            let top = max(simd, sc);
            let acc = &mut self.acc[h * hd..(h + 1) * hd];
            if top > self.m[h] {
                if self.s[h] > 0.0 {
                    let f = exp1(self.m[h] - top);
                    self.s[h] *= f;
                    scale(simd, f, acc);
                }
                self.m[h] = top;
            }
            self.s[h] += exp_shift_sum(sc, self.m[h]);
            matvec_int(&vt[span], sc, tmp);
            for i in 0..hd {
                acc[i] += tmp[i] * vs[h * hd + i];
            }
        }
    }

    /// Fold in another running average over a different set of keys (same query), as if its keys had come here.
    pub fn merge(&mut self, o: &Soft) {
        let hd = self.acc.len() / self.m.len();
        for h in 0..self.m.len() {
            if o.s[h] <= 0.0 {
                continue;
            }
            let top = self.m[h].max(o.m[h]);
            let fa = if self.s[h] > 0.0 { exp1(self.m[h] - top) } else { 0.0 };
            let fb = exp1(o.m[h] - top);
            self.s[h] = self.s[h] * fa + o.s[h] * fb;
            for i in h * hd..(h + 1) * hd {
                self.acc[i] = self.acc[i] * fa + o.acc[i] * fb;
            }
            self.m[h] = top;
        }
    }

    /// Divide through by the total weight; `acc` is then the attention output before the output projection.
    #[inline(always)]
    pub fn finish(&mut self) {
        let hd = self.acc.len() / self.m.len();
        for (acc, s) in self.acc.chunks_exact_mut(hd).zip(&self.s) {
            let inv = 1.0 / s;
            acc.iter_mut().for_each(|v| *v *= inv);
        }
    }
}

/// One whole block for one token against a fixed set of `n` keys and values laid out `[d][n]`.
#[inline(always)]
pub fn cached_block<S: Simd>(simd: S, b: &Block, scale: &[f32], kt: &[f32], vt: &[f32], n: usize, x: &mut [f32], ws: &mut Ws) {
    query(simd, b.query_w(), scale, x, &mut ws.t, &mut ws.h64, &mut ws.mm, &mut ws.q);
    ws.soft.reset();
    ws.soft.update(simd, &ws.q, kt, vt, n, &mut ws.sc, &mut ws.tmp);
    ws.soft.finish();
    linear(simd, &b.out_w, &b.out_b, &ws.soft.acc, &mut ws.t);
    for i in 0..x.len() {
        x[i] += ws.t[i];
    }
    feed_forward(simd, b.ff1_w(), b.ff2_w(), x, &mut ws.t, &mut ws.h);
}

/// Rotary position tables for the row stage: `cos` and `sin` of `position * frequency`, `[positions][half head]`.
pub struct Rope {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}

impl Rope {
    pub fn new(freqs: &[f32], positions: usize) -> Rope {
        let mut r = Rope { cos: Vec::new(), sin: Vec::new() };
        for p in 0..positions {
            for f in freqs {
                let a = p as f32 * f;
                r.cos.push(a.cos());
                r.sin.push(a.sin());
            }
        }
        r
    }
}

pub struct RowWs {
    tn: Vec<f32>,
    qkv: Vec<f32>,
    q: Vec<f32>,
    kt: Vec<f32>,
    vt: Vec<f32>,
    new: Vec<f32>,
    sc: Vec<f32>,
    att: Vec<f32>,
    tmp: Vec<f32>,
    h: Vec<f32>,
}

impl RowWs {
    pub fn new(m: &Model, tokens: usize) -> RowWs {
        let d = m.e;
        RowWs {
            tn: vec![0.0; d],
            qkv: vec![0.0; 3 * d],
            q: vec![0.0; tokens * d],
            kt: vec![0.0; tokens * d],
            vt: vec![0.0; tokens * d],
            new: vec![0.0; tokens * d],
            sc: vec![0.0; tokens],
            att: vec![0.0; d],
            tmp: vec![0.0; d / m.row_heads],
            h: vec![0.0; m.row[0].ff],
        }
    }
}

/// The row stage for one row: `tok` holds the class tokens followed by the row's column embeddings (`[tokens][e]`);
/// `out` receives the class tokens after the last block, normalised and laid end to end (`cls * e` floats).
#[inline(always)]
pub fn row_forward<S: Simd>(simd: S, m: &Model, rope: &Rope, tok: &mut [f32], tokens: usize, ws: &mut RowWs, out: &mut [f32]) {
    let d = m.e;
    let heads = m.row_heads;
    let hd = d / heads;
    let half = hd / 2;
    let inv = 1.0 / (hd as f32).sqrt();
    let last = m.row.len() - 1;
    for (bi, b) in m.row.iter().enumerate() {
        let nq = if bi == last { m.cls } else { tokens };
        for t in 0..tokens {
            layer_norm(simd, &tok[t * d..(t + 1) * d], &b.n1w, &b.n1b, &mut ws.tn);
            if t < nq {
                linear(simd, &b.in_w, &b.in_b, &ws.tn, &mut ws.qkv);
            } else {
                linear(simd, &b.in_w[d * d..], &b.in_b[d..], &ws.tn, &mut ws.qkv[d..]);
            }
            let (cos, sin) = (&rope.cos[t * half..(t + 1) * half], &rope.sin[t * half..(t + 1) * half]);
            for h in 0..heads {
                if t < nq {
                    let src = &ws.qkv[h * hd..(h + 1) * hd];
                    let dst = &mut ws.q[t * d + h * hd..t * d + (h + 1) * hd];
                    for i in 0..half {
                        let (a, c) = (src[i], src[i + half]);
                        dst[i] = (a * cos[i] - c * sin[i]) * inv;
                        dst[i + half] = (c * cos[i] + a * sin[i]) * inv;
                    }
                }
                let src = &ws.qkv[d + h * hd..d + (h + 1) * hd];
                for i in 0..half {
                    let (a, c) = (src[i], src[i + half]);
                    ws.kt[(h * hd + i) * tokens + t] = a * cos[i] - c * sin[i];
                    ws.kt[(h * hd + i + half) * tokens + t] = c * cos[i] + a * sin[i];
                }
            }
            for j in 0..d {
                ws.vt[j * tokens + t] = ws.qkv[2 * d + j];
            }
        }
        for t in 0..nq {
            for h in 0..heads {
                let span = h * hd * tokens..(h + 1) * hd * tokens;
                ws.sc.fill(0.0);
                weighted_sum(simd, &ws.q[t * d + h * hd..t * d + (h + 1) * hd], &ws.kt[span.clone()], &mut ws.sc);
                let top = max(simd, &ws.sc);
                let s = 1.0 / exp_shift_sum(&mut ws.sc, top);
                matvec(simd, &ws.vt[span], &ws.sc, &mut ws.tmp);
                for i in 0..hd {
                    ws.att[h * hd + i] = ws.tmp[i] * s;
                }
            }
            let x = &mut ws.new[t * d..(t + 1) * d];
            linear(simd, &b.out_w, &b.out_b, &ws.att, x);
            for i in 0..d {
                x[i] += tok[t * d + i];
            }
            feed_forward(simd, b.ff1_w(), b.ff2_w(), x, &mut ws.tn, &mut ws.h);
        }
        tok[..nq * d].copy_from_slice(&ws.new[..nq * d]);
    }
    for c in 0..m.cls {
        layer_norm(simd, &tok[c * d..(c + 1) * d], &m.out_lnw, &m.out_lnb, &mut out[c * d..(c + 1) * d]);
    }
}

/// The three values the column stage sees for feature group `g` of a row of `h` features: the features one, two
/// and four places to the right, wrapping round.
#[inline(always)]
pub fn group_values(row: &[f32], g: usize, size: usize, out: &mut [f32]) {
    let h = row.len();
    for i in 0..size {
        out[i] = row[(g + (1 << i)) % h];
    }
}

/// How fast this machine runs the inner loop (a 512 x 512 matrix-vector product), in multiply-adds per second,
/// measured for about `ms` milliseconds on one thread. Needs no weights: a caller can use it to estimate how long
/// a fit will take before starting one.
pub fn calibrate(ms: f64) -> f64 {
    let (o, i) = (512usize, 512usize);
    let w: Vec<f32> = (0..o * i).map(|k| (k % 13) as f32 * 0.01).collect();
    let x: Vec<f32> = (0..i).map(|k| (k % 7) as f32 * 0.1).collect();
    let mut y = vec![0.0f32; o];
    let (w, x, y) = (&w[..], &x[..], &mut y[..]);
    let t = web_time::Instant::now();
    let mut n = 0u64;
    while t.elapsed().as_secs_f64() * 1e3 < ms {
        for _ in 0..20 {
            with_simd!(|simd| [w: &'a [f32], x: &'a [f32], y: &'a mut [f32]] -> () { crate::kern::matvec(simd, w, x, y) });
        }
        n += 20;
    }
    std::hint::black_box(&y);
    (n * (o * i) as u64) as f64 / t.elapsed().as_secs_f64().max(1e-9)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, seed: usize) -> Vec<f32> {
        (0..n).map(|i| (((i + seed) * 7919 % 2003) as f32 - 1001.0) * 2e-3).collect()
    }

    /// Plain softmax attention for one query, per head, keys and values laid out `[d][r]`.
    fn attend(q: &[f32], kt: &[f32], vt: &[f32], r: usize, heads: usize) -> Vec<f32> {
        let d = q.len();
        let hd = d / heads;
        let mut out = vec![0.0f32; d];
        for h in 0..heads {
            let sc: Vec<f64> = (0..r).map(|j| (0..hd).map(|i| q[h * hd + i] as f64 * kt[(h * hd + i) * r + j] as f64).sum()).collect();
            let top = sc.iter().copied().fold(f64::MIN, f64::max);
            let w: Vec<f64> = sc.iter().map(|s| (s - top).exp()).collect();
            let sum: f64 = w.iter().sum();
            for i in 0..hd {
                out[h * hd + i] = ((0..r).map(|j| w[j] * vt[(h * hd + i) * r + j] as f64).sum::<f64>() / sum) as f32;
            }
        }
        out
    }

    /// The column of `kt` / `vt` (laid out `[d][r]`) for keys `a..b`, as its own `[d][b - a]` block.
    fn cols(t: &[f32], d: usize, r: usize, a: usize, b: usize) -> Vec<f32> {
        (0..d).flat_map(|i| t[i * r + a..i * r + b].iter().copied()).collect()
    }

    #[test]
    fn streaming_softmax_matches_plain_attention_in_any_pieces() {
        let (d, heads, r) = (16, 4, 37);
        let (q, kt, vt) = (data(d, 1), data(d * r, 2), data(d * r, 3));
        let want = attend(&q, &kt, &vt, r, heads);
        let (mut sc, mut tmp) = (vec![0.0f32; r], vec![0.0f32; d / heads]);
        for pieces in [vec![37], vec![1, 36], vec![10, 10, 10, 7], vec![36, 1]] {
            let mut s = Soft::new(heads, d);
            let mut at = 0;
            for p in pieces {
                let (k, v) = (cols(&kt, d, r, at, at + p), cols(&vt, d, r, at, at + p));
                s.update(pulp::Scalar::new(), &q, &k, &v, p, &mut sc, &mut tmp);
                at += p;
            }
            s.finish();
            for (a, b) in s.acc.iter().zip(&want) {
                assert!((a - b).abs() < 1e-5, "{a} {b}");
            }
        }
        // two halves seen by separate accumulators, merged: the same again (and merging an empty one is a no-op)
        let (mut a, mut b) = (Soft::new(heads, d), Soft::new(heads, d));
        a.update(pulp::Scalar::new(), &q, &cols(&kt, d, r, 0, 20), &cols(&vt, d, r, 0, 20), 20, &mut sc, &mut tmp);
        b.update(pulp::Scalar::new(), &q, &cols(&kt, d, r, 20, r), &cols(&vt, d, r, 20, r), r - 20, &mut sc, &mut tmp);
        a.merge(&Soft::new(heads, d));
        let mut empty = Soft::new(heads, d);
        empty.merge(&b);
        empty.merge(&a);
        empty.finish();
        for (x, y) in empty.acc.iter().zip(&want) {
            assert!((x - y).abs() < 1e-5);
        }
        a.reset();
        assert!(a.m.iter().all(|m| *m == f32::NEG_INFINITY) && a.s.iter().all(|s| *s == 0.0) && a.acc.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn integer_keys_and_values() {
        let (d, heads, r) = (8, 2, 13);
        let q = data(d, 4);
        let ki: Vec<i16> = (0..d * r).map(|i| ((i * 31) % 601) as i16 - 300).collect();
        let vi: Vec<i16> = (0..d * r).map(|i| ((i * 17) % 501) as i16 - 250).collect();
        let (ks, vs): (Vec<f32>, Vec<f32>) = ((0..d).map(|j| 1e-3 * (j + 1) as f32).collect(), (0..d).map(|j| 2e-3 * (j + 2) as f32).collect());
        // what the integers stand for
        let kt: Vec<f32> = (0..d * r).map(|i| ki[i] as f32 * ks[i / r]).collect();
        let vt: Vec<f32> = (0..d * r).map(|i| vi[i] as f32 * vs[i / r]).collect();
        let want = attend(&q, &kt, &vt, r, heads);
        let qs: Vec<f32> = q.iter().zip(&ks).map(|(a, b)| a * b).collect();
        let (mut sc, mut tmp) = (vec![0.0f32; r], vec![0.0f32; d / heads]);
        let mut s = Soft::new(heads, d);
        s.update_int(pulp::Scalar::new(), &qs, &ki, &vi, &vs, r, &mut sc, &mut tmp);
        s.finish();
        let vf: Vec<f32> = vi.iter().map(|v| *v as f32).collect();
        let mut s2 = Soft::new(heads, d);
        s2.update_scaled(pulp::Scalar::new(), &q, &kt, &vf, Some(&vs), r, &mut sc, &mut tmp);
        s2.finish();
        for i in 0..d {
            assert!((s.acc[i] - want[i]).abs() < 1e-4 && (s2.acc[i] - want[i]).abs() < 1e-4, "{i}: {} {} {}", s.acc[i], s2.acc[i], want[i]);
        }
    }

    #[test]
    fn small_pieces() {
        let mut out = [0.0f32; 3];
        group_values(&[1.0, 2.0, 3.0, 4.0, 5.0], 3, 3, &mut out);
        assert_eq!(out, [5.0, 1.0, 3.0], "one, two and four places to the right, wrapping round");
        let r = Rope::new(&[1.0, 0.5], 3);
        assert_eq!(r.cos.len(), 6);
        assert_eq!((r.cos[0], r.sin[0]), (1.0, 0.0), "position 0 does not rotate");
        assert!((r.sin[2 * 2 + 1] - 1.0f32.sin()).abs() < 1e-7, "position 2, frequency 0.5");
        assert!(calibrate(5.0) > 1e6, "even a slow machine does a million multiply-adds a second");
    }
}
