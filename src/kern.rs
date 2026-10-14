//! The few inner loops everything else is built from.
//!
//! Every kernel is generic over `pulp::Simd`, so one source compiles to AVX2/FMA, AVX-512, NEON or plain scalar,
//! and the choice is made once at run time by `pulp::Arch`. Callers enter through a `WithSimd` job (see `nn.rs`),
//! which lets all of these inline into one function built for the detected CPU.
//!
//! Matrices are row-major `[rows][cols]` in flat `&[f32]`. Weight matrices are `[out][in]`, as PyTorch stores them.

use pulp::Simd;

/// Sum of `a[i] * b[i]`.
#[inline(always)]
pub fn dot<S: Simd>(simd: S, a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let (ah, at) = S::as_simd_f32s(a);
    let (bh, bt) = S::as_simd_f32s(b);
    let z = simd.splat_f32s(0.0);
    let (mut s0, mut s1, mut s2, mut s3) = (z, z, z, z);
    let (mut ac, mut bc) = (ah.chunks_exact(4), bh.chunks_exact(4));
    for (x, y) in (&mut ac).zip(&mut bc) {
        s0 = simd.mul_add_e_f32s(x[0], y[0], s0);
        s1 = simd.mul_add_e_f32s(x[1], y[1], s1);
        s2 = simd.mul_add_e_f32s(x[2], y[2], s2);
        s3 = simd.mul_add_e_f32s(x[3], y[3], s3);
    }
    for (x, y) in ac.remainder().iter().zip(bc.remainder()) {
        s0 = simd.mul_add_e_f32s(*x, *y, s0);
    }
    let mut s = simd.reduce_sum_f32s(simd.add_f32s(simd.add_f32s(s0, s1), simd.add_f32s(s2, s3)));
    for (x, y) in at.iter().zip(bt) {
        s += x * y;
    }
    s
}

/// `y += alpha * x`.
#[inline(always)]
pub fn axpy<S: Simd>(simd: S, alpha: f32, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    let a = simd.splat_f32s(alpha);
    let (xh, xt) = S::as_simd_f32s(x);
    let (yh, yt) = S::as_mut_simd_f32s(y);
    for (yv, xv) in yh.iter_mut().zip(xh) {
        *yv = simd.mul_add_e_f32s(a, *xv, *yv);
    }
    for (yv, xv) in yt.iter_mut().zip(xt) {
        *yv += alpha * xv;
    }
}

/// `y *= alpha`.
#[inline(always)]
pub fn scale<S: Simd>(simd: S, alpha: f32, y: &mut [f32]) {
    let a = simd.splat_f32s(alpha);
    let (yh, yt) = S::as_mut_simd_f32s(y);
    for yv in yh.iter_mut() {
        *yv = simd.mul_f32s(a, *yv);
    }
    for yv in yt.iter_mut() {
        *yv *= alpha;
    }
}

/// `y[o] = w[o] . x`, for a matrix `w` of `y.len()` rows by `x.len()` columns. Eight rows share each load of `x`.
#[inline(always)]
pub fn matvec<S: Simd>(simd: S, w: &[f32], x: &[f32], y: &mut [f32]) {
    let n = x.len();
    debug_assert_eq!(w.len(), n * y.len());
    if n == 0 {
        y.fill(0.0);
        return;
    }
    let (xh, xt) = S::as_simd_f32s(x);
    let nv = xh.len();
    let z = simd.splat_f32s(0.0);
    let mut rows = w.chunks_exact(8 * n);
    let mut outs = y.chunks_exact_mut(8);
    for (blk, out) in (&mut rows).zip(&mut outs) {
        let (r0, rest) = blk.split_at(n);
        let (r1, rest) = rest.split_at(n);
        let (r2, rest) = rest.split_at(n);
        let (r3, rest) = rest.split_at(n);
        let (r4, rest) = rest.split_at(n);
        let (r5, rest) = rest.split_at(n);
        let (r6, r7) = rest.split_at(n);
        let (h0, t0) = S::as_simd_f32s(r0);
        let (h1, t1) = S::as_simd_f32s(r1);
        let (h2, t2) = S::as_simd_f32s(r2);
        let (h3, t3) = S::as_simd_f32s(r3);
        let (h4, t4) = S::as_simd_f32s(r4);
        let (h5, t5) = S::as_simd_f32s(r5);
        let (h6, t6) = S::as_simd_f32s(r6);
        let (h7, t7) = S::as_simd_f32s(r7);
        let (h0, h1, h2, h3) = (&h0[..nv], &h1[..nv], &h2[..nv], &h3[..nv]);
        let (h4, h5, h6, h7) = (&h4[..nv], &h5[..nv], &h6[..nv], &h7[..nv]);
        let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7) = (z, z, z, z, z, z, z, z);
        for i in 0..nv {
            let xv = xh[i];
            a0 = simd.mul_add_e_f32s(h0[i], xv, a0);
            a1 = simd.mul_add_e_f32s(h1[i], xv, a1);
            a2 = simd.mul_add_e_f32s(h2[i], xv, a2);
            a3 = simd.mul_add_e_f32s(h3[i], xv, a3);
            a4 = simd.mul_add_e_f32s(h4[i], xv, a4);
            a5 = simd.mul_add_e_f32s(h5[i], xv, a5);
            a6 = simd.mul_add_e_f32s(h6[i], xv, a6);
            a7 = simd.mul_add_e_f32s(h7[i], xv, a7);
        }
        let tail = |t: &[f32]| t.iter().zip(xt).map(|(a, b)| a * b).sum::<f32>();
        out[0] = simd.reduce_sum_f32s(a0) + tail(t0);
        out[1] = simd.reduce_sum_f32s(a1) + tail(t1);
        out[2] = simd.reduce_sum_f32s(a2) + tail(t2);
        out[3] = simd.reduce_sum_f32s(a3) + tail(t3);
        out[4] = simd.reduce_sum_f32s(a4) + tail(t4);
        out[5] = simd.reduce_sum_f32s(a5) + tail(t5);
        out[6] = simd.reduce_sum_f32s(a6) + tail(t6);
        out[7] = simd.reduce_sum_f32s(a7) + tail(t7);
    }
    for (row, out) in rows.remainder().chunks_exact(n.max(1)).zip(outs.into_remainder()) {
        *out = dot(simd, row, x);
    }
}

/// `y = w x + b`.
#[inline(always)]
pub fn linear<S: Simd>(simd: S, w: &[f32], b: &[f32], x: &[f32], y: &mut [f32]) {
    matvec(simd, w, x, y);
    for (yv, bv) in y.iter_mut().zip(b) {
        *yv += bv;
    }
}

/// `acc += sum_r wts[r] * rows[r]`, for `rows` of `wts.len()` rows by `acc.len()` columns.
#[inline(always)]
pub fn weighted_sum<S: Simd>(simd: S, wts: &[f32], rows: &[f32], acc: &mut [f32]) {
    let n = acc.len();
    debug_assert_eq!(rows.len(), n * wts.len());
    if n == 0 {
        return;
    }
    let (ah, at) = S::as_mut_simd_f32s(acc);
    let nv = ah.len();
    let mut rc = rows.chunks_exact(4 * n);
    let mut wc = wts.chunks_exact(4);
    for (blk, w) in (&mut rc).zip(&mut wc) {
        let (r0, rest) = blk.split_at(n);
        let (r1, rest) = rest.split_at(n);
        let (r2, r3) = rest.split_at(n);
        let (h0, t0) = S::as_simd_f32s(r0);
        let (h1, t1) = S::as_simd_f32s(r1);
        let (h2, t2) = S::as_simd_f32s(r2);
        let (h3, t3) = S::as_simd_f32s(r3);
        let (h0, h1, h2, h3) = (&h0[..nv], &h1[..nv], &h2[..nv], &h3[..nv]);
        let (w0, w1, w2, w3) = (simd.splat_f32s(w[0]), simd.splat_f32s(w[1]), simd.splat_f32s(w[2]), simd.splat_f32s(w[3]));
        for i in 0..nv {
            let a = simd.mul_add_e_f32s(w0, h0[i], ah[i]);
            let b = simd.mul_f32s(w1, h1[i]);
            let a = simd.mul_add_e_f32s(w2, h2[i], a);
            let b = simd.mul_add_e_f32s(w3, h3[i], b);
            ah[i] = simd.add_f32s(a, b);
        }
        for i in 0..at.len() {
            at[i] += w[0] * t0[i] + w[1] * t1[i] + w[2] * t2[i] + w[3] * t3[i];
        }
    }
    for (row, w) in rc.remainder().chunks_exact(n.max(1)).zip(wc.remainder()) {
        axpy(simd, *w, row, acc);
    }
}

/// Largest value in `x` (negative infinity if empty).
#[inline(always)]
pub fn max<S: Simd>(simd: S, x: &[f32]) -> f32 {
    let (h, t) = S::as_simd_f32s(x);
    let mut m = simd.splat_f32s(f32::NEG_INFINITY);
    for v in h {
        m = simd.max_f32s(m, *v);
    }
    let mut out = simd.reduce_max_f32s(m);
    for v in t {
        out = out.max(*v);
    }
    out
}

const LOG2E: f32 = std::f32::consts::LOG2_E;
const LN2_HI: f32 = 0.693_359_4;
const LN2_LO: f32 = -2.121_944_4e-4;

/// e^x to within f32 rounding, written without branches so the compiler turns a loop of these into vector code.
#[inline(always)]
pub fn exp1(x: f32) -> f32 {
    let x = x.clamp(-87.0, 88.0);
    let n = (x * LOG2E + 0.5).floor();
    // LN2_HI has few significant bits, so n * LN2_HI is exact and no fused multiply-add is needed (which matters
    // where there is none, such as WebAssembly: a library call per element otherwise)
    let r = (x - n * LN2_HI) - n * LN2_LO;
    let mut p = 1.0 / 5040.0;
    p = p * r + 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    p * f32::from_bits(((n as i32 + 127) as u32) << 23)
}

/// `x[i] = e^(x[i] - shift)`, returning the sum.
#[inline(always)]
pub fn exp_shift_sum(x: &mut [f32], shift: f32) -> f32 {
    let mut s = [0.0f32; 8];
    let mut c = x.chunks_exact_mut(8);
    for v in &mut c {
        for i in 0..8 {
            v[i] = exp1(v[i] - shift);
            s[i] += v[i];
        }
    }
    let mut t = s.iter().sum::<f32>();
    for v in c.into_remainder() {
        *v = exp1(*v - shift);
        t += *v;
    }
    t
}

/// The error function to about 2e-7 (Abramowitz and Stegun 7.1.26, and a power series near zero), written so
/// that loops of it vectorise.
#[inline(always)]
pub fn erf1(x: f32) -> f32 {
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    let mut p = 1.061_405_4;
    p = p * t - 1.453_152_1;
    p = p * t + 1.421_413_8;
    p = p * t - 0.284_496_74;
    p = p * t + 0.254_829_6;
    let far = (1.0 - p * t * exp1(-a * a)).copysign(x);
    // near zero the formula above loses digits to cancellation; the power series is exact enough there
    let x2 = x * x;
    let mut q = -1.0 / 1320.0;
    q = q * x2 + 1.0 / 216.0;
    q = q * x2 - 1.0 / 42.0;
    q = q * x2 + 0.1;
    q = q * x2 - 1.0 / 3.0;
    q = q * x2 + 1.0;
    let near = q * x * std::f32::consts::FRAC_2_SQRT_PI;
    if a < 0.5 {
        near
    } else {
        far
    }
}

/// GELU as PyTorch defines it by default: `x * 0.5 * (1 + erf(x / sqrt 2))`.
#[inline(always)]
pub fn gelu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = 0.5 * *v * (1.0 + erf1(*v * std::f32::consts::FRAC_1_SQRT_2));
    }
}

/// tanh through one exponential.
#[inline(always)]
pub fn tanh1(x: f32) -> f32 {
    let e = exp1(2.0 * x.clamp(-10.0, 10.0));
    (e - 1.0) / (e + 1.0)
}

/// `acc += sum_j wts[j] * rows[j]` for rows stored as small integers (`rows` is `wts.len()` rows of `acc.len()`).
/// Plain loops on purpose: element-wise, so the compiler widens and multiplies in vector registers directly.
#[inline(always)]
pub fn weighted_sum_int<T: Copy + Into<f32>>(wts: &[f32], rows: &[T], acc: &mut [f32]) {
    let n = acc.len();
    debug_assert_eq!(rows.len(), n * wts.len());
    if n == 0 {
        return;
    }
    let mut rc = rows.chunks_exact(4 * n);
    let mut wc = wts.chunks_exact(4);
    for (blk, w) in (&mut rc).zip(&mut wc) {
        let (r0, rest) = blk.split_at(n);
        let (r1, rest) = rest.split_at(n);
        let (r2, r3) = rest.split_at(n);
        for i in 0..n {
            acc[i] += (w[0] * r0[i].into() + w[1] * r1[i].into()) + (w[2] * r2[i].into() + w[3] * r3[i].into());
        }
    }
    for (row, w) in rc.remainder().chunks_exact(n.max(1)).zip(wc.remainder()) {
        for i in 0..n {
            acc[i] += *w * row[i].into();
        }
    }
}

/// `y[j] = rows[j] . x` for rows stored as small integers. Sixteen running sums per row keep the additions in
/// vector registers.
#[inline(always)]
pub fn matvec_int<T: Copy + Into<f32>>(rows: &[T], x: &[f32], y: &mut [f32]) {
    let n = x.len();
    debug_assert_eq!(rows.len(), n * y.len());
    if n == 0 {
        y.fill(0.0);
        return;
    }
    for (row, out) in rows.chunks_exact(n).zip(y.iter_mut()) {
        let mut a = [0.0f32; 16];
        let mut rc = row.chunks_exact(16);
        let mut xc = x.chunks_exact(16);
        for (r, v) in (&mut rc).zip(&mut xc) {
            for l in 0..16 {
                a[l] += Into::<f32>::into(r[l]) * v[l];
            }
        }
        let mut s = 0.0f32;
        for l in 0..8 {
            s += a[l] + a[l + 8];
        }
        for (r, v) in rc.remainder().iter().zip(xc.remainder()) {
            s += Into::<f32>::into(*r) * v;
        }
        *out = s;
    }
}

/// Layer normalisation over the whole of `x` (eps 1e-5, as PyTorch), then `* w + b`.
#[inline(always)]
pub fn layer_norm<S: Simd>(simd: S, x: &[f32], w: &[f32], b: &[f32], y: &mut [f32]) {
    let n = x.len() as f32;
    let (xh, xt) = S::as_simd_f32s(x);
    let mut s = simd.splat_f32s(0.0);
    for v in xh {
        s = simd.add_f32s(s, *v);
    }
    let mean = (simd.reduce_sum_f32s(s) + xt.iter().sum::<f32>()) / n;
    let mv = simd.splat_f32s(mean);
    let mut q = simd.splat_f32s(0.0);
    for v in xh {
        let d = simd.sub_f32s(*v, mv);
        q = simd.mul_add_e_f32s(d, d, q);
    }
    let var = (simd.reduce_sum_f32s(q) + xt.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>()) / n;
    let inv = 1.0 / (var + 1e-5).sqrt();
    for i in 0..x.len() {
        y[i] = (x[i] - mean) * inv * w[i] + b[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_matches_std() {
        let mut worst = 0.0f64;
        let mut x = -80.0f32;
        while x < 80.0 {
            let (a, b) = (exp1(x) as f64, (x as f64).exp());
            worst = worst.max(((a - b) / b).abs());
            x += 0.0137;
        }
        assert!(worst < 2e-7, "worst relative error {worst}");
    }

    #[test]
    fn erf_and_tanh_are_close() {
        let mut x = -6.0f32;
        while x < 6.0 {
            assert!((erf1(x) - libm::erff(x)).abs() < 3e-7, "erf {x}");
            assert!((tanh1(x) - libm::tanhf(x)).abs() < 4e-7, "tanh {x}");
            x += 0.00171;
        }
    }

    #[test]
    fn integer_kernels() {
        let rows: Vec<i16> = (0..5 * 37).map(|i| ((i * 7919) % 2001) as i16 - 1000).collect();
        let w: Vec<f32> = (0..5).map(|i| i as f32 * 0.3 - 0.5).collect();
        let mut acc = vec![1.0f32; 37];
        weighted_sum_int(&w, &rows, &mut acc);
        for i in 0..37 {
            let want: f32 = 1.0 + (0..5).map(|j| w[j] * rows[j * 37 + i] as f32).sum::<f32>();
            assert!((acc[i] - want).abs() < 1e-2, "{i}");
        }
        let x: Vec<f32> = (0..37).map(|i| (i as f32).sin()).collect();
        let mut y = vec![0.0f32; 5];
        matvec_int(&rows, &x, &mut y);
        for j in 0..5 {
            let want: f32 = (0..37).map(|i| rows[j * 37 + i] as f32 * x[i]).sum();
            assert!((y[j] - want).abs() < 0.5, "{j}: {} {want}", y[j]);
        }
    }

    #[test]
    fn kernels_match_plain_loops() {
        let arch = pulp::Arch::new();
        let n = 67;
        let x: Vec<f32> = (0..n).map(|i| ((i * 37 % 19) as f32 - 9.0) * 0.13).collect();
        let w: Vec<f32> = (0..n * 11).map(|i| ((i * 53 % 23) as f32 - 11.0) * 0.07).collect();
        let mut y = vec![0.0f32; 11];
        arch.dispatch(|| matvec(pulp::Scalar::new(), &w, &x, &mut y));
        struct J<'a>(&'a [f32], &'a [f32], &'a mut [f32]);
        impl pulp::WithSimd for J<'_> {
            type Output = ();
            fn with_simd<S: Simd>(self, simd: S) {
                matvec(simd, self.0, self.1, self.2)
            }
        }
        let mut y2 = vec![0.0f32; 11];
        arch.dispatch(J(&w, &x, &mut y2));
        for o in 0..11 {
            let want: f32 = (0..n).map(|i| w[o * n + i] * x[i]).sum();
            assert!((y[o] - want).abs() < 1e-4 && (y2[o] - want).abs() < 1e-4, "{o}: {} {} {want}", y[o], y2[o]);
        }
    }

    fn data(n: usize, seed: usize) -> Vec<f32> {
        (0..n).map(|i| (((i + seed) * 7919 % 2003) as f32 - 1001.0) * 1e-3).collect()
    }

    /// Every SIMD kernel against a plain loop, at every length from 0 to 70 (so every split between vector
    /// body and scalar tail is met), with the scalar path and the best one this CPU offers.
    #[test]
    fn simd_kernels_at_every_length() {
        struct All;
        impl pulp::WithSimd for All {
            type Output = ();
            fn with_simd<S: Simd>(self, simd: S) {
                for n in 0..70usize {
                    let (a, b) = (data(n, 1), data(n, 2));
                    let want: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
                    assert!((dot(simd, &a, &b) - want).abs() < 1e-4, "dot {n}");
                    let mut y = b.clone();
                    axpy(simd, 0.5, &a, &mut y);
                    assert!(y.iter().zip(a.iter().zip(&b)).all(|(v, (x, z))| (v - (z + 0.5 * x)).abs() < 1e-6), "axpy {n}");
                    scale(simd, -2.0, &mut y);
                    assert!(y.iter().zip(a.iter().zip(&b)).all(|(v, (x, z))| (v + 2.0 * (z + 0.5 * x)).abs() < 1e-5), "scale {n}");
                    let m = max(simd, &a);
                    assert_eq!(m, a.iter().copied().fold(f32::NEG_INFINITY, f32::max), "max {n}");
                    for rows in [0usize, 1, 3, 4, 5, 8, 9, 17] {
                        let w = data(rows * n, 3);
                        let mut out = vec![0.0f32; rows];
                        matvec(simd, &w, &a, &mut out);
                        let bias = data(rows, 4);
                        let mut lin = vec![0.0f32; rows];
                        linear(simd, &w, &bias, &a, &mut lin);
                        for r in 0..rows {
                            let want: f32 = (0..n).map(|i| w[r * n + i] * a[i]).sum();
                            assert!((out[r] - want).abs() < 1e-4, "matvec {rows}x{n} row {r}");
                            assert!((lin[r] - want - bias[r]).abs() < 1e-4);
                        }
                        let wts = data(rows, 5);
                        let mut acc = b.clone();
                        weighted_sum(simd, &wts, &w, &mut acc);
                        for i in 0..n {
                            let want = b[i] + (0..rows).map(|r| wts[r] * w[r * n + i]).sum::<f32>();
                            assert!((acc[i] - want).abs() < 1e-4, "weighted_sum {rows}x{n} at {i}");
                        }
                    }
                    if n > 0 {
                        let (w, bb) = (data(n, 6), data(n, 7));
                        let mut y = vec![0.0f32; n];
                        layer_norm(simd, &a, &w, &bb, &mut y);
                        let mean = a.iter().sum::<f32>() / n as f32;
                        let var = a.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
                        for i in 0..n {
                            let want = (a[i] - mean) / (var + 1e-5).sqrt() * w[i] + bb[i];
                            assert!((y[i] - want).abs() < 1e-4, "layer_norm {n} at {i}: {} {want}", y[i]);
                        }
                    }
                }
            }
        }
        use pulp::WithSimd;
        pulp::Arch::new().dispatch(All);
        All.with_simd(pulp::Scalar::new());
    }

    #[test]
    fn softmax_pieces() {
        for n in [0usize, 1, 7, 8, 9, 23] {
            let mut x = data(n, 9);
            let orig = x.clone();
            let s = exp_shift_sum(&mut x, 0.25);
            for (v, o) in x.iter().zip(&orig) {
                assert!((v - ((o - 0.25) as f64).exp() as f32).abs() < 1e-6);
            }
            assert!((s - x.iter().sum::<f32>()).abs() < 1e-5);
        }
        assert_eq!(exp1(-1000.0), exp1(-87.0), "clamped below: no denormal or zero surprises");
        assert!(exp1(1000.0).is_finite(), "clamped above: never infinity");
        let mut g = vec![-10.0f32, -1.0, 0.0, 0.1, 1.0, 10.0];
        gelu(&mut g);
        let want = [-0.0, -0.158_655_25, 0.0, 0.053_982_8, 0.841_344_7, 10.0];
        for (a, b) in g.iter().zip(want) {
            assert!((a - b).abs() < 1e-6, "{a} {b}");
        }
        assert_eq!(tanh1(100.0), tanh1(10.0));
        assert!((tanh1(-0.5) + 0.462_117_16).abs() < 1e-6);
    }

    #[test]
    fn integer_kernels_at_every_length() {
        for n in 0..40usize {
            for rows in [0usize, 1, 3, 4, 5, 9] {
                let r8: Vec<i8> = (0..rows * n).map(|i| ((i * 37) % 255) as i8).collect();
                let w = data(rows, 1);
                let mut acc = vec![0.5f32; n];
                weighted_sum_int(&w, &r8, &mut acc);
                for i in 0..n {
                    let want = 0.5 + (0..rows).map(|r| w[r] * r8[r * n + i] as f32).sum::<f32>();
                    assert!((acc[i] - want).abs() < 1e-3, "{rows}x{n}");
                }
                let x = data(n, 2);
                let mut y = vec![0.0f32; rows];
                matvec_int(&r8, &x, &mut y);
                for r in 0..rows {
                    let want: f32 = (0..n).map(|i| r8[r * n + i] as f32 * x[i]).sum();
                    assert!((y[r] - want).abs() < 1e-3, "{rows}x{n} row {r}");
                }
            }
        }
    }
}
