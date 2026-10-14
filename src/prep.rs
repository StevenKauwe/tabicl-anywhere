//! Table preprocessing, made to give the same numbers as the reference's scikit-learn pipeline:
//!
//! 1. missing cells (NaN) take the column's training mean; columns with a single value are dropped;
//! 2. each column is standardised (mean 0, deviation 1, clipped to +-100);
//! 3. for the "power" variant, a Yeo-Johnson power transform with the likelihood-maximising exponent, then
//!    standardised again;
//! 4. values beyond 4 deviations (computed twice, the second time without the outliers) are pulled in
//!    logarithmically.
//!
//! Everything here is 64-bit and per column, and fitting keeps only a handful of numbers per column, so a table
//! can be pushed through a chunk of rows at a time.
//!
//! Also here: Python's `random.Random` (Mersenne Twister with Python's own shuffle and choice), because the
//! reference derives its ensemble's column orders from it and matching it member for member is what lets the
//! two implementations be compared exactly.

/// Python's `random.Random(seed)` for a non-negative integer seed: `shuffle` and `choice` give Python's results.
pub struct PyRandom {
    mt: [u32; 624],
    i: usize,
}

impl PyRandom {
    pub fn new(seed: u32) -> PyRandom {
        let mut mt = [0u32; 624];
        mt[0] = 19650218;
        for i in 1..624 {
            mt[i] = 1812433253u32.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        // init_by_array with the one-word key [seed]
        let (mut i, mut j) = (1usize, 0u32);
        for _ in 0..624 {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1664525)).wrapping_add(seed).wrapping_add(j);
            i += 1;
            j = 0; // key length 1
            if i >= 624 {
                mt[0] = mt[623];
                i = 1;
            }
        }
        for _ in 0..623 {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                mt[0] = mt[623];
                i = 1;
            }
        }
        mt[0] = 0x8000_0000;
        PyRandom { mt, i: 624 }
    }

    fn next_u32(&mut self) -> u32 {
        if self.i >= 624 {
            for k in 0..624 {
                let y = (self.mt[k] & 0x8000_0000) | (self.mt[(k + 1) % 624] & 0x7fff_ffff);
                self.mt[k] = self.mt[(k + 397) % 624] ^ (y >> 1) ^ if y & 1 != 0 { 0x9908_b0df } else { 0 };
            }
            self.i = 0;
        }
        let mut y = self.mt[self.i];
        self.i += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// A uniform integer in `0..n` (Python's `_randbelow`); `n` must fit 32 bits.
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0 && n <= u32::MAX as usize);
        let k = usize::BITS - n.leading_zeros();
        loop {
            let r = (self.next_u32() >> (32 - k)) as usize;
            if r < n {
                return r;
            }
        }
    }

    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.below(i + 1);
            x.swap(i, j);
        }
    }
}

/// The reference's "latin" column orders: the rows of a random Latin square of side `n`, `n` orders in all.
pub fn latin_orders(n: usize, seed: u32) -> Vec<Vec<usize>> {
    let mut rng = PyRandom::new(seed);
    let mut symbols: Vec<usize> = (0..n).collect();
    let mut picks = Vec::new();
    while symbols.len() > 1 {
        let at = rng.below(symbols.len());
        picks.push(symbols.remove(at));
    }
    let mut square = vec![symbols];
    for sym in picks.into_iter().rev() {
        square.push(square[0].clone());
        for (i, row) in square.iter_mut().enumerate() {
            row.insert(i, sym);
        }
    }
    rng.shuffle(&mut square);
    let mut cols: Vec<Vec<usize>> = (0..n).map(|c| square.iter().map(|row| row[c]).collect()).collect();
    rng.shuffle(&mut cols);
    cols
}

/// One ensemble member: which normalisation, the order its features are presented in, and how class `c` is
/// relabelled (`classes[c]`).
#[derive(Clone, Debug, PartialEq)]
pub struct Member {
    pub norm: usize,
    pub features: Vec<usize>,
    pub classes: Vec<usize>,
}

/// The reference's ensemble for `n_features` kept features, `n_classes` classes and `n_norms` normalisations.
pub fn ensemble(n_features: usize, n_classes: usize, n_norms: usize, n_estimators: usize, seed: u32) -> Vec<Member> {
    let ident = |n: usize| vec![(0..n).collect::<Vec<usize>>()];
    let feats = if n_estimators == 1 { ident(n_features) } else { latin_orders(n_features, seed) };
    let classes = if n_estimators == 1 {
        ident(n_classes)
    } else {
        // every circular shift
        (0..n_classes).map(|i| (0..n_classes).map(|c| (c + n_classes - i) % n_classes).collect()).collect()
    };
    let mut configs: Vec<(usize, usize)> = (0..feats.len()).flat_map(|f| (0..classes.len()).map(move |c| (f, c))).collect();
    PyRandom::new(seed).shuffle(&mut configs);
    configs
        .iter()
        .flat_map(|&(f, c)| (0..n_norms).map(move |norm| (f, c, norm)))
        .take(n_estimators)
        .map(|(f, c, norm)| Member { norm, features: feats[f].clone(), classes: classes[c].clone() })
        .collect()
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// Variance about `m` with divisor `len - ddof`.
fn var(x: &[f64], m: f64, ddof: usize) -> f64 {
    x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (x.len() - ddof) as f64
}

/// The Yeo-Johnson power transform of one value.
#[inline]
pub fn yeo_johnson(x: f64, lambda: f64) -> f64 {
    let eps = f64::EPSILON;
    if x >= 0.0 {
        if lambda.abs() < eps {
            x.ln_1p()
        } else {
            (lambda * x.ln_1p()).exp_m1() / lambda
        }
    } else if (lambda - 2.0).abs() > eps {
        -((2.0 - lambda) * (-x).ln_1p()).exp_m1() / (2.0 - lambda)
    } else {
        -(-x).ln_1p()
    }
}

/// Minimise `f` on `[a, b]` by golden-section steps with parabolic interpolation (the classic bounded Brent
/// method, the algorithm behind SciPy's `fminbound`), to an absolute tolerance of `xtol` on the argument.
pub fn minimize_bounded(mut f: impl FnMut(f64) -> f64, mut a: f64, mut b: f64, xtol: f64) -> f64 {
    let sqrt_eps = (2.2e-16f64).sqrt();
    let golden = 0.5 * (3.0 - 5.0f64.sqrt());
    let (mut fulc, mut nfc, mut xf);
    fulc = a + golden * (b - a);
    nfc = fulc;
    xf = fulc;
    let (mut rat, mut e) = (0.0f64, 0.0f64);
    let mut fx = f(xf);
    let (mut ffulc, mut fnfc) = (fx, fx);
    let mut xm = 0.5 * (a + b);
    let mut tol1 = sqrt_eps * xf.abs() + xtol / 3.0;
    let mut tol2 = 2.0 * tol1;
    let sign = |v: f64| {
        if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }
    };
    for _ in 0..500 {
        if (xf - xm).abs() <= tol2 - 0.5 * (b - a) {
            break;
        }
        let mut use_golden = true;
        if e.abs() > tol1 {
            use_golden = false;
            let mut r = (xf - nfc) * (fx - ffulc);
            let mut q = (xf - fulc) * (fx - fnfc);
            let mut p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            r = e;
            e = rat;
            if p.abs() < (0.5 * q * r).abs() && p > q * (a - xf) && p < q * (b - xf) {
                rat = p / q;
                let x = xf + rat;
                if x - a < tol2 || b - x < tol2 {
                    let si = sign(xm - xf) + if xm - xf == 0.0 { 1.0 } else { 0.0 };
                    rat = tol1 * si;
                }
            } else {
                use_golden = true;
            }
        }
        if use_golden {
            e = if xf >= xm { a - xf } else { b - xf };
            rat = golden * e;
        }
        let si = sign(rat) + if rat == 0.0 { 1.0 } else { 0.0 };
        let x = xf + si * rat.abs().max(tol1);
        let fu = f(x);
        if fu <= fx {
            if x >= xf {
                a = xf;
            } else {
                b = xf;
            }
            (fulc, ffulc) = (nfc, fnfc);
            (nfc, fnfc) = (xf, fx);
            (xf, fx) = (x, fu);
        } else {
            if x < xf {
                a = x;
            } else {
                b = x;
            }
            if fu <= fnfc || nfc == xf {
                (fulc, ffulc) = (nfc, fnfc);
                (nfc, fnfc) = (x, fu);
            } else if fu <= ffulc || fulc == xf || fulc == nfc {
                (fulc, ffulc) = (x, fu);
            }
        }
        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * xf.abs() + xtol / 3.0;
        tol2 = 2.0 * tol1;
    }
    xf
}

/// The Yeo-Johnson exponent that makes `x` look most normal (maximum likelihood), searched over the range of
/// exponents for which the transform cannot overflow, as SciPy does.
pub fn yeo_johnson_lambda(x: &[f64]) -> f64 {
    if x.iter().all(|v| *v == 0.0) {
        return 1.0;
    }
    let big = x.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    let log1p_max = (20.0 * big).ln_1p();
    let log_eps = f64::EPSILON.ln();
    let (mut lb, mut ub) = ((f64::MIN_POSITIVE.ln() - log_eps) / 2.0 / log1p_max, (f64::MAX.ln() + log_eps) / 2.0 / log1p_max);
    if x.iter().all(|v| *v < 0.0) {
        (lb, ub) = (2.0 - ub, 2.0 - lb);
    } else if x.iter().any(|v| *v < 0.0) {
        (lb, ub) = ((2.0 - ub).max(lb), (2.0 - lb).min(ub));
    }
    let n = x.len() as f64;
    let jac: f64 = x.iter().map(|v| v.signum() * v.abs().ln_1p() * if *v == 0.0 { 0.0 } else { 1.0 }).sum();
    let mut y = vec![0.0f64; x.len()];
    let neg_llf = |l: f64| {
        for (o, v) in y.iter_mut().zip(x) {
            *o = yeo_johnson(*v, l);
        }
        let s = var(&y, mean(&y), 0);
        let logvar = if s >= f64::MIN_POSITIVE { s.ln() } else { f64::NEG_INFINITY };
        let llf = -n / 2.0 * logvar + (l - 1.0) * jac;
        if llf.is_infinite() || llf.is_nan() {
            f64::INFINITY
        } else {
            -llf
        }
    };
    minimize_bounded(neg_llf, lb, ub, 1.48e-8)
}

/// scikit-learn's test for "this column is constant as far as floating point can tell".
fn is_constant(var: f64, mean: f64, n: usize) -> bool {
    let n = n as f64;
    var <= n * f64::EPSILON * var + (n * mean * f64::EPSILON).powi(2)
}

/// What one normalisation variant remembers about one column.
#[derive(Clone, Debug, Default)]
pub struct ColNorm {
    pub mean: f64,
    pub scale: f64,
    /// Yeo-Johnson exponent, and the mean and deviation applied after it (unused for the plain variant).
    pub lambda: f64,
    pub pmean: f64,
    pub pscale: f64,
    pub lower: f64,
    pub upper: f64,
}

pub const COL_NORM_FIELDS: usize = 7;

impl ColNorm {
    pub fn to_array(&self) -> [f64; COL_NORM_FIELDS] {
        [self.mean, self.scale, self.lambda, self.pmean, self.pscale, self.lower, self.upper]
    }
    pub fn from_array(a: &[f64]) -> ColNorm {
        ColNorm { mean: a[0], scale: a[1], lambda: a[2], pmean: a[3], pscale: a[4], lower: a[5], upper: a[6] }
    }

    /// Learn the column's numbers from its training values (already imputed), transforming them in place.
    pub fn fit(col: &mut [f64], power: bool, threshold: f64) -> ColNorm {
        let n = col.len();
        let mut c = ColNorm { lambda: 1.0, pscale: 1.0, ..Default::default() };
        c.mean = mean(col);
        c.scale = var(col, c.mean, 0).sqrt() + 1e-6;
        col.iter_mut().for_each(|v| *v = ((*v - c.mean) / c.scale).clamp(-100.0, 100.0));
        if power {
            let m = mean(col);
            if !is_constant(var(col, m, 0), m, n) {
                c.lambda = yeo_johnson_lambda(col);
                col.iter_mut().for_each(|v| *v = yeo_johnson(*v, c.lambda));
            }
            c.pmean = mean(col);
            let v = var(col, c.pmean, 0);
            c.pscale = if is_constant(v, c.pmean, n) { 1.0 } else { v.sqrt() };
            col.iter_mut().for_each(|v| *v = (*v - c.pmean) / c.pscale);
        }
        // outlier bounds: mean and deviation, then again without what falls outside
        let ddof = if n > 1 { 1 } else { 0 };
        let m = mean(col);
        let s = var(col, m, ddof).sqrt().max(1e-6);
        let (lo, hi) = (m - threshold * s, m + threshold * s);
        let kept: Vec<f64> = col.iter().copied().filter(|v| !(*v < lo || *v > hi)).collect();
        let m = mean(&kept);
        let s = if kept.len() > ddof { var(&kept, m, ddof).sqrt() } else { f64::NAN }.max(1e-6);
        c.lower = m - threshold * s;
        c.upper = m + threshold * s;
        col.iter_mut().for_each(|v| *v = c.squash(*v));
        c
    }

    #[inline]
    fn squash(&self, x: f64) -> f64 {
        let x = (-x.abs().ln_1p() + self.lower).max(x);
        (x.abs().ln_1p() + self.upper).min(x)
    }

    /// Transform one new value (already imputed).
    #[inline]
    pub fn apply(&self, x: f64, power: bool) -> f64 {
        let mut v = ((x - self.mean) / self.scale).clamp(-100.0, 100.0);
        if power {
            v = (yeo_johnson(v, self.lambda) - self.pmean) / self.pscale;
        }
        self.squash(v)
    }
}

/// The fitted preprocessing of a table.
#[derive(Clone, Debug, Default)]
pub struct Prep {
    /// Features in the raw table.
    pub n_in: usize,
    /// Training mean of every raw column (0 where the whole column was missing): what missing cells become.
    pub impute: Vec<f64>,
    /// Raw indices of the columns that have more than one distinct value.
    pub keep: Vec<usize>,
    /// For each variant (`power[v]` says which kind): one `ColNorm` per kept column.
    pub power: Vec<bool>,
    pub norms: Vec<Vec<ColNorm>>,
}

impl Prep {
    /// Learn from a training table `x` (`[n][h]`, NaN for missing). Returns, per variant, the transformed
    /// training table `[n][kept]`.
    pub fn fit(x: &[f64], n: usize, h: usize, power: &[bool], threshold: f64) -> (Prep, Vec<Vec<f64>>) {
        assert_eq!(x.len(), n * h);
        let mut p = Prep { n_in: h, power: power.to_vec(), ..Default::default() };
        let mut cols: Vec<Vec<f64>> = Vec::new();
        for j in 0..h {
            let mut col: Vec<f64> = (0..n).map(|r| x[r * h + j]).collect();
            let seen: Vec<f64> = col.iter().copied().filter(|v| !v.is_nan()).collect();
            let fill = if seen.is_empty() { 0.0 } else { mean(&seen) };
            col.iter_mut().for_each(|v| {
                if v.is_nan() {
                    *v = fill
                }
            });
            p.impute.push(fill);
            if n <= 1 || col.iter().any(|v| *v != col[0]) {
                p.keep.push(j);
                cols.push(col);
            }
        }
        if p.keep.is_empty() {
            // the reference keeps the first column rather than none
            p.keep.push(0);
            cols.push(vec![p.impute[0]; n]);
        }
        let k = p.keep.len();
        let mut out = Vec::new();
        for &pw in power {
            let mut t = vec![0.0f64; n * k];
            let mut norms = Vec::new();
            for (j, col) in cols.iter().enumerate() {
                let mut c = col.clone();
                norms.push(ColNorm::fit(&mut c, pw, threshold));
                for r in 0..n {
                    t[r * k + j] = c[r];
                }
            }
            p.norms.push(norms);
            out.push(t);
        }
        (p, out)
    }

    /// Transform one raw row for variant `v` into `out` (`kept` values).
    pub fn apply(&self, v: usize, row: &[f64], out: &mut [f64]) {
        for (j, &src) in self.keep.iter().enumerate() {
            let x = if row[src].is_nan() { self.impute[src] } else { row[src] };
            out[j] = self.norms[v][j].apply(x, self.power[v]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_random_matches() {
        // random.Random(42): [getrandbits(32) for _ in range(3)], then shuffle(list(range(10)))
        let mut r = PyRandom::new(42);
        assert_eq!([r.next_u32(), r.next_u32(), r.next_u32()], [2746317213, 478163327, 107420369]);
        let mut r = PyRandom::new(42);
        let mut v: Vec<usize> = (0..10).collect();
        r.shuffle(&mut v);
        assert_eq!(v, [7, 3, 2, 8, 5, 6, 9, 4, 0, 1]);
    }

    #[test]
    fn bounded_minimum() {
        let x = minimize_bounded(|x| (x - 1.3).powi(2) + 0.1 * (5.0 * x).sin(), -2.0, 4.0, 1.48e-8);
        let slope = 2.0 * (x - 1.3) + 0.5 * (5.0 * x).cos();
        assert!(slope.abs() < 1e-5, "{x} {slope}");
    }
}
