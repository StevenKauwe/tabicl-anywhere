//! Optional graphics-card acceleration (`--features gpu`), through wgpu: Vulkan on Linux, Metal on macOS,
//! DirectX on Windows, and the browser's own WebGPU when built for the web.
//!
//! Prediction runs all three stages on the card, against a fitted model held in graphics memory (`VramModel`,
//! 16 bits per number, kept by a `VramTier` with per-tenant budgets). Fitting runs its three stages there too,
//! a batch of columns, of rows, or a layer at a time (`impl Attend for Gpu`); the cache written is in the same
//! format as the CPU's.
//!
//! The shaders are plain WGSL with no optional features. What makes them quick is the same on any card:
//!
//! * every number fetched from graphics memory is used for many multiplications (a block of queries against a
//!   block of keys, a block of rows against a block of weights), instead of one dot product per invocation;
//! * invocations that are neighbours in a workgroup read the same keys, values or input rows;
//! * running sums live in plainly named variables, which drivers keep in registers, never in arrays walked by
//!   a loop, which they do not;
//! * nothing large is written to graphics memory between passes, and working buffers are reused across calls.
//!
//! Waiting for the card, natively and in a browser. A browser lets nothing stop and wait for the card: finding
//! it and reading results back only ever complete later, when the code that asked has handed control back. So
//! every function here that needs a result from the card is `async`, and the waiting happens in exactly one
//! place, `Gpu::take`: natively it stops right there until the submission is through (`device.poll`), so the
//! future is finished the first time it is asked and costs nothing; in a browser it hands control back until the
//! browser calls with the result. Recording and submitting work never waits, in either. The loops that call the
//! card (`fit::fit_raw_async`, `predict::predict_raw_async`, and through them `table`) are `async` in the same
//! way, and their plain, blocking forms are `wait::block_on` of them; so there is one copy of the maths and of
//! the loops, and the native interface is what it was. The few cases where native code used a second thread
//! (loading a model into the tier in the background, overlapping the card with packing results in fitting)
//! are, respectively, done in line in a browser, and done by handing the card the next layer before waiting for
//! the last one, which needs no thread anywhere.
//!
//! Graphics memory, at worst, for one `Gpu` and one `VramTier`:
//!
//!   the tier's budget (fitted models; `VramTier::new`)
//!   + the working-memory budget (`Gpu::set_work_budget`, `WORK_BUDGET` = 1.5 GB unless set; see `Pool`)
//!   + the weights in use: about 80 MB per model kind (`GpuWeights::bytes`; shared, not charged to tenants)
//!   + a few kilobytes of parameter blocks and the shaders.
//!
//! Prediction cuts batches too large for the working-memory budget into sub-batches of rows; fitting sizes its
//! batches to it, and leaves to the CPU what does not fit even one at a time. `Gpu::release` gives back the idle
//! working buffers and the cached fitting weights; `VramTier::evict` the fitted models.

use crate::fit::Attend;
use crate::model::Model;
use crate::predict::{layer_rows, AttendCached, Cache};
use crate::store::Bundle;
use crate::wait::{once, Later, Once};
use crate::warm::{Key, Ledger};
use std::collections::HashMap;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex};
use wgpu::util::DeviceExt;

// The whole in-context stage on the card, as a chain of small passes that all share one binding layout:
// two parameter blocks, the layer's weights, up to three inputs and one output.
//
// What a graphics card is slow at is one thread doing many memory reads in a row, so the work is cut fine:
// each invocation produces a single output number from a few hundred reads at most, everything is read as
// four-vectors, and anything needed by many outputs (a layer norm, the hidden layer of a small network) is its
// own pass. Even one test row then spreads over hundreds of invocations per pass.
const STAGE_HEAD: &str = r#"
struct C { nq: u32, n: u32, parts: u32, heads: u32, d: u32, ff: u32, out_n: u32, set_mod: u32, set_div: u32, t: u32, pad0: u32, pad1: u32, sel: u32, pad2: u32, pad3: u32, pad4: u32 }
struct L { n1w: u32, n1b: u32, qw: u32, qb: u32, s0w: u32, s0b: u32, s2w: u32, s2b: u32, ow: u32, ob: u32, n2w: u32, n2b: u32, l1w: u32, l1b: u32, l2w: u32, l2b: u32, layer: u32, kw: u32, kb: u32, vw: u32, vb: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<uniform> c: C;
@group(0) @binding(1) var<uniform> l: L;
@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> x0: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> x1: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> x2: array<vec4<f32>>;

fn erf(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    var p = 1.0614054;
    p = p * t - 1.4531521;
    p = p * t + 1.4214138;
    p = p * t - 0.28449674;
    p = p * t + 0.2548296;
    let far = sign(x) * (1.0 - p * t * exp(-a * a));
    let x2 = x * x;
    var q = -1.0 / 1320.0;
    q = q * x2 + 1.0 / 216.0;
    q = q * x2 - 1.0 / 42.0;
    q = q * x2 + 0.1;
    q = q * x2 - 1.0 / 3.0;
    q = q * x2 + 1.0;
    return select(far, q * x * 1.1283791671, a < 0.5);
}
fn gelu(x: f32) -> f32 { return 0.5 * x * (1.0 + erf(x * 0.70710678)); }

// element j of the vector starting at four-vector `off` of the weights
fn wat(off: u32, j: u32) -> f32 { return w[off + j / 4u][j % 4u]; }

// The `linear` passes apply one of the layer's matrices, chosen by c.sel: 0 query, 1 key, 2 value projection,
// 3 attention output (adds x1, the block's input), 4 first feed-forward layer (through the GELU), 5 second
// (adds x1). These give where its weights and bias start, and how wide its input and output are.
fn lin_w() -> u32 {
    switch c.sel {
        case 0u { return l.qw; }
        case 1u { return l.kw; }
        case 2u { return l.vw; }
        case 3u { return l.ow; }
        case 4u { return l.l1w; }
        default { return l.l2w; }
    }
}
fn lin_b() -> u32 {
    switch c.sel {
        case 0u { return l.qb; }
        case 1u { return l.kb; }
        case 2u { return l.vb; }
        case 3u { return l.ob; }
        case 4u { return l.l1b; }
        default { return l.l2b; }
    }
}
fn lin_in() -> u32 { return select(c.d, c.ff, c.sel == 5u); }
fn lin_out() -> u32 { return select(c.d, c.ff, c.sel == 4u); }
"#;

const SHADER_STAGE: &str = r#"
@group(0) @binding(6) var<storage, read_write> o: array<f32>;

// row `row` (width `n4` four-vectors) of x0 dotted with the weight row starting at `off`
fn rowdot(off: u32, row: u32, n4: u32) -> f32 {
    var s = vec4<f32>(0.0);
    for (var i = 0u; i < n4; i++) { s += w[off + i] * x0[row * n4 + i]; }
    return s.x + s.y + s.z + s.w;
}

// mean and 1 / deviation of each row of x0 (width d), once per row: o = [rows][4] = (mean, 1/deviation, -, -)
@compute @workgroup_size(64)
fn stats(@builtin(global_invocation_id) id: vec3<u32>) {
    let ix = id.y * 2097152u + id.x;
    if (ix >= c.nq) { return; }
    let d4 = c.d / 4u;
    var m4 = vec4<f32>(0.0);
    for (var i = 0u; i < d4; i++) { m4 += x0[ix * d4 + i]; }
    let m = (m4.x + m4.y + m4.z + m4.w) / f32(c.d);
    var v4 = vec4<f32>(0.0);
    for (var i = 0u; i < d4; i++) { let t = x0[ix * d4 + i] - vec4<f32>(m); v4 += t * t; }
    o[ix * 4u] = m;
    o[ix * 4u + 1u] = inverseSqrt((v4.x + v4.y + v4.z + v4.w) / f32(c.d) + 1e-5);
}

// one element of the layer norm of row `row` of x0 (its statistics are in x1), weight and bias at the offsets
fn normed(row: u32, j: u32, nw: u32, nb: u32) -> f32 {
    let st = x1[row];
    let a = row * c.d + j;
    return (x0[a / 4u][a % 4u] - st.x) * st.y * wat(nw, j) + wat(nb, j);
}

// Rotary positions. x0 = queries or keys [tokens][d]; x1 = per position, cos then sin of each frequency.
// A token's position is its index within its row of t tokens.
fn rotated(idx: u32) -> f32 {
    let hd = c.d / c.heads; let half = hd / 2u;
    let tok = idx / c.d; let j = idx % c.d;
    let i = j % hd;
    let pos = tok % c.t;
    let f = i % half;
    let cs = x1[(pos * hd + f) / 4u][(pos * hd + f) % 4u];
    let sn = x1[(pos * hd + half + f) / 4u][(pos * hd + half + f) % 4u];
    let me = x0[idx / 4u][idx % 4u];
    if (i < half) {
        let other = idx + half;
        return me * cs - x0[other / 4u][other % 4u] * sn;
    }
    let other = idx - half;
    return me * cs + x0[other / 4u][other % 4u] * sn;
}
@compute @workgroup_size(64)
fn ropeq(@builtin(global_invocation_id) id: vec3<u32>) {
    let ix = id.y * 2097152u + id.x;
    if (ix >= c.nq * c.d) { return; }
    o[ix] = rotated(ix);
}
// keys also take the 1 / sqrt(head width) that scores are divided by
@compute @workgroup_size(64)
fn ropek(@builtin(global_invocation_id) id: vec3<u32>) {
    let ix = id.y * 2097152u + id.x;
    if (ix >= c.nq * c.d) { return; }
    o[ix] = rotated(ix) * inverseSqrt(f32(c.d / c.heads));
}

// Lay out each row's tokens for the row stage: out_n class tokens (x1), then its column embeddings (x0).
@compute @workgroup_size(64)
fn place(@builtin(global_invocation_id) id: vec3<u32>) {
    let ix = id.y * 2097152u + id.x;
    if (ix >= c.nq * c.t * c.d) { return; }
    let row = ix / (c.t * c.d); let tk = (ix / c.d) % c.t; let j = ix % c.d;
    if (tk < c.out_n) {
        let a = tk * c.d + j;
        o[ix] = x1[a / 4u][a % 4u];
    } else {
        let a = (row * (c.t - c.out_n) + tk - c.out_n) * c.d + j;
        o[ix] = x0[a / 4u][a % 4u];
    }
}

// The input layer. x0 = the rows' features [rows][n]; the weights are one four-vector per output coordinate:
// its three weights and its bias. A cell's three values are the features one, two and four places to the right
// of its own, wrapping round; a cell whose values are all missing (-100) stays marked as missing.
// o = cells [rows][n][d].
@compute @workgroup_size(64)
fn cells(@builtin(global_invocation_id) id: vec3<u32>) {
    let ix = id.y * 2097152u + id.x;
    if (ix >= c.nq * c.n * c.d) { return; }
    let g = (ix / c.d) % c.n;
    let a = (ix / (c.n * c.d)) * c.n;
    let i0 = a + (g + 1u) % c.n; let i1 = a + (g + 2u) % c.n; let i2 = a + (g + 4u) % c.n;
    let v = vec3<f32>(x0[i0 / 4u][i0 % 4u], x0[i1 / 4u][i1 % 4u], x0[i2 / 4u][i2 % 4u]);
    let wj = w[ix % c.d];
    o[ix] = select(dot(wj.xyz, v) + wj.w, -100.0, all(v == vec3<f32>(-100.0)));
}

// Keys and values worked out on the card, rearranged for the many-keys attention. x0 = [row][set][d] (c.nq
// rows, c.parts sets): each set is one group of keys.
// turnk: o = [set][coordinate][key], each coordinate padded with zeros to c.n keys (whole blocks of 16).
@compute @workgroup_size(64)
fn turnk(@builtin(global_invocation_id) id: vec3<u32>) {
    let ix = id.y * 2097152u + id.x;
    if (ix >= c.parts * c.d * c.n) { return; }
    let r = ix % c.n; let j = (ix / c.n) % c.d; let ks = ix / (c.n * c.d);
    let a = (min(r, c.nq - 1u) * c.parts + ks) * c.d + j;
    o[ix] = select(0.0, x0[a / 4u][a % 4u], r < c.nq);
}
// turnv: o = [set][key][d].
@compute @workgroup_size(64)
fn turnv(@builtin(global_invocation_id) id: vec3<u32>) {
    let ix = id.y * 2097152u + id.x;
    if (ix >= c.parts * c.nq * c.d) { return; }
    let j = ix % c.d; let r = (ix / c.d) % c.nq; let ks = ix / (c.d * c.nq);
    let a = (r * c.parts + ks) * c.d + j;
    o[ix] = x0[a / 4u][a % 4u];
}

// The first out_n tokens of each row of t tokens (the class tokens), end to end. x0 = tokens.
@compute @workgroup_size(64)
fn first(@builtin(global_invocation_id) id: vec3<u32>) {
    let ix = id.y * 2097152u + id.x;
    if (ix >= c.nq * c.out_n * c.d) { return; }
    let row = ix / (c.out_n * c.d); let rest = ix % (c.out_n * c.d);
    let a = row * c.t * c.d + rest;
    o[ix] = x0[a / 4u][a % 4u];
}

// The row stage's result: the first out_n tokens of each row, normalised, end to end. x0 = tokens.
@compute @workgroup_size(64)
fn clsnorm(@builtin(global_invocation_id) id: vec3<u32>) {
    let ix = id.y * 2097152u + id.x;
    if (ix >= c.nq * c.out_n * c.d) { return; }
    let row = ix / (c.out_n * c.d); let cc = (ix / c.d) % c.out_n; let j = ix % c.d;
    o[ix] = normed(row * c.t + cc, j, l.n1w, l.n1b);
}

// x0 = raw query. o = hidden layer of the query-aware scaling: [rows][heads][64].
@compute @workgroup_size(64)
fn sshid(@builtin(global_invocation_id) id: vec3<u32>) {
    let ix = id.y * 2097152u + id.x;
    if (ix >= c.nq * c.heads * 64u) { return; }
    let hd4 = c.d / c.heads / 4u;
    let rh = ix / 64u; let u = ix % 64u;
    var s = vec4<f32>(0.0);
    for (var i = 0u; i < hd4; i++) { s += w[l.s0w + u * hd4 + i] * x0[rh * hd4 + i]; }
    o[ix] = gelu(wat(l.s0b, u) + s.x + s.y + s.z + s.w);
}

// x0 = that hidden layer, x1 = raw query, x2 = the fitted model's per-layer scales (with the scales of the
// stored keys folded in). o = query, scaled.
@compute @workgroup_size(64)
fn ssout(@builtin(global_invocation_id) id: vec3<u32>) {
    let ix = id.y * 2097152u + id.x;
    if (ix >= c.nq * c.d) { return; }
    let hd = c.d / c.heads;
    let row = ix / c.d; let j = ix % c.d;
    let rh = row * c.heads + j / hd; let e = j % hd;
    var s = vec4<f32>(0.0);
    for (var i = 0u; i < 16u; i++) { s += w[l.s2w + e * 16u + i] * x0[rh * 16u + i]; }
    let m = wat(l.s2b, e) + s.x + s.y + s.z + s.w;
    let sj = l.layer * c.d + j;
    // laid out [coordinate][row] (rows rounded up to a multiple of 4) for the attention
    let nq4 = (c.nq + 3u) / 4u;
    o[(j * nq4 + row / 4u) * 4u + row % 4u] = x1[ix / 4u][ix % 4u] * x2[sj / 4u][sj % 4u] * (1.0 + tanh(m));
}

// A matrix of the layer (see `lin_w`) applied to the rows of x0, one output number per invocation. This is the
// form for a handful of rows: it spreads them over the most invocations, and with so few the card is short of
// things to do at once, not of memory traffic. (`linear`, in the next shader, is the form for many rows.)
@compute @workgroup_size(64)
fn linear1(@builtin(global_invocation_id) id: vec3<u32>) {
    let ix = id.y * 2097152u + id.x;
    let wide = lin_out();
    if (ix >= c.nq * wide) { return; }
    let row = ix / wide; let j = ix % wide;
    let n4 = lin_in() / 4u;
    var t = wat(lin_b(), j) + rowdot(lin_w() + j * n4, row, n4);
    if (c.sel == 4u) { t = gelu(t); }
    if (c.sel == 3u || c.sel == 5u) { t += x1[ix / 4u][ix % 4u]; }
    o[ix] = t;
}

// x0 = the decoder's middle. o = the first out_n outputs.
@compute @workgroup_size(64)
fn dec(@builtin(global_invocation_id) id: vec3<u32>) {
    let ix = id.y * 2097152u + id.x;
    if (ix >= c.nq * c.out_n) { return; }
    let row = ix / c.out_n; let j = ix % c.out_n;
    o[ix] = wat(l.l2b, j) + rowdot(l.l2w + j * (c.ff / 4u), row, c.ff / 4u);
}
"#;

// The network's matrices applied to many rows at once. One invocation makes a 4 x 4 block of outputs (4 rows of
// the input, 4 rows of the weights): each step of its loop fetches 8 four-vectors and does 64 multiplications
// with them, where one output per invocation would fetch 2 for every 4. (4 x 8 blocks were tried and were a
// fifth slower: too much for the registers.)
const SHADER_LINEAR: &str = r#"
@group(0) @binding(6) var<storage, read_write> o: array<vec4<f32>>;

// rows r0..r0+3 of x0 (n4 four-vectors wide) times the four weight rows starting at four-vector `off`:
// the 16 dot products, as [row][output]
fn block(off: u32, r0: u32, n4: u32) -> mat4x4<f32> {
    var a0 = vec4<f32>(0.0); var a1 = a0; var a2 = a0; var a3 = a0;
    let xo = r0 * n4;
    for (var i = 0u; i < n4; i++) {
        let m = mat4x4<f32>(w[off + i], w[off + n4 + i], w[off + 2u * n4 + i], w[off + 3u * n4 + i]);
        a0 += x0[xo + i] * m;
        a1 += x0[xo + n4 + i] * m;
        a2 += x0[xo + 2u * n4 + i] * m;
        a3 += x0[xo + 3u * n4 + i] * m;
    }
    return mat4x4<f32>(a0, a1, a2, a3);
}

// x0 = h, x1 = its rows' statistics. o = norm1(h), a four-vector per invocation.
@compute @workgroup_size(64)
fn norm1(@builtin(global_invocation_id) id: vec3<u32>) {
    let ix = id.y * 2097152u + id.x;
    let d4 = c.d / 4u;
    if (ix >= c.nq * d4) { return; }
    let st = x1[ix / d4];
    o[ix] = (x0[ix] - st.x) * st.y * w[l.n1w + ix % d4] + w[l.n1b + ix % d4];
}

// The same with the second norm's weights.
@compute @workgroup_size(64)
fn norm2(@builtin(global_invocation_id) id: vec3<u32>) {
    let ix = id.y * 2097152u + id.x;
    let d4 = c.d / 4u;
    if (ix >= c.nq * d4) { return; }
    let st = x1[ix / d4];
    o[ix] = (x0[ix] - st.x) * st.y * w[l.n2w + ix % d4] + w[l.n2b + ix % d4];
}

// A matrix of the layer (see `lin_w`) applied to the rows of x0. Invocation ix works on rows 4 * (ix / out4)..
// and outputs 4 * (ix % out4).. (out4 = four-vectors of output per row), so consecutive invocations share
// their input rows, which a workgroup then fetches once.
@compute @workgroup_size(64)
fn linear(@builtin(global_invocation_id) id: vec3<u32>) {
    let ix = id.y * 2097152u + id.x;
    let n4 = lin_in() / 4u; let o4 = lin_out() / 4u;
    let r0 = (ix / o4) * 4u; let jb = ix % o4;
    if (r0 >= c.nq) { return; }
    let a = block(lin_w() + jb * 4u * n4, r0, n4);
    let bias = w[lin_b() + jb];
    for (var r = 0u; r < 4u; r++) {
        if (r0 + r >= c.nq) { break; }
        let at = (r0 + r) * o4 + jb;
        var t = a[r] + bias;
        if (c.sel == 4u) { t = vec4<f32>(gelu(t.x), gelu(t.y), gelu(t.z), gelu(t.w)); }
        if (c.sel == 3u || c.sel == 5u) { t += x1[at]; }
        o[at] = t;
    }
}
"#;

// Attention over a few keys (a column's inducing points, a row's own tokens) on the shared binding layout:
// x0 = queries [queries][d], x1 = keys and x2 = values [sets][n][d], o = outputs [queries][d].
//
// One invocation takes one head of 4 queries that share a set of keys, and goes through the keys 4 at a time:
// 4 x 4 scores from the 4 keys just fetched, then the 4 values weighted into each query's running average. So
// each fetched key and value serves four queries, and the exponentials come four at a time. The running
// average is kept relative to the largest score so far, as everywhere else; when a larger one turns up, what
// has been gathered is scaled down to match (`f` below, which is 1 otherwise).
//
// As in the many-keys attention further down, the queries and the averages have to be plainly named variables
// to stay in registers, so the lines that repeat per four-vector of the head are written out by `few_source`.
const SHADER_FEW: &str = r#"
struct C { nq: u32, n: u32, parts: u32, heads: u32, d: u32, ff: u32, out_n: u32, set_mod: u32, set_div: u32, t: u32, pad0: u32, pad1: u32, sel: u32, pad2: u32, pad3: u32, pad4: u32 }
@group(0) @binding(0) var<uniform> c: C;
@group(0) @binding(3) var<storage, read> x0: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> x1: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> x2: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> o: array<vec4<f32>>;

// `parts` queries share each set of keys: they sit pad0 apart, set s starting at query s * pad1. Invocations
// that share a set are neighbours, so a workgroup fetches its keys and values once.
@compute @workgroup_size(64)
fn attend1(@builtin(global_invocation_id) id: vec3<u32>) {
    let ix = id.y * 2097152u + id.x;
    let fours = (c.parts + 3u) / 4u;
    let sg = ix % fours;
    let h = (ix / fours) % c.heads;
    let ks = ix / (fours * c.heads);
    if (ks * c.parts >= c.nq) { return; }
    let d4 = c.d / 4u;
    // where this head of each of the 4 queries is (a group's missing queries repeat its last one)
    let last = c.parts - 1u;
    let q0 = (min(sg * 4u, last) * c.pad0 + ks * c.pad1) * d4 + h * H4u;
    let q1 = (min(sg * 4u + 1u, last) * c.pad0 + ks * c.pad1) * d4 + h * H4u;
    let q2 = (min(sg * 4u + 2u, last) * c.pad0 + ks * c.pad1) * d4 + h * H4u;
    let q3 = (min(sg * 4u + 3u, last) * c.pad0 + ks * c.pad1) * d4 + h * H4u;
    // the queries, turned so that one multiplication of a key by `m` gives part of all 4 scores
    TURN_QUERIES
    DECLARE_AVERAGES
    var top = vec4<f32>(-1.0e30);
    var sum = vec4<f32>(0.0);
    let kb = ks * c.n * d4 + h * H4u;
    for (var r = 0u; r < c.n; r += 4u) {
        // the 4 keys of this step (past the end, the last key again, which then weighs nothing)
        let k0 = kb + r * d4;
        let k1 = kb + min(r + 1u, c.n - 1u) * d4;
        let k2 = kb + min(r + 2u, c.n - 1u) * d4;
        let k3 = kb + min(r + 3u, c.n - 1u) * d4;
        let s0 = SCORE0;
        let s1 = select(vec4<f32>(-1.0e30), SCORE1, r + 1u < c.n);
        let s2 = select(vec4<f32>(-1.0e30), SCORE2, r + 2u < c.n);
        let s3 = select(vec4<f32>(-1.0e30), SCORE3, r + 3u < c.n);
        let now = max(max(top, max(s0, s1)), max(s2, s3));
        let f = exp(top - now);
        top = now;
        let w0 = exp(s0 - top); let w1 = exp(s1 - top); let w2 = exp(s2 - top); let w3 = exp(s3 - top);
        sum = sum * f + w0 + w1 + w2 + w3;
        ADD_VALUES
    }
    WRITE_AVERAGES
}
"#;

/// `SHADER_FEW` for heads `h4` four-vectors wide, with its repetitive lines written out.
fn few_source(h4: usize) -> String {
    let lines = |f: &dyn Fn(usize) -> String| (0..h4).map(f).collect::<Vec<_>>().join("\n    ");
    let score = |t: usize| (0..h4).map(|j| format!("x1[k{t} + {j}u] * m{j}")).collect::<Vec<_>>().join(" + ");
    let mut src = SHADER_FEW
        .replace("TURN_QUERIES", &lines(&|j| format!("let m{j} = mat4x4<f32>(x0[q0 + {j}u], x0[q1 + {j}u], x0[q2 + {j}u], x0[q3 + {j}u]);")))
        .replace("DECLARE_AVERAGES", &lines(&|j| format!("var a0_{j} = vec4<f32>(0.0); var a1_{j} = a0_{j}; var a2_{j} = a0_{j}; var a3_{j} = a0_{j};")))
        .replace(
            "ADD_VALUES",
            &lines(&|j| {
                let add = (0..4).map(|t| format!("a{t}_{j} = a{t}_{j} * f[{t}] + w0[{t}] * v0 + w1[{t}] * v1 + w2[{t}] * v2 + w3[{t}] * v3;")).collect::<Vec<_>>().join(" ");
                format!("    {{ let v0 = x2[k0 + {j}u]; let v1 = x2[k1 + {j}u]; let v2 = x2[k2 + {j}u]; let v3 = x2[k3 + {j}u]; {add} }}")
            }),
        )
        .replace(
            "WRITE_AVERAGES",
            &(0..4)
                .map(|t| format!("if (sg * 4u + {t}u < c.parts) {{ {} }}", (0..h4).map(|j| format!("o[q{t} + {j}u] = a{t}_{j} / sum[{t}];")).collect::<Vec<_>>().join(" ")))
                .collect::<Vec<_>>()
                .join("\n    "),
        )
        .replace("H4u", &format!("{h4}u"));
    for t in 0..4 {
        src = src.replace(&format!("SCORE{t}"), &score(t));
    }
    src
}

// The attention of the in-context stage: many keys, shared by every query.
//
// One invocation of `attend` takes one head of QN queries over a slice of the keys, 16 keys at a time:
//
//   scores   the QN x 16 scores of the block. Queries are stored coordinate by coordinate and so are keys, so
//            each step of the loop over the head's coordinates fetches the queries' number and the 16 keys'
//            and does 16 QN multiplications with them.
//   weights  the scores' exponentials relative to the largest score so far, as everywhere else; when a larger
//            one turns up, what has been gathered is scaled down to match (`f` below, which is 1 otherwise).
//   values   each of the 16 values, a whole head of it, weighted into the queries' running averages.
//
// Every number fetched is used at least QN times and most 16 QN times, and nothing but the averages of each
// slice goes back to graphics memory; `merge` then adds the slices together. (Holding all the scores in
// graphics memory between a scores pass and a values pass was tried: moving them there and back cost more
// than everything else together.)
//
// The scores and the running averages must stay in the card's registers for this to be quick, and drivers
// only keep plainly named variables there, not arrays that a loop walks through. So the repetitive lines (one
// per score, one per coordinate of each average) are written out by `attend_source` below; `S` stands for the
// type that holds one number per query.
//
// Keys and values come in two forms, chosen when the shader is built: plain floats (fitting, which must be
// exact), or 16-bit integers packed two to a word with one scale per coordinate (a resident model: half the
// graphics memory). The scales never appear in `attend`: the keys' are folded into the queries beforehand, the
// values' are applied in `merge`.
const SHADER_ATTEND: &str = r#"
struct A { np: u32, n: u32, parts: u32, heads: u32, d: u32, q0: u32, qb: u32, nb: u32, layer: u32, rows: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> c: A;
@group(0) @binding(3) var<storage, read> x0: array<S>;
@group(0) @binding(4) var<storage, read> x1: array<KVTYPE>;
@group(0) @binding(5) var<storage, read> x2: array<KVTYPE>;
@group(0) @binding(6) var<storage, read_write> o: array<vec4<f32>>;

// sixteen consecutive keys of one coordinate (from x1), or sixteen consecutive coordinates of one value (x2)
SIXTEEN

// x0 = queries [coordinate][group of QN], x1 = keys [coordinate][key], x2 = values [key][coordinate].
// o = [group][head][slice][record]: each query's weighted sum of values (a head of them), then (largest
// score, total weight) of each query, two queries to a four-vector.
@compute @workgroup_size(64)
fn attend(@builtin(global_invocation_id) id: vec3<u32>) {
    let ix = id.y * 2097152u + id.x;
    if (ix >= c.parts * c.heads * c.qb) { return; }
    // groups of queries vary fastest, so a workgroup's invocations fetch the same keys and values at the same time
    let g = ix % c.qb;
    let h = (ix / c.qb) % c.heads;
    let part = ix / (c.qb * c.heads);
    let base = c.nb / c.parts;
    let rem = c.nb % c.parts;
    let b0 = part * base + min(part, rem);
    let b1 = b0 + base + select(0u, 1u, part < rem);
    let qo = h * HDu * c.np + c.q0 + g;
    let ko = h * HDu * c.nb;
    var top = S(-1.0e30);
    var sum = S(0.0);
    DECLARE_AVERAGES
    for (var kb = b0; kb < b1; kb++) {
        DECLARE_SCORES
        for (var j = 0u; j < HDu; j++) {
            let qv = x0[qo + j * c.np];
            let k = keys16(ko + j * c.nb + kb);
            ADD_SCORES
        }
        // keys past the end (the last block's padding) must weigh nothing
        var s = array<S, 16>(LIST_SCORES);
        var now = top;
        for (var i = 0u; i < 16u; i++) {
            s[i] = select(S(-1.0e30), s[i], kb * 16u + i < c.n);
            now = max(now, s[i]);
        }
        let f = exp(top - now);
        top = now;
        sum *= f;
        SCALE_AVERAGES
        for (var i = 0u; i < 16u; i++) {
            let w = exp(s[i] - top);
            sum += w;
            let vo = (kb * 16u + i) * (c.d / 16u) + h * (HDu / 16u);
            LOAD_VALUES
            ADD_VALUES
        }
    }
    let oo = (((c.q0 + g) * c.heads + h) * c.parts + part) * RECORDu;
    WRITE_RECORD
}
"#;

/// `SHADER_ATTEND` for `qn` queries per invocation (1, 2 or 4) and heads of `h4` four-vectors (a multiple of 4),
/// with its repetitive lines written out.
fn attend_source(qn: usize, h4: usize, kv_type: &str, sixteen: &str) -> String {
    let s_type = ["", "f32", "vec2<f32>", "", "vec4<f32>"][qn];
    // component `q` of a per-query quantity
    let of = |name: &str, q: usize| if qn == 1 { name.to_string() } else { format!("{name}[{q}]") };
    let lines = |n: usize, f: &dyn Fn(usize) -> String| (0..n).map(f).collect::<Vec<_>>().join(" ");
    // average `j` (a four-vector of coordinates) of query `q` is the variable a{q}_{j}
    let averages = |f: &dyn Fn(usize, usize) -> String| (0..qn).map(|q| lines(h4, &|j| f(q, j))).collect::<Vec<_>>().join("\n        ");
    let record = (0..qn.div_ceil(2))
        .map(|p| {
            let second = if 2 * p + 1 < qn { format!("{}, {}", of("top", 2 * p + 1), of("sum", 2 * p + 1)) } else { "0.0, 0.0".to_string() };
            format!("o[oo + {}u] = vec4<f32>({}, {}, {second});", qn * h4 + p, of("top", 2 * p), of("sum", 2 * p))
        })
        .collect::<Vec<_>>()
        .join(" ");
    SHADER_ATTEND
        .replace("DECLARE_AVERAGES", &averages(&|q, j| format!("var a{q}_{j} = vec4<f32>(0.0);")))
        .replace("DECLARE_SCORES", &lines(16, &|i| format!("var s{i} = S(0.0);")))
        .replace("ADD_SCORES", &lines(16, &|i| format!("s{i} += qv * k[{}][{}];", i / 4, i % 4)))
        .replace("LIST_SCORES", &(0..16).map(|i| format!("s{i}")).collect::<Vec<_>>().join(", "))
        .replace("SCALE_AVERAGES", &averages(&|q, j| format!("a{q}_{j} *= {};", of("f", q))))
        .replace("ADD_VALUES", &averages(&|q, j| format!("a{q}_{j} += {} * v{}[{}];", of("w", q), j / 4, j % 4)))
        .replace("LOAD_VALUES", &lines(h4 / 4, &|b| format!("let v{b} = values16(vo + {b}u);")))
        .replace("WRITE_RECORD", &format!("{}\n    {record}", averages(&|q, j| format!("o[oo + {}u] = a{q}_{j};", q * h4 + j))))
        .replace("RECORD", &(qn * h4 + qn.div_ceil(2)).to_string())
        .replace("HDu", &format!("{}u", 4 * h4))
        .replace("KVTYPE", kv_type)
        .replace("SIXTEEN", sixteen)
        .replace("QN", &qn.to_string())
        .replace("array<S>", &format!("array<{s_type}>"))
        .replace("S(", &format!("{s_type}("))
        .replace("<S, 16>", &format!("<{s_type}, 16>"))
}

// Adding the slices of `attend` together, each brought to the largest score of all first. x0 = the slices,
// x2 = the values' scales [layers][d] (all ones for plain floats). o = attention output [rows][d], one
// four-vector per invocation. c.pad0 = queries per invocation of `attend`.
const SHADER_MERGE: &str = r#"
struct A { np: u32, n: u32, parts: u32, heads: u32, d: u32, q0: u32, qb: u32, nb: u32, layer: u32, rows: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> c: A;
@group(0) @binding(3) var<storage, read> x0: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> x2: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> o: array<vec4<f32>>;

@compute @workgroup_size(64)
fn merge(@builtin(global_invocation_id) id: vec3<u32>) {
    let ix = id.y * 2097152u + id.x;
    let d4 = c.d / 4u;
    let row = ix / d4;
    if (row >= c.rows) { return; }
    let j4 = ix % d4;
    let h4 = d4 / c.heads;
    let qn = c.pad0;
    let rec = qn * h4 + (qn + 1u) / 2u;
    let me = row % qn;
    let first = (((row / qn) * c.heads + j4 / h4) * c.parts) * rec;
    // where this query's (largest score, total weight) and its values are within a slice's record
    let ts = first + qn * h4 + me / 2u;
    let tc = (me % 2u) * 2u;
    let at = first + me * h4 + j4 % h4;
    var top = -1.0e30;
    for (var p = 0u; p < c.parts; p++) { top = max(top, x0[ts + p * rec][tc]); }
    var val = vec4<f32>(0.0);
    var tot = 0.0;
    for (var p = 0u; p < c.parts; p++) {
        let t = x0[ts + p * rec];
        let f = exp(t[tc] - top);
        tot += t[tc + 1u] * f;
        val += x0[at + p * rec] * f;
    }
    o[row * d4 + j4] = val / tot * x2[c.layer * d4 + j4];
}
"#;

const SIXTEEN_F32: &str = "fn keys16(at: u32) -> array<vec4<f32>, 4> {
    return array<vec4<f32>, 4>(x1[at * 4u], x1[at * 4u + 1u], x1[at * 4u + 2u], x1[at * 4u + 3u]);
}
fn values16(at: u32) -> array<vec4<f32>, 4> {
    return array<vec4<f32>, 4>(x2[at * 4u], x2[at * 4u + 1u], x2[at * 4u + 2u], x2[at * 4u + 3u]);
}";

// Two words of four integers each hold 16 numbers: of every eight, the first four are the low halves of a
// word's four integers and the next four the high halves, so unpacking is two shifts and no shuffling.
const SIXTEEN_I16: &str = "fn unpack(a: vec4<i32>, b: vec4<i32>) -> array<vec4<f32>, 4> {
    let s = vec4<u32>(16u);
    return array<vec4<f32>, 4>(vec4<f32>((a << s) >> s), vec4<f32>(a >> s), vec4<f32>((b << s) >> s), vec4<f32>(b >> s));
}
fn keys16(at: u32) -> array<vec4<f32>, 4> { return unpack(x1[at * 2u], x1[at * 2u + 1u]); }
fn values16(at: u32) -> array<vec4<f32>, 4> { return unpack(x2[at * 2u], x2[at * 2u + 1u]); }";

/// Keys `[rows][d]` as the attention shader wants them: coordinate by coordinate, each padded to whole blocks
/// of 16 keys.
fn keys_f32(k: &[f32], rows: usize, d: usize) -> Vec<f32> {
    let per = rows.div_ceil(16) * 16;
    let mut out = vec![0.0f32; d * per];
    for (r, row) in k.chunks_exact(d).enumerate() {
        for j in 0..d {
            out[j * per + r] = row[j];
        }
    }
    out
}

/// One scale per coordinate that makes every number of `[rows][d]` fit a 16-bit integer.
fn scales_i16(x: &[f32], d: usize) -> Vec<f32> {
    let mut big = vec![0.0f32; d];
    for row in x.chunks_exact(d) {
        for j in 0..d {
            big[j] = big[j].max(row[j].abs());
        }
    }
    big.iter().map(|b| if *b > 0.0 { b / 32767.0 } else { 1.0 }).collect()
}

/// Where number `i` of a run goes in the packed form (see `SIXTEEN_I16`): the word, and how far to shift.
fn slot_i16(i: usize) -> (usize, u32) {
    (i / 8 * 4 + i % 4, if i % 8 >= 4 { 16 } else { 0 })
}

/// Keys `[rows][d]`, packed: coordinate by coordinate as in `keys_f32`, 16-bit. Returns the scales too.
fn keys_i16(k: &[f32], rows: usize, d: usize) -> (Vec<u32>, Vec<f32>) {
    let scale = scales_i16(k, d);
    let per = rows.div_ceil(16) * 8;
    let mut out = vec![0u32; d * per];
    for (r, row) in k.chunks_exact(d).enumerate() {
        let (w, sh) = slot_i16(r);
        for j in 0..d {
            out[j * per + w] |= (((row[j] / scale[j]).round() as i16 as u16) as u32) << sh;
        }
    }
    (out, scale)
}

/// Values `[rows][d]`, packed: row by row, 16-bit. Returns the scales too.
fn values_i16(v: &[f32], d: usize) -> (Vec<u32>, Vec<f32>) {
    let scale = scales_i16(v, d);
    let mut out = vec![0u32; v.len() / 2];
    for (row, dst) in v.chunks_exact(d).zip(out.chunks_exact_mut(d / 2)) {
        for j in 0..d {
            let (w, sh) = slot_i16(j);
            dst[w] |= (((row[j] / scale[j]).round() as i16 as u16) as u32) << sh;
        }
    }
    (out, scale)
}

/// Queries `[nq][d]` laid out `[coordinate][query]` (rounded up to a multiple of 4 queries), each multiplied by `scale[coordinate]`.
fn queries_by_four(q: &[f32], d: usize, scale: Option<&[f32]>) -> Vec<f32> {
    let nq4 = (q.len() / d).div_ceil(4);
    let mut out = vec![0.0f32; nq4 * d * 4];
    for (r, row) in q.chunks_exact(d).enumerate() {
        for j in 0..d {
            out[(j * nq4 + r / 4) * 4 + r % 4] = row[j] * scale.map_or(1.0, |s| s[j]);
        }
    }
    out
}

/// Print a line of timing if `BROWSERTABICL_TRACE` is set.
fn trace(line: impl FnOnce() -> String) {
    if std::env::var_os("BROWSERTABICL_TRACE").is_some() {
        eprintln!("{}", line());
    }
}

fn fail(what: &str, e: impl std::fmt::Display) -> io::Error {
    io::Error::other(format!("graphics card: {what}: {e}"))
}

/// A graphics card ready to do attention.
pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    stage: Stage,
    pool: Mutex<Pool>,
    /// Told whenever working memory is given back, for calls waiting in `reserve`.
    pool_freed: Condvar,
    /// The weights of the model last fitted with, and what identifies it.
    fit: Mutex<Option<((usize, u64), Arc<GpuWeights>)>>,
    /// Offsets into a buffer given to a shader must be a multiple of this.
    align: usize,
    /// The largest buffer a shader may be given, in bytes.
    max_binding: usize,
    /// The first thing the card complained of since a call last looked (out of memory, device lost).
    trouble: Arc<Mutex<Option<String>>>,
    /// The adapter's own description, for logs.
    pub name: String,
    /// True if the adapter is a program standing in for a card: correct, but slower than the CPU path.
    pub software: bool,
    /// Stages of fitting the card has done (each `Attend` call it took rather than declined), for callers that
    /// want to know whether it was used at all.
    pub stages: AtomicU64,
}

/// Working memory a card may use unless told otherwise (`Gpu::set_work_budget`): 1.5 GB. Fitting is at its
/// quickest with this much (its slabs of cells are 128 MB and a stage holds about a dozen of them); prediction
/// needs about 0.6 MB per row of a 100-column table and cuts larger batches into sub-batches that fit.
pub const WORK_BUDGET: usize = 1536 << 20;

/// Working buffers, between uses and in use: all of the card's memory that is not a fitted model or weights.
///
/// Making and releasing a few gigabytes of them on every call cost more than a tenth of a large prediction, so
/// they are kept from one call to the next. Three rules keep that from growing without limit:
///
/// * every call says beforehand how much working memory it will borrow at most (`Gpu::reserve`) and is made to
///   wait while the calls in progress have promised the rest of the budget away. A prediction too large for the
///   budget is cut into sub-batches of rows that fit; fitting sizes its batches of columns and rows to it;
/// * a buffer is made new only if no idle one of that size exists, and idle ones are released first whenever a
///   new one would take the total over the budget. Sizes are rounded up to four steps per doubling, so buffers
///   for nearly the same number of tokens (a row with and without its class tokens, say) are shared. What a
///   call promises is the sum of every buffer it asks for: one it gave back itself cannot be released before
///   its commands are handed to the card, so within a call only reuse, not release, makes room;
/// * a call releases the idle buffers that the call before it did not use (`new_call`), and `Gpu::release`
///   releases all idle ones.
///
/// So `bytes` never exceeds `budget`, whatever is asked of the card and from however many threads.
///
/// A buffer may be lent out again as soon as its `Loan` is dropped, even if the commands that use it have not
/// run yet: whoever borrows it next submits a set of commands that writes it before reading it and runs either
/// wholly before or wholly after, the card taking submissions one at a time. (A caller that spreads its work
/// over several submissions keeps its loans until the last one.)
struct Pool {
    /// Each with the number of the call that last used it, and, until that thread next submits its work, the
    /// thread that gave it back: a buffer that commands not yet handed to the card refer to must not be
    /// released (it may be lent again, see below).
    free: Vec<(wgpu::Buffer, u64, Option<std::thread::ThreadId>)>,
    call: u64,
    /// Bytes of working buffers in existence: idle, on loan, or waiting to be read back.
    bytes: usize,
    /// The most `bytes` has been.
    peak: usize,
    /// Bytes on loan or waiting to be read back, and the most that ever exceeded what was promised (0 unless a
    /// call's own estimate of its needs is wrong: the tests check this).
    out: usize,
    overdrawn: usize,
    /// Bytes promised to calls in progress, of `budget`.
    reserved: usize,
    budget: usize,
}

impl Pool {
    /// Count a buffer of `size` bytes that is about to be made. It must not take the total over the budget:
    /// idle buffers are released first, longest unused first.
    fn grow(&mut self, size: usize) {
        while self.bytes + size > self.budget {
            let Some(oldest) = (0..self.free.len()).filter(|i| self.free[*i].2.is_none()).min_by_key(|i| self.free[*i].1) else { break };
            let (b, _, _) = self.free.swap_remove(oldest);
            self.bytes -= b.size() as usize;
            b.destroy();
        }
        self.bytes += size;
        self.peak = self.peak.max(self.bytes);
    }
}

/// A working buffer on loan from the pool; it goes back when this is dropped.
struct Loan<'a> {
    buf: Option<wgpu::Buffer>,
    pool: &'a Mutex<Pool>,
}

impl std::ops::Deref for Loan<'_> {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &wgpu::Buffer {
        self.buf.as_ref().unwrap()
    }
}

impl Drop for Loan<'_> {
    fn drop(&mut self) {
        let mut pool = self.pool.lock().unwrap();
        let call = pool.call;
        let buf = self.buf.take().unwrap();
        pool.out -= buf.size() as usize;
        pool.free.push((buf, call, Some(std::thread::current().id())));
    }
}

/// A share of the working-memory budget promised to one call; given back when this is dropped.
struct Reserved<'a> {
    gpu: &'a Gpu,
    bytes: usize,
}

impl Reserved<'_> {
    /// Give back all but `keep` bytes (what stays is for results still waiting to be read back).
    fn shrink(&mut self, keep: usize) {
        if keep < self.bytes {
            self.gpu.pool.lock().unwrap().reserved -= self.bytes - keep;
            self.bytes = keep;
            self.gpu.pool_freed.notify_all();
        }
    }
}

impl Drop for Reserved<'_> {
    fn drop(&mut self) {
        self.shrink(0);
    }
}

/// A buffer that results are copied into so that they can be read back. It is not pooled (a buffer that can be
/// read from outside cannot be given to a shader), but it counts as working memory while it exists.
struct Back<'a> {
    buf: wgpu::Buffer,
    pool: &'a Mutex<Pool>,
}

impl Drop for Back<'_> {
    fn drop(&mut self) {
        self.buf.destroy();
        let mut pool = self.pool.lock().unwrap();
        pool.bytes -= self.buf.size() as usize;
        pool.out -= self.buf.size() as usize;
    }
}

/// Results on their way back from the card: submitted, not yet read (`Gpu::take`).
struct Pending<'a> {
    back: Back<'a>,
    done: Once<Result<(), wgpu::BufferAsyncError>>,
    /// The submission that fills `back`.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    index: wgpu::SubmissionIndex,
    floats: usize,
}

/// Bytes a read-back buffer for `floats` numbers takes.
fn back_bytes(floats: usize) -> usize {
    floats.max(4).div_ceil(4) * 16
}

fn too_big(what: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::OutOfMemory, format!("graphics card: {what}"))
}

/// The many-keys attention (`SHADER_ATTEND`) built for one way of storing keys and values, one head width and
/// one number of queries per invocation.
struct Att {
    pipe: wgpu::ComputePipeline,
    /// Queries per invocation.
    qn: usize,
    /// Numbers per head.
    hd: usize,
}

/// A buffer from a byte offset on, as a shader's input or output.
#[derive(Clone, Copy)]
struct At<'a>(&'a wgpu::Buffer, usize);

/// Graphics memory the attention may use for the results of its slices.
const PARTIAL_BUDGET: usize = 128 << 20;

/// How one attention call is cut up, and its working buffer.
struct AttWork<'a> {
    /// Queries per invocation.
    qn: usize,
    /// Groups of `qn` queries in all, and per block (one block unless the slices' results would not fit a buffer).
    np: usize,
    qb: usize,
    /// Blocks of 16 keys.
    nb: usize,
    /// Slices of the keys.
    parts: usize,
    partial: Loan<'a>,
}

impl Gpu {
    /// Find the most capable card and prepare it. An error (rather than a panic) if there is none.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> io::Result<Gpu> {
        crate::wait::block_on(Gpu::new_async())
    }

    /// `new` for a browser, where finding the card has to be waited for.
    pub async fn new_async() -> io::Result<Gpu> {
        let instance = wgpu::Instance::default();
        let adapter =
            instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() }).await.map_err(|e| fail("none found", e))?;
        let info = adapter.get_info();
        let limits = adapter.limits();
        let max_binding = (limits.max_storage_buffer_binding_size as usize).min(limits.max_buffer_size as usize);
        let align = limits.min_storage_buffer_offset_alignment as usize;
        let (device, queue) =
            adapter.request_device(&wgpu::DeviceDescriptor { label: Some("tabicl-anywhere"), required_limits: limits, ..Default::default() }).await.map_err(|e| fail("could not be opened", e))?;
        // What the card objects to (out of memory, most of all) is kept for the call in progress to report as
        // an error, instead of stopping the program.
        let trouble: Arc<Mutex<Option<String>>> = Arc::default();
        let slot = trouble.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            slot.lock().unwrap().get_or_insert(e.to_string());
        }));
        let slot = trouble.clone();
        device.set_device_lost_callback(move |reason, text| {
            // (the device being let go of at the end is not trouble)
            if !matches!(reason, wgpu::DeviceLostReason::Destroyed) {
                slot.lock().unwrap().get_or_insert(format!("the card was lost: {text}"));
            }
        });
        let stage = Stage::new(&device);
        let pool = Pool { free: Vec::new(), call: 0, bytes: 0, peak: 0, out: 0, overdrawn: 0, reserved: 0, budget: WORK_BUDGET };
        let software = info.device_type == wgpu::DeviceType::Cpu;
        Ok(Gpu {
            device,
            queue,
            stage,
            pool: Mutex::new(pool),
            pool_freed: Condvar::new(),
            fit: Mutex::default(),
            align,
            max_binding,
            trouble,
            name: format!("{} ({:?})", info.name, info.backend),
            software,
            stages: AtomicU64::new(0),
        })
    }

    /// Set how much working memory the card may use (see `Pool`). Idle buffers beyond it are released at once;
    /// calls already in progress finish with what they were promised.
    pub fn set_work_budget(&self, bytes: usize) {
        let mut pool = self.pool.lock().unwrap();
        pool.budget = bytes;
        let mut keep = Vec::new();
        while pool.bytes > bytes {
            let Some(f) = pool.free.pop() else { break };
            if f.2.is_some() {
                keep.push(f);
                continue;
            }
            pool.bytes -= f.0.size() as usize;
            f.0.destroy();
        }
        pool.free.extend(keep);
    }

    pub fn work_budget(&self) -> usize {
        self.pool.lock().unwrap().budget
    }

    /// Working memory on the card right now, and the most it has been since the last `release`, in bytes.
    pub fn working(&self) -> (usize, usize) {
        let pool = self.pool.lock().unwrap();
        (pool.bytes, pool.peak)
    }

    /// The most that calls ever borrowed beyond what they had said they would (0 if all is well; for tests).
    pub fn overdrawn(&self) -> usize {
        self.pool.lock().unwrap().overdrawn
    }

    /// Give back everything the card holds that is not in use this instant: the idle working buffers and the
    /// weights kept from the last fit. Fitted models in a `VramTier` are not touched (see `VramTier::evict`).
    pub fn release(&self) {
        self.fit.lock().unwrap().take();
        let mut pool = self.pool.lock().unwrap();
        let (gone, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut pool.free).into_iter().partition(|f| f.2.is_none());
        pool.free = kept;
        for (b, _, _) in gone {
            pool.bytes -= b.size() as usize;
            b.destroy();
        }
        pool.peak = pool.bytes;
    }

    /// An error if the card has objected to something since this was last asked.
    fn check(&self) -> io::Result<()> {
        match self.trouble.lock().unwrap().take() {
            Some(text) => Err(fail("the card reported", text)),
            None => Ok(()),
        }
    }

    /// One step: `pipe` over `count` invocations with the stage's bindings (two parameter blocks, weights,
    /// three inputs, one output).
    fn step(&self, pass: &mut wgpu::ComputePass, pipe: &wgpu::ComputePipeline, bufs: [At; 7], count: usize) {
        let entries: Vec<_> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: b.0, offset: b.1 as u64, size: None }) })
            .collect();
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.stage.layout, entries: &entries });
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &group, &[]);
        // rows of 32768 workgroups (2097152 invocations), as many rows as needed
        let groups = count.div_ceil(64);
        pass.dispatch_workgroups(groups.min(32768) as u32, groups.div_ceil(32768) as u32, 1);
    }

    /// The bytes a working buffer for `bytes` is really given: the next of four steps per doubling, so that
    /// nearly equal needs share buffers (and never more than a shader may be given at once).
    fn bucket(&self, bytes: usize) -> usize {
        let need = bytes.max(64).div_ceil(16) * 16;
        let step = (need.next_power_of_two() / 8).max(16);
        let round = need.div_ceil(step) * step;
        if round > self.max_binding {
            need
        } else {
            round
        }
    }

    /// Bytes `storage(floats)` takes from the pool.
    fn bk(&self, floats: usize) -> usize {
        self.bucket(floats * 4)
    }

    /// Bytes `work(rows, d, ff, query_rows)` takes from the pool.
    fn work_need(&self, rows: usize, d: usize, ff: usize, query_rows: usize) -> usize {
        self.bk(rows * 4) + 4 * self.bk(rows * d) + self.bk(query_rows * d) + self.bk(rows * ff)
    }

    /// Promise `bytes` of the working-memory budget to the caller, waiting while calls in progress hold too
    /// much of it. An out-of-memory error if `bytes` is more than the whole budget.
    fn reserve(&self, bytes: usize) -> io::Result<Reserved<'_>> {
        let mut pool = self.pool.lock().unwrap();
        loop {
            if bytes > pool.budget {
                return Err(too_big(format!("this needs {} MB of working memory at once and the budget is {} MB", bytes >> 20, pool.budget >> 20)));
            }
            if pool.reserved + bytes <= pool.budget {
                pool.reserved += bytes;
                return Ok(Reserved { gpu: self, bytes });
            }
            // A browser cannot wait here (and has no second thread that could be in the way): two calls were
            // started at once on a card whose working memory holds one of them.
            #[cfg(target_arch = "wasm32")]
            return Err(fail("busy", "another call is using its working memory"));
            #[cfg(not(target_arch = "wasm32"))]
            {
                pool = self.pool_freed.wait(pool).unwrap();
            }
        }
    }

    /// As `reserve`, but `None` instead of waiting.
    fn try_reserve(&self, bytes: usize) -> io::Result<Option<Reserved<'_>>> {
        let mut pool = self.pool.lock().unwrap();
        if bytes > pool.budget {
            return Err(too_big(format!("this needs {} MB of working memory at once and the budget is {} MB", bytes >> 20, pool.budget >> 20)));
        }
        if pool.reserved + bytes > pool.budget {
            return Ok(None);
        }
        pool.reserved += bytes;
        Ok(Some(Reserved { gpu: self, bytes }))
    }

    /// A working buffer for `floats` numbers: an idle one of that size from the pool if there is one, otherwise
    /// new. What it holds to begin with is whatever its last user left there.
    fn storage(&self, floats: usize) -> Loan<'_> {
        let size = self.bk(floats);
        let mut pool = self.pool.lock().unwrap();
        let buf = match pool.free.iter().position(|(b, _, _)| b.size() as usize == size) {
            Some(i) => pool.free.swap_remove(i).0,
            None => {
                pool.grow(size);
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: size as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            }
        };
        pool.out += size;
        pool.overdrawn = pool.overdrawn.max(pool.out.saturating_sub(pool.reserved));
        Loan { buf: Some(buf), pool: &self.pool }
    }

    /// Called as a call starts: release the idle buffers that the call before it did not use, so that what is
    /// kept between calls is what calls are actually using.
    fn new_call(&self) {
        let mut pool = self.pool.lock().unwrap();
        pool.call += 1;
        let now = pool.call;
        let mut freed = 0;
        pool.free.retain(|(b, used, waiting)| {
            let keep = *used + 1 >= now || waiting.is_some();
            if !keep {
                freed += b.size() as usize;
                b.destroy();
            }
            keep
        });
        pool.bytes -= freed;
    }

    /// A working buffer holding `data`.
    fn fill(&self, data: &[f32]) -> Loan<'_> {
        let b = self.storage(data.len());
        self.queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
        b
    }

    fn upload<T: bytemuck::Pod>(&self, data: &[T]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE })
    }

    /// A buffer to read `floats` numbers back through.
    fn back(&self, floats: usize) -> Back<'_> {
        let size = back_bytes(floats);
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("back"),
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut pool = self.pool.lock().unwrap();
        pool.grow(size);
        pool.out += size;
        pool.overdrawn = pool.overdrawn.max(pool.out.saturating_sub(pool.reserved));
        Back { buf, pool: &self.pool }
    }

    /// How the attention of `nq` queries over `n` keys is cut up, for a shader taking `qn` queries per
    /// invocation with heads `hd` numbers wide: (groups of queries, groups per block, blocks of 16 keys, slices
    /// of the keys, numbers in the working buffer for the slices' results).
    fn att_cut(&self, qn: usize, hd: usize, nq: usize, n: usize, heads: usize) -> (usize, usize, usize, usize, usize) {
        let np = nq.div_ceil(4) * 4 / qn;
        let nb = n.div_ceil(16);
        // bytes one slice's result takes for every group of queries
        let rec = (qn * (hd / 4) + qn.div_ceil(2)) * 16;
        let budget = PARTIAL_BUDGET.min(self.max_binding);
        // enough slices of the keys to keep the whole card busy, each at least a few blocks long, as far as
        // their results fit the budget
        let parts = ((1 << 18) / (np * heads)).min(budget / (np * heads * rec)).clamp(1, (nb / 3).max(1));
        // with very many queries even one slice does not fit: then the queries go in blocks
        let mut qb = (budget / (heads * parts * rec)).clamp(1, np);
        if qb >= 64 {
            qb -= qb % 64;
        }
        (np, qb, nb, parts, np * heads * parts * rec / 4)
    }

    /// Decide how to cut up the attention of `nq` queries over `n` keys with `at`, and make its working buffer.
    fn att_work(&self, at: &Att, nq: usize, n: usize, heads: usize) -> AttWork<'_> {
        let (np, qb, nb, parts, floats) = self.att_cut(at.qn, at.hd, nq, n, heads);
        AttWork { qn: at.qn, np, qb, nb, parts, partial: self.storage(floats) }
    }

    /// Working memory `attend_buffers` borrows for `nq` queries over `n` keys.
    fn att_need(&self, qn: usize, hd: usize, nq: usize, n: usize, d: usize, heads: usize) -> usize {
        self.bk(self.att_cut(qn, hd, nq, n, heads).4) + self.bk(nq.div_ceil(4) * 4 * d) + self.bk(nq * d) + back_bytes(nq * d)
    }

    /// The attention passes for blocks `blocks` (of `wk.qb` groups of queries each) of the queries in `qt`
    /// (`[coordinate][query]`, the keys' scales folded in) against `n` keys `k` and values `v`, stored the way
    /// `at` was built for. Outputs go to `out` (`[rows][d]`), times row `layer` of `vscale`. The merge that
    /// follows the last block covers all rows, so `blocks` must end with the last one for `out` to be whole.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self, pass: &mut wgpu::ComputePass, at: &Att, wk: &AttWork, blocks: std::ops::Range<usize>, rows: usize, n: usize, d: usize, heads: usize, layer: usize, qt: &wgpu::Buffer, k: At, v: At,
        vscale: &wgpu::Buffer, out: At, skip: &str,
    ) {
        let st = &self.stage;
        let no = At(&st.dummy, 0);
        let params = |q0: usize, qb: usize| {
            let p = [wk.np, n, wk.parts, heads, d, q0, qb, wk.nb, layer, rows, wk.qn, 0].map(|v| v as u32);
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("attention"), contents: bytemuck::cast_slice(&p), usage: wgpu::BufferUsages::UNIFORM })
        };
        let last = blocks.end * wk.qb >= wk.np;
        for b in blocks {
            let q0 = b * wk.qb;
            let qb = wk.qb.min(wk.np - q0);
            if !skip.contains('a') {
                self.step(pass, &at.pipe, [At(&params(q0, qb), 0), At(&st.dummy_u, 0), no, At(qt, 0), k, v, At(&wk.partial, 0)], wk.parts * heads * qb);
            }
        }
        if last && !skip.contains('m') {
            self.step(pass, &st.merge, [At(&params(0, wk.np), 0), At(&st.dummy_u, 0), no, At(&wk.partial, 0), no, At(vscale, 0), out], rows * (d / 4));
        }
    }

    /// Hand `enc` to the card. The buffers this thread gave back while recording may be released from now on.
    fn submit(&self, enc: wgpu::CommandEncoder) -> wgpu::SubmissionIndex {
        let index = self.queue.submit([enc.finish()]);
        self.settled();
        index
    }

    /// Everything this thread has recorded is with the card: the buffers it gave back may be released. Called
    /// on submitting, and at the end of each call (whose loans go back after its last submission).
    fn settled(&self) {
        let me = Some(std::thread::current().id());
        for f in self.pool.lock().unwrap().free.iter_mut().filter(|f| f.2 == me) {
            f.2 = None;
        }
    }

    /// Hand `enc` to the card and ask for `back` (which `enc` copies results into) to be made readable when
    /// that work is done. Nothing waits here.
    fn send<'a>(&'a self, enc: wgpu::CommandEncoder, back: Back<'a>, floats: usize) -> Pending<'a> {
        let index = self.submit(enc);
        let (tx, done) = once();
        back.buf.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r));
        Pending { back, done, index, floats }
    }

    /// The numbers of `p`, once the card has them. This is the one place anything waits for the card (see the
    /// top of the file): natively by stopping right here until the submission is through, in a browser by
    /// handing control back to it until it calls with the result.
    async fn take(&self, p: Pending<'_>) -> io::Result<Vec<f32>> {
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::PollType::Wait { submission_index: Some(p.index.clone()), timeout: None }).map_err(|e| fail("stopped responding", e))?;
        let got = p.done.await;
        self.check()?;
        got.ok_or_else(|| fail("no answer", "the request was dropped"))?.map_err(|e| fail("could not read the result back", e))?;
        let view = p.back.buf.slice(..).get_mapped_range().map_err(|e| fail("could not read the result back", e))?;
        let mut out: Vec<f32> = bytemuck::pod_collect_to_vec(&view[..]);
        out.truncate(p.floats);
        drop(view);
        p.back.buf.unmap();
        Ok(out)
    }

    /// Attention outputs `[nq][d]` for queries `q` (`[nq][d]`, already scaled) against keys and values already
    /// on the card in the attention shader's layout (`n` of them; `kscale` and row `layer` of `vscale` are their
    /// scales if they are packed). The caller has reserved `att_need` of working memory.
    #[allow(clippy::too_many_arguments)]
    async fn attend_buffers(
        &self, q: &[f32], k: &wgpu::Buffer, v: &wgpu::Buffer, kscale: Option<&[f32]>, vscale: &wgpu::Buffer, layer: usize, n: usize, d: usize, heads: usize,
    ) -> io::Result<Vec<f32>> {
        let nq = q.len() / d;
        let at = if kscale.is_some() { &self.stage.att_i16 } else { &self.stage.att_f32 };
        assert!(d / heads == at.hd, "the shader is built for heads {} numbers wide", at.hd);
        let pending = {
            let wk = self.att_work(at, nq, n, heads);
            let qt = self.fill(&queries_by_four(q, d, kscale));
            let ob = self.storage(nq * d);
            let bytes = (nq * d * 4) as u64;
            let back = self.back(nq * d);
            // a few blocks of queries per submission, so that no single one runs long enough to trip a driver watchdog
            let blocks = wk.np.div_ceil(wk.qb);
            for b0 in (0..blocks).step_by(8) {
                let mut enc = self.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    self.attention(&mut pass, at, &wk, b0..(b0 + 8).min(blocks), nq, n, d, heads, layer, &qt, At(k, 0), At(v, 0), vscale, At(&ob, 0), "");
                }
                self.submit(enc);
            }
            let mut enc = self.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&ob, 0, &back.buf, 0, bytes);
            self.send(enc, back, nq * d)
        };
        self.settled();
        self.take(pending).await
    }

    /// Attention outputs `[nq][d]` for queries `q` (`[nq][d]`, already scaled) over keys and values `[n][d]`,
    /// in full precision.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn attend_rows(&self, q: &[f32], k: &[f32], v: &[f32], d: usize, heads: usize) -> io::Result<Vec<f32>> {
        crate::wait::block_on(self.attend_rows_async(q, k, v, d, heads))
    }

    /// `attend_rows` in the form a browser can wait for.
    pub async fn attend_rows_async(&self, q: &[f32], k: &[f32], v: &[f32], d: usize, heads: usize) -> io::Result<Vec<f32>> {
        let n = k.len() / d;
        let t0 = web_time::Instant::now();
        let at = &self.stage.att_f32;
        let _hold = self.reserve(self.bk(n.div_ceil(16) * 16 * d) + self.bk(n * d) + self.att_need(at.qn, at.hd, q.len() / d, n, d, heads))?;
        let (kb, vb, ones) = (self.fill(&keys_f32(k, n, d)), self.fill(v), self.upload(&vec![1.0f32; d]));
        let out = self.attend_buffers(q, &kb, &vb, None, &ones, 0, n, d, heads).await?;
        trace(|| format!("gpu: attention of {} rows over {n}: {:.3} s", q.len() / d, t0.elapsed().as_secs_f64()));
        Ok(out)
    }
}

/// Graphics memory one working buffer of the fitting stages may take (the widest are twice this), if the
/// working-memory budget allows that much.
const FIT_BUDGET: usize = 128 << 20;

/// Results of several buffers on their way back in one read (`Gpu::fetch`), and how long each is.
struct Fetch<'a> {
    pending: Pending<'a>,
    lens: Vec<usize>,
}

impl Gpu {
    /// The weights of `m` on the card, for fitting: uploaded the first time, kept while the same model is used.
    fn fit_weights(&self, m: &Model) -> Arc<GpuWeights> {
        // the same model: the same place in memory and the same input weights
        let key = (m as *const Model as usize, m.in_w.iter().fold(0u64, |a, v| a.wrapping_mul(31).wrapping_add(v.to_bits() as u64)));
        let mut slot = self.fit.lock().unwrap();
        match &*slot {
            Some((k, gw)) if *k == key => gw.clone(),
            _ => {
                let t = web_time::Instant::now();
                let gw = Arc::new(GpuWeights::new(self, m));
                trace(|| format!("gpu: weights uploaded and fitting shaders built in {:.0} ms", t.elapsed().as_secs_f64() * 1e3));
                *slot = Some((key, gw.clone()));
                gw
            }
        }
    }

    /// The weights of `m` on the card, for fitting and for prediction alike: uploaded the first time, kept while
    /// the same model is used (until `release`).
    pub fn weights(&self, m: &Model) -> Arc<GpuWeights> {
        self.fit_weights(m)
    }

    /// How fast this card does the in-context attention, in multiply-adds per second: the best of repeated
    /// runs of 256 queries over 2,048 keys for about `ms` milliseconds. For estimating how long work will take.
    pub async fn calibrate(&self, ms: f64) -> io::Result<f64> {
        let (nq, n, d, heads) = (256, 2048, 512, 8);
        let mut s = 0x2545F4914F6CDD1Du64;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let (q, k, v): (Vec<f32>, Vec<f32>, Vec<f32>) = ((0..nq * d).map(|_| rnd()).collect(), (0..n * d).map(|_| rnd()).collect(), (0..n * d).map(|_| rnd()).collect());
        let t0 = web_time::Instant::now();
        let mut best = 0.0f64;
        // the first run includes setting up; at least three runs, then as many as the time allows
        for i in 0.. {
            let t = web_time::Instant::now();
            self.attend_rows_async(&q, &k, &v, d, heads).await?;
            best = best.max((2 * nq * n * d) as f64 / t.elapsed().as_secs_f64().max(1e-6));
            if i >= 2 && t0.elapsed().as_secs_f64() * 1e3 > ms {
                break;
            }
        }
        Ok(best)
    }

    /// Submit `enc` after adding copies of the first `floats` numbers of each buffer in `what`; `fetched` waits
    /// for them.
    fn fetch(&self, mut enc: wgpu::CommandEncoder, what: &[(&wgpu::Buffer, usize)]) -> Fetch<'_> {
        let total: usize = what.iter().map(|(_, n)| n.div_ceil(4) * 4).sum();
        let back = self.back(total);
        let mut at = 0;
        for (b, n) in what {
            enc.copy_buffer_to_buffer(b, 0, &back.buf, (at * 4) as u64, (n.div_ceil(4) * 16) as u64);
            at += n.div_ceil(4) * 4;
        }
        Fetch { pending: self.send(enc, back, total), lens: what.iter().map(|(_, n)| *n).collect() }
    }

    async fn fetched(&self, f: Fetch<'_>) -> io::Result<Vec<Vec<f32>>> {
        let all = self.take(f.pending).await?;
        let mut at = 0;
        Ok(f.lens
            .iter()
            .map(|n| {
                let part = all[at..at + n].to_vec();
                at += n.div_ceil(4) * 4;
                part
            })
            .collect())
    }

    /// Working memory the column stage of fitting borrows for `g` feature groups of `rows` rows.
    fn columns_need(&self, m: &Model, rows: usize, g: usize) -> usize {
        let (e, inds, nb) = (m.e, m.inds, m.col.len());
        let (n, toks, per) = (rows * g, g * inds, rows.div_ceil(16) * 16);
        let partial = self.att_cut(4, e / m.col_heads, inds, rows, m.col_heads).4;
        2 * self.bk(n * e)
            + self.bk(g * e * per)
            + self.work_need(n, e, 2 * e, n)
            + self.work_need(toks, e, 2 * e, toks)
            + self.bk(partial)
            + nb * (3 * self.bk(toks * e) + self.bk(inds * e))
            + back_bytes(n * e + 2 * nb * toks * e)
    }

    /// Working memory the row stage borrows for `n` rows of `features` cells (wherever it is recorded).
    fn row_stage_need(&self, e: usize, cls: usize, n: usize, features: usize) -> usize {
        let toks = n * (features + cls);
        self.work_need(toks, e, 2 * e, toks) + 3 * self.bk(toks * e) + self.bk(n * cls * e)
    }

    /// Working memory the row stage of fitting borrows for `n` rows.
    fn rows_need(&self, m: &Model, n: usize, features: usize) -> usize {
        self.bk(n * features * m.e) + self.bk(n * m.d) + self.row_stage_need(m.e, m.cls, n, features) + back_bytes(n * m.d)
    }

    /// Working memory the in-context stage of fitting borrows for a problem of `rows` rows.
    fn in_context_need(&self, m: &Model, rows: usize) -> usize {
        let (d, ff, heads, at) = (m.d, m.dec0b.len(), m.icl_heads, &self.stage.att_f32);
        let (np, _, _, _, partial) = self.att_cut(at.qn, at.hd, rows, rows, heads);
        // two layers' keys and values are on their way back at any time
        self.bk(partial) + 3 * self.bk(rows * d) + self.work_need(rows, d, ff, np * at.qn) + self.bk(d * rows.div_ceil(16) * 16) + self.bk(rows * heads * 64) + 2 * back_bytes(2 * rows * d)
    }

    /// Working memory `attend` (one layer's attention for all `rows` rows of a problem) borrows.
    fn attend_need(&self, rows: usize, d: usize) -> usize {
        let at = &self.stage.att_f32;
        self.bk(rows.div_ceil(16) * 16 * d) + self.bk(rows * d) + self.att_need(at.qn, at.hd, rows, rows, d, d / at.hd)
    }

    /// The most of `total` items a stage may take at a time: as many as `need` says fit the working-memory
    /// budget, and no more than `cap`. 0 if not even one does.
    fn batch(&self, total_cap: usize, need: impl Fn(usize) -> usize) -> usize {
        let budget = self.work_budget();
        if need(1) > budget {
            return 0;
        }
        // need grows with the batch: the largest that fits is found by halving
        let (mut lo, mut hi) = (1, total_cap.max(1));
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if need(mid) <= budget {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        lo
    }
}

impl Attend for Gpu {
    fn attend<'a>(&'a self, q: &'a [f32], kv: &'a [f32], rows: usize, chunk: usize, d: usize, heads: usize) -> Later<'a, io::Result<Vec<f32>>> {
        Box::pin(async move {
            // the cache keeps each chunk coordinate by coordinate; here the keys go coordinate by coordinate over
            // all rows and the values row by row
            let per = rows.div_ceil(16) * 16;
            let (mut k, mut v) = (vec![0.0f32; per * d], vec![0.0f32; rows * d]);
            for (ci, c) in kv.chunks(2 * d * chunk).enumerate() {
                let rc = c.len() / (2 * d);
                let (kt, vt) = c.split_at(d * rc);
                for j in 0..d {
                    k[j * per + ci * chunk..j * per + ci * chunk + rc].copy_from_slice(&kt[j * rc..(j + 1) * rc]);
                }
                for r in 0..rc {
                    let at = (ci * chunk + r) * d;
                    for j in 0..d {
                        v[at + j] = vt[j * rc + r];
                    }
                }
            }
            let _hold = self.reserve(self.attend_need(rows, d))?;
            self.stages.fetch_add(1, Relaxed);
            let (kb, vb, ones) = (self.fill(&k), self.fill(&v), self.upload(&vec![1.0f32; d]));
            self.attend_buffers(q, &kb, &vb, None, &ones, 0, rows, d, heads).await
        })
    }

    fn attend_fits(&self, rows: usize, d: usize) -> bool {
        d / self.stage.att_f32.hd > 0 && rows * d * 4 <= self.max_binding && self.attend_need(rows, d) <= self.work_budget()
    }

    fn column_batch(&self, m: &Model, rows: usize) -> usize {
        let per_group = rows * m.e * 4;
        let hd = m.e / m.col_heads;
        // what the shaders and the buffer offsets below assume
        if m.group != 3 || !hd.is_multiple_of(16) || !(m.e * 64).is_multiple_of(self.align) || 2 * per_group > self.max_binding {
            return 0;
        }
        self.batch((FIT_BUDGET.min(self.max_binding / 2) / per_group).max(1), |g| self.columns_need(m, rows, g))
    }

    fn columns<'a>(&'a self, m: &'a Model, cells: &'a mut [f32], rows: usize, g: usize, ind_q: &'a [Vec<f32>], rec: &'a mut [f32]) -> Later<'a, io::Result<()>> {
        Box::pin(async move {
            let gw = self.fit_weights(m);
            let t = web_time::Instant::now();
            let (e, inds, heads) = (gw.e, gw.inds, gw.col_heads);
            let (n, toks) = (rows * g, g * inds);
            let blocks = gw.ind.len();
            let hold = self.reserve(self.columns_need(m, rows, g))?;
            self.stages.fetch_add(1, Relaxed);
            self.new_call();
            let fetch = {
                let (st, at) = (&self.stage, gw.col_att.as_ref().expect("a model loaded for fitting"));
                let no = &st.dummy;
                let skip = Skip(String::new());
                // keys per coordinate once padded to whole blocks of 16
                let per = rows.div_ceil(16) * 16;
                let ct = self.fill(cells);
                let (w, wi) = (self.work(n, e, 2 * e, n), self.work(toks, e, 2 * e, toks));
                let (kc, vr) = (self.storage(g * e * per), self.storage(n * e));
                let wk = self.att_work(at, inds, rows, heads);
                let ones = self.upload(&vec![1.0f32; e]);
                // the cells as tokens (and, for the few-keys attention, `rows` queries per set of keys, g apart);
                // the same cells as `rows` keys in each of g sets; the inducing points of all groups as tokens
                let p = self.params([n, inds, rows, heads, e, 2 * e, 0, g, 1, 0, g, 1]);
                let pt = self.params([rows, per, g, heads, e, 2 * e, 0, 1, 1, 0, 0, 0]);
                let pi = self.params([toks, inds, 1, heads, e, 2 * e, 0, 1, 1, 0, 0, 0]);
                // per block: the inducing points (every group starts from the same ones), their queries, and the keys
                // and values they end up with
                let hid: Vec<_> = gw.ind.iter().map(|ind| self.fill(&ind.repeat(g))).collect();
                let qt: Vec<_> = ind_q.iter().map(|q| self.fill(&queries_by_four(q, e, None))).collect();
                let kv2: Vec<_> = gw.ind.iter().map(|_| (self.storage(toks * e), self.storage(toks * e))).collect();
                let mut enc = self.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    let pass = &mut pass;
                    for (bi, (a1, a2)) in gw.col1.iter().zip(&gw.col).enumerate() {
                        // every cell's key and value, then each group's laid out as one set of keys
                        self.run(pass, &skip, "", &st.stats, &p.sel[0], a1, [&ct, no, no], &w.st, n);
                        self.run(pass, &skip, "", &st.norm1, &p.sel[0], a1, [&ct, &w.st, no], &w.xn, n * e / 4);
                        self.lin(pass, &skip, "", &p, 1, a1, &w.xn, no, &w.q0, n, e);
                        self.lin(pass, &skip, "", &p, 2, a1, &w.xn, no, &w.att, n, e);
                        self.run(pass, &skip, "", &st.turnk, &pt.sel[0], a1, [&w.q0, no, no], &kc, g * e * per);
                        self.run(pass, &skip, "", &st.turnv, &pt.sel[0], a1, [&w.att, no, no], &vr, n * e);
                        // the inducing points attend to their group's cells
                        for gi in 0..g {
                            self.attention(
                                pass,
                                at,
                                &wk,
                                0..wk.np.div_ceil(wk.qb),
                                inds,
                                rows,
                                e,
                                heads,
                                0,
                                &qt[bi],
                                At(&kc, gi * e * per * 4),
                                At(&vr, gi * rows * e * 4),
                                &ones,
                                At(&wi.att, gi * inds * e * 4),
                                "",
                            );
                        }
                        self.tail(pass, &skip, "", &pi, a1, &hid[bi], &wi, toks, e, 2 * e);
                        // their keys and values for the second attention: this is what prediction reuses
                        let (k2, v2) = &kv2[bi];
                        self.run(pass, &skip, "", &st.stats, &pi.sel[0], a2, [&hid[bi], no, no], &wi.st, toks);
                        self.run(pass, &skip, "", &st.norm1, &pi.sel[0], a2, [&hid[bi], &wi.st, no], &wi.xn, toks * e / 4);
                        self.lin(pass, &skip, "", &pi, 1, a2, &wi.xn, no, k2, toks, e);
                        self.lin(pass, &skip, "", &pi, 2, a2, &wi.xn, no, v2, toks, e);
                        // and every cell attends to its group's inducing points, as in prediction
                        self.column_block(pass, &skip, &gw, a2, k2, v2, rows, g, &ct, &w, &p);
                    }
                }
                let mut what: Vec<(&wgpu::Buffer, usize)> = vec![(&ct, n * e)];
                for (k2, v2) in &kv2 {
                    what.push((k2, toks * e));
                    what.push((v2, toks * e));
                }
                self.fetch(enc, &what)
            };
            self.settled();
            let recorded = t.elapsed().as_secs_f64() * 1e3;
            let got = self.fetched(fetch).await?;
            drop(hold);
            trace(|| format!("gpu: column stage for {g} groups: recorded {recorded:.0} ms, answer back {:.0} ms", t.elapsed().as_secs_f64() * 1e3));
            cells.copy_from_slice(&got[0]);
            // the record keeps keys and values coordinate by coordinate, the keys without the 1 / sqrt(head width)
            let per_block = 2 * e * inds;
            for (bi, kv) in got[1..].chunks_exact(2).enumerate() {
                for gi in 0..g {
                    let (rk, rv) = rec[(gi * blocks + bi) * per_block..][..per_block].split_at_mut(e * inds);
                    for i in 0..inds {
                        for j in 0..e {
                            rk[j * inds + i] = kv[0][(gi * inds + i) * e + j] / gw.col_inv;
                            rv[j * inds + i] = kv[1][(gi * inds + i) * e + j];
                        }
                    }
                }
            }
            Ok(())
        })
    }

    fn row_batch(&self, m: &Model, features: usize) -> usize {
        let per_row = (features + m.cls) * m.e * 4;
        if m.group != 3 || features + m.cls > 4096 || 2 * per_row > self.max_binding {
            return 0;
        }
        self.batch((FIT_BUDGET.min(self.max_binding / 2) / per_row).max(1), |n| self.rows_need(m, n, features))
    }

    fn rows<'a>(&'a self, m: &'a Model, cells: &'a [f32], n: usize, reps: &'a mut [f32]) -> Later<'a, io::Result<()>> {
        Box::pin(async move {
            let gw = self.fit_weights(m);
            let features = cells.len() / (n * gw.e);
            let t = web_time::Instant::now();
            let hold = self.reserve(self.rows_need(m, n, features))?;
            self.stages.fetch_add(1, Relaxed);
            self.new_call();
            let fetch = {
                let (ct, hb) = (self.fill(cells), self.storage(n * gw.d));
                let mut enc = self.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    self.row_stage(&mut pass, &Skip(String::new()), &gw, n, features, &ct, &hb);
                }
                self.fetch(enc, &[(&hb, n * gw.d)])
            };
            self.settled();
            let recorded = t.elapsed().as_secs_f64() * 1e3;
            reps.copy_from_slice(&self.fetched(fetch).await?[0]);
            drop(hold);
            trace(|| format!("gpu: row stage for {n} rows: recorded {recorded:.0} ms, answer back {:.0} ms", t.elapsed().as_secs_f64() * 1e3));
            Ok(())
        })
    }

    fn in_context<'a>(&'a self, m: &'a Model, h: &'a [f32], rows: usize, scales: &'a [f32], sink: &'a mut dyn FnMut(usize, &[f32], &[f32]) -> io::Result<()>) -> Later<'a, io::Result<bool>> {
        Box::pin(async move {
            let gw = self.fit_weights(m);
            let (d, ff, heads, layers) = (gw.d, gw.ff, gw.heads, gw.layers);
            let (st, at) = (&self.stage, &self.stage.att_f32);
            // one layer's keys (or values) must fit a buffer and the stage's working set the budget; larger
            // tables keep to the CPU with `attend` for as long as that fits
            if d / heads != at.hd || rows * d * 8 > self.max_binding.min(1 << 30) || self.in_context_need(m, rows) > self.work_budget() {
                return Ok(false);
            }
            let _hold = self.reserve(self.in_context_need(m, rows))?;
            self.stages.fetch_add(1, Relaxed);
            self.new_call();
            let no = &st.dummy;
            let skip = Skip(String::new());
            let per = rows.div_ceil(16) * 16;
            let wk = self.att_work(at, rows, rows, heads);
            let hb = self.fill(h);
            let w = self.work(rows, d, ff, wk.np * wk.qn);
            let (kr, vr, kc, hid) = (self.storage(rows * d), self.storage(rows * d), self.storage(d * per), self.storage(rows * heads * 64));
            let (sc, ones) = (self.upload(scales), self.upload(&vec![1.0f32; d]));
            let cu = self.params([rows, rows, 1, heads, d, ff, 0, 1, 1, 0, 0, 0]);
            let pt = self.params([rows, per, 1, heads, d, ff, 0, 1, 1, 0, 0, 0]);
            // The card is given each layer as soon as the one before it has been handed over, and works on it
            // while the caller packs and writes the keys and values of the layer before: neither waits for the
            // other. (A copy of a layer's keys and values is taken on the card as part of that layer's work, so
            // the next layer is free to overwrite them.)
            let mut before: Option<(usize, Fetch, web_time::Instant)> = None;
            for l in 0..layers {
                let b = &gw.blocks[l];
                let mut enc = self.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    let pass = &mut pass;
                    // every row's query, key and value
                    self.head(pass, &skip, "", &cu, b, &hb, &w, rows, d);
                    self.lin(pass, &skip, "", &cu, 1, b, &w.xn, no, &kr, rows, d);
                    self.lin(pass, &skip, "", &cu, 2, b, &w.xn, no, &vr, rows, d);
                    // every row attends to all rows (the last layer's outputs for the training rows are
                    // never used, so it stops at its keys and values)
                    if l + 1 < layers {
                        self.run(pass, &skip, "", &st.sshid, &cu.sel[0], b, [&w.q0, no, no], &hid, rows * heads * 64);
                        self.run(pass, &skip, "", &st.ssout, &cu.sel[0], b, [&hid, &w.q0, &sc], &w.q, rows * d);
                        self.run(pass, &skip, "", &st.turnk, &pt.sel[0], b, [&kr, no, no], &kc, d * per);
                        self.attention(pass, at, &wk, 0..wk.np.div_ceil(wk.qb), rows, rows, d, heads, 0, &w.q, At(&kc, 0), At(&vr, 0), &ones, At(&w.att, 0), "");
                        self.tail(pass, &skip, "", &cu, b, &hb, &w, rows, d, ff);
                    }
                }
                let sent = (l, self.fetch(enc, &[(&kr, rows * d), (&vr, rows * d)]), web_time::Instant::now());
                if let Some((lb, f, t)) = before.replace(sent) {
                    let got = self.fetched(f).await?;
                    trace(|| format!("gpu: in-context layer {lb}: {:.0} ms", t.elapsed().as_secs_f64() * 1e3));
                    sink(lb, &got[0], &got[1])?;
                }
            }
            drop((wk, hb, w, kr, vr, kc, hid));
            self.settled();
            if let Some((lb, f, t)) = before {
                let got = self.fetched(f).await?;
                trace(|| format!("gpu: in-context layer {lb}: {:.0} ms", t.elapsed().as_secs_f64() * 1e3));
                sink(lb, &got[0], &got[1])?;
            }
            Ok(true)
        })
    }
}

/// The pipelines of the whole-stage path and the binding layout they share.
struct Stage {
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    stats: wgpu::ComputePipeline,
    norm1: wgpu::ComputePipeline,
    norm2: wgpu::ComputePipeline,
    linear: wgpu::ComputePipeline,
    linear1: wgpu::ComputePipeline,
    sshid: wgpu::ComputePipeline,
    ssout: wgpu::ComputePipeline,
    /// The many-keys attention for the in-context stage, with keys and values as plain floats, and packed.
    att_f32: Att,
    att_i16: Att,
    merge: wgpu::ComputePipeline,
    ropeq: wgpu::ComputePipeline,
    ropek: wgpu::ComputePipeline,
    place: wgpu::ComputePipeline,
    cells: wgpu::ComputePipeline,
    first: wgpu::ComputePipeline,
    turnk: wgpu::ComputePipeline,
    turnv: wgpu::ComputePipeline,
    clsnorm: wgpu::ComputePipeline,
    dec: wgpu::ComputePipeline,
    dummy: wgpu::Buffer,
    dummy_u: wgpu::Buffer,
}

impl Stage {
    /// The many-keys attention for heads `hd` numbers wide, `qn` queries per invocation.
    fn att(device: &wgpu::Device, layout: &wgpu::PipelineLayout, packed: bool, hd: usize, qn: usize) -> Att {
        assert!(hd.is_multiple_of(16), "the many-keys attention is written for heads a multiple of 16 numbers wide");
        let (ty, sixteen) = if packed { ("vec4<i32>", SIXTEEN_I16) } else { ("vec4<f32>", SIXTEEN_F32) };
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("attend"), source: wgpu::ShaderSource::Wgsl(attend_source(qn, hd / 4, ty, sixteen).into()) });
        let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("attend"),
            layout: Some(layout),
            module: &module,
            entry_point: Some("attend"),
            compilation_options: Default::default(),
            cache: None,
        });
        Att { pipe, qn, hd }
    }

    /// The few-keys attention (`SHADER_FEW`) for heads `hd` numbers wide.
    fn few(&self, device: &wgpu::Device, hd: usize) -> wgpu::ComputePipeline {
        assert!(hd.is_multiple_of(4) && hd <= 64, "the few-keys attention is written for heads of up to 64 numbers, in fours");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("few-keys"), source: wgpu::ShaderSource::Wgsl(few_source(hd / 4).into()) });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("attend1"),
            layout: Some(&self.pipeline_layout),
            module: &module,
            entry_point: Some("attend1"),
            compilation_options: Default::default(),
            cache: None,
        })
    }
}

impl Stage {
    fn new(device: &wgpu::Device) -> Stage {
        let slot = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let (ro, rw) = (wgpu::BufferBindingType::Storage { read_only: true }, wgpu::BufferBindingType::Storage { read_only: false });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stage"),
            entries: &[slot(0, wgpu::BufferBindingType::Uniform), slot(1, wgpu::BufferBindingType::Uniform), slot(2, ro), slot(3, ro), slot(4, ro), slot(5, ro), slot(6, rw)],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("stage"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let main = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("stage"), source: wgpu::ShaderSource::Wgsl(format!("{STAGE_HEAD}{SHADER_STAGE}").into()) });
        let lin = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("linear"), source: wgpu::ShaderSource::Wgsl(format!("{STAGE_HEAD}{SHADER_LINEAR}").into()) });
        let make = |module: &wgpu::ShaderModule, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let dummy = device.create_buffer(&wgpu::BufferDescriptor { label: Some("unused"), size: 64, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let dummy_u = device.create_buffer(&wgpu::BufferDescriptor { label: Some("unused"), size: 96, usage: wgpu::BufferUsages::UNIFORM, mapped_at_creation: false });
        let merge = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("merge"), source: wgpu::ShaderSource::Wgsl(SHADER_MERGE.into()) });
        Stage {
            // 64 numbers per head, as every published model has. Two queries per invocation was the quickest
            // there: with one, each fetched key does half the work; with four, the running averages (256
            // numbers) no longer fit the registers.
            att_f32: Stage::att(device, &pl, false, 64, 2),
            att_i16: Stage::att(device, &pl, true, 64, 2),
            merge: make(&merge, "merge"),
            stats: make(&main, "stats"),
            norm1: make(&lin, "norm1"),
            norm2: make(&lin, "norm2"),
            linear: make(&lin, "linear"),
            linear1: make(&main, "linear1"),
            sshid: make(&main, "sshid"),
            ssout: make(&main, "ssout"),
            ropeq: make(&main, "ropeq"),
            ropek: make(&main, "ropek"),
            place: make(&main, "place"),
            cells: make(&main, "cells"),
            first: make(&main, "first"),
            turnk: make(&main, "turnk"),
            turnv: make(&main, "turnv"),
            clsnorm: make(&main, "clsnorm"),
            dec: make(&main, "dec"),
            dummy,
            dummy_u,
            layout,
            pipeline_layout: pl.clone(),
        }
    }
}

/// The in-context weights of a model in graphics memory (about 80 MB), shared by every tenant using that model.
pub struct GpuWeights {
    /// Per layer, then the final norm and decoder: the weights and the block of offsets into them.
    blocks: Vec<(wgpu::Buffer, wgpu::Buffer)>,
    /// The column stage's cached-attention blocks, and the row stage's blocks followed by its closing norm.
    col: Vec<(wgpu::Buffer, wgpu::Buffer)>,
    row: Vec<(wgpu::Buffer, wgpu::Buffer)>,
    /// For fitting (empty otherwise): the column blocks in which the inducing points attend to the training
    /// cells, the inducing points themselves `[inducing][e]` per block, and the attention built for them.
    col1: Vec<(wgpu::Buffer, wgpu::Buffer)>,
    ind: Vec<Vec<f32>>,
    col_att: Option<Att>,
    /// 1 / sqrt(head width) of the column stage.
    col_inv: f32,
    /// Class tokens `[cls][e]` and the rotary table for up to `rope_positions` positions.
    cls: wgpu::Buffer,
    rope: wgpu::Buffer,
    /// The few-keys attention built for the head widths of the column and the row stage.
    col_few: wgpu::ComputePipeline,
    row_few: wgpu::ComputePipeline,
    rope_positions: usize,
    /// The input layer: per output coordinate, its three weights and its bias.
    input: (wgpu::Buffer, wgpu::Buffer),
    e: usize,
    n_cls: usize,
    inds: usize,
    col_heads: usize,
    row_heads: usize,
    d: usize,
    ff: usize,
    heads: usize,
    layers: usize,
    /// Graphics memory these weights take. They are the published model's, not any tenant's data, so they are
    /// shared and not charged to anyone's budget.
    pub bytes: usize,
}

impl GpuWeights {
    /// Upload from a model loaded with its in-context weights (`Load::Predict` or `Load::Fit`).
    pub fn new(gpu: &Gpu, m: &Model) -> GpuWeights {
        assert!(m.icl.len() == m.icl_layers, "needs the in-context weights in memory");
        let d = m.d;
        let pack = |parts: &[&[f32]], layer: usize| {
            // offsets in the order of the shader's `L` block (the layer number sits between parts 16 and 17)
            let mut data: Vec<f32> = Vec::new();
            let mut offs = [0u32; 24];
            for (i, p) in parts.iter().enumerate() {
                // offsets count four-vectors; each part starts on one
                offs[if i < 16 { i } else { i + 1 }] = (data.len() / 4) as u32;
                data.extend_from_slice(p);
                data.resize(data.len().div_ceil(4) * 4, 0.0);
            }
            offs[16] = layer as u32;
            data.resize(data.len().max(4), 0.0);
            let w = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("weights"), contents: bytemuck::cast_slice(&data), usage: wgpu::BufferUsages::STORAGE });
            let u = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("offsets"), contents: bytemuck::cast_slice(&offs), usage: wgpu::BufferUsages::UNIFORM });
            (w, u)
        };
        // a model loaded for fitting has the pieces prediction leaves out
        let fit = m.col[0].a1.is_some();
        let e: &[f32] = &[];
        let mut blocks = Vec::new();
        let mut push = |parts: &[&[f32]], layer: usize| blocks.push(pack(parts, layer));
        for (l, b) in m.icl.iter().enumerate() {
            let ss = b.ss.as_ref().expect("in-context blocks have the scalable softmax");
            // the key and value projections follow if the model was loaded with them (for fitting)
            let (kw, kb, vw, vb) = if fit { (&b.in_w[d * d..2 * d * d], &b.in_b[d..2 * d], &b.in_w[2 * d * d..], &b.in_b[2 * d..]) } else { (e, e, e, e) };
            push(&[&b.n1w, &b.n1b, &b.in_w[..d * d], &b.in_b[..d], &ss.q0w, &ss.q0b, &ss.q2w, &ss.q2b, &b.out_w, &b.out_b, &b.n2w, &b.n2b, &b.l1w, &b.l1b, &b.l2w, &b.l2b, kw, kb, vw, vb], l);
        }
        push(&[e, e, e, e, e, e, e, e, e, e, &m.lnw, &m.lnb, &m.dec0w, &m.dec0b, &m.dec2w, &m.dec2b], 0);
        let e_ = m.e;
        // the inducing points' keys are stored divided by sqrt(head width); when the card makes them itself
        // (fitting), it does so with key weights that already carry that factor
        let col_inv = 1.0 / ((e_ / m.col_heads) as f32).sqrt();
        let col: Vec<_> = m
            .col
            .iter()
            .map(|isab| {
                let b = &isab.a2;
                let scaled = |x: &[f32]| x.iter().map(|v| v * col_inv).collect::<Vec<f32>>();
                let (kw, kb, vw, vb) =
                    if fit { (scaled(&b.in_w[e_ * e_..2 * e_ * e_]), scaled(&b.in_b[e_..2 * e_]), &b.in_w[2 * e_ * e_..], &b.in_b[2 * e_..]) } else { (Vec::new(), Vec::new(), e, e) };
                pack(&[&b.n1w, &b.n1b, &b.in_w[..e_ * e_], &b.in_b[..e_], e, e, e, e, &b.out_w, &b.out_b, &b.n2w, &b.n2b, &b.l1w, &b.l1b, &b.l2w, &b.l2b, &kw, &kb, vw, vb], 0)
            })
            .collect();
        // fitting only: the blocks in which the inducing points attend to a column's training cells
        let col1: Vec<_> = m
            .col
            .iter()
            .filter_map(|isab| isab.a1.as_ref())
            .map(|b| {
                pack(
                    &[
                        &b.n1w,
                        &b.n1b,
                        e,
                        e,
                        e,
                        e,
                        e,
                        e,
                        &b.out_w,
                        &b.out_b,
                        &b.n2w,
                        &b.n2b,
                        &b.l1w,
                        &b.l1b,
                        &b.l2w,
                        &b.l2b,
                        &b.in_w[e_ * e_..2 * e_ * e_],
                        &b.in_b[e_..2 * e_],
                        &b.in_w[2 * e_ * e_..],
                        &b.in_b[2 * e_..],
                    ],
                    0,
                )
            })
            .collect();
        let mut row: Vec<_> = m
            .row
            .iter()
            .map(|b| {
                let (w3, b3) = (&b.in_w, &b.in_b);
                pack(
                    &[
                        &b.n1w,
                        &b.n1b,
                        &w3[..e_ * e_],
                        &b3[..e_],
                        e,
                        e,
                        e,
                        e,
                        &b.out_w,
                        &b.out_b,
                        &b.n2w,
                        &b.n2b,
                        &b.l1w,
                        &b.l1b,
                        &b.l2w,
                        &b.l2b,
                        &w3[e_ * e_..2 * e_ * e_],
                        &b3[e_..2 * e_],
                        &w3[2 * e_ * e_..],
                        &b3[2 * e_..],
                    ],
                    0,
                )
            })
            .collect();
        row.push(pack(&[&m.out_lnw, &m.out_lnb], 0));
        assert!(m.group == 3, "the card's input layer is written for feature groups of three");
        let input: Vec<f32> = (0..m.e).flat_map(|j| [m.in_w[j * 3], m.in_w[j * 3 + 1], m.in_w[j * 3 + 2], m.in_b[j]]).collect();
        let input = (gpu.upload(&input), gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("unused"), size: 96, usage: wgpu::BufferUsages::UNIFORM, mapped_at_creation: false }));
        let init = |data: &[f32]| gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE });
        // rotary table: per position, cos of each frequency then sin of each (one head's width in all)
        let rope_positions = 4096;
        let mut table = Vec::with_capacity(rope_positions * 2 * m.rope_freqs.len());
        for p in 0..rope_positions {
            table.extend(m.rope_freqs.iter().map(|f| (p as f32 * f).cos()));
            table.extend(m.rope_freqs.iter().map(|f| (p as f32 * f).sin()));
        }
        let size = |v: &[(wgpu::Buffer, wgpu::Buffer)]| v.iter().map(|(a, b)| (a.size() + b.size()) as usize).sum::<usize>();
        let bytes = size(&blocks) + size(&col) + size(&row) + size(&col1) + (input.0.size() + input.1.size()) as usize + (m.cls_tokens.len() + table.len()) * 4;
        GpuWeights {
            bytes,
            blocks,
            col,
            row,
            col1,
            col_inv,
            ind: m.col.iter().map(|isab| isab.ind.clone()).collect(),
            // 16 numbers per head here, so four queries' running averages fit the registers
            col_att: fit.then(|| Stage::att(&gpu.device, &gpu.stage.pipeline_layout, false, m.e / m.col_heads, 4)),
            col_few: gpu.stage.few(&gpu.device, m.e / m.col_heads),
            row_few: gpu.stage.few(&gpu.device, m.e / m.row_heads),
            cls: init(&m.cls_tokens),
            rope: init(&table),
            rope_positions,
            input,
            e: m.e,
            n_cls: m.cls,
            inds: m.inds,
            col_heads: m.col_heads,
            row_heads: m.row_heads,
            d,
            ff: m.dec0b.len(),
            heads: m.icl_heads,
            layers: m.icl_layers,
        }
    }
}

/// One in-context problem of one fitted member, resident in graphics memory: a keys buffer and a values buffer
/// per layer, as 16-bit integers with one scale per coordinate (see `SHADER_ATTEND`). The buffers are released
/// when the last reference goes.
pub struct VramModel {
    layers: Vec<(wgpu::Buffer, wgpu::Buffer)>,
    /// The scales of the packed keys and values, `[layers][d]`: the keys' for queries made on the CPU (the
    /// card's own queries get them through `scales`), the values' for the card.
    kscale: Vec<f32>,
    vscale: wgpu::Buffer,
    /// Per column block: every feature group's inducing keys (already divided by sqrt(head width)) and values,
    /// `[groups][inducing][e]`. Empty for problems other than a member's only one.
    col: Vec<(wgpu::Buffer, wgpu::Buffer)>,
    features: usize,
    /// What each layer multiplies its queries by for this fitted model, times the keys' scales: `[layers][d]`.
    scales: wgpu::Buffer,
    rows: usize,
    pub bytes: usize,
    /// Where these bytes are counted while they exist, when loaded through a `VramTier`.
    lease: Option<Lease>,
}

impl Drop for VramModel {
    fn drop(&mut self) {
        for (k, v) in self.layers.iter().chain(&self.col) {
            k.destroy();
            v.destroy();
        }
        self.vscale.destroy();
        self.scales.destroy();
    }
}

impl Gpu {
    /// Graphics memory problem `node` of the member in `c` takes once loaded (`load`).
    pub fn model_bytes(gw: &GpuWeights, c: &Cache, node: usize) -> usize {
        let (layers, d) = (gw.layers, gw.d);
        let col = if c.nodes.len() == 1 && c.digits == 1 { gw.col.len() * 2 * c.features * gw.e * gw.inds * 4 } else { 0 };
        layers * 2 * c.nodes[node].rows.div_ceil(16) * 16 * d * 2 + 2 * layers * d * 4 + col
    }

    /// Stream one problem's keys and values from its bundle into graphics memory.
    pub fn load(&self, gw: &GpuWeights, c: &Cache, node: usize) -> io::Result<VramModel> {
        let (layers, d) = (gw.layers, gw.d);
        let rows = c.nodes[node].rows;
        let mut out = Vec::new();
        let (mut kscale, mut vscale) = (Vec::new(), Vec::new());
        for l in 0..layers {
            // whatever the cache's own format, the card holds 16 bits per number
            let (k, v) = layer_rows(c, node, l, d)?;
            let ((k, ks), (v, vs)) = (keys_i16(&k, rows, d), values_i16(&v, d));
            out.push((self.upload(&k), self.upload(&v)));
            kscale.extend(ks);
            vscale.extend(vs);
        }
        let mut scales = c.bundle.f32s(&format!("{}n{node}_scale", c.prefix))?;
        scales.iter_mut().zip(&kscale).for_each(|(s, k)| *s *= k);
        let (scales, vscale) = (self.upload(&scales), self.upload(&vscale));
        // the column stage's cache, for members with a single problem and a single label digit
        let mut col = Vec::new();
        if c.nodes.len() == 1 && c.digits == 1 {
            let all = c.bundle.f32s(&format!("{}col_kv", c.prefix))?;
            let g = c.features;
            let per = all.len() / g; // blocks * 2 * e * inducing
            let nb = gw.col.len();
            let ei = per / (nb * 2); // e * inducing
            let e = gw.e;
            let inds = ei / e;
            let inv = 1.0 / ((e / gw.col_heads) as f32).sqrt();
            for b in 0..nb {
                let (mut k, mut v) = (vec![0.0f32; g * ei], vec![0.0f32; g * ei]);
                for gi in 0..g {
                    let (kt, vt) = all[gi * per + b * 2 * ei..gi * per + (b + 1) * 2 * ei].split_at(ei);
                    for i in 0..inds {
                        for j in 0..e {
                            k[(gi * inds + i) * e + j] = kt[j * inds + i] * inv;
                            v[(gi * inds + i) * e + j] = vt[j * inds + i];
                        }
                    }
                }
                let put =
                    |data: &[f32]| self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("col"), contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE });
                col.push((put(&k), put(&v)));
            }
        }
        self.check()?;
        Ok(VramModel { layers: out, kscale, vscale, col, features: c.features, scales, rows, bytes: Gpu::model_bytes(gw, c, node), lease: None })
    }

    /// Attention outputs `[nq][d]` for queries against a resident layer, a block of queries at a time so that
    /// each block's working memory fits the budget.
    pub async fn attend_resident(&self, vm: &VramModel, layer: usize, q: &[f32], d: usize, heads: usize) -> io::Result<Vec<f32>> {
        let (k, v) = &vm.layers[layer];
        let at = &self.stage.att_i16;
        let nq = q.len() / d;
        let block = self.batch(nq, |n| self.att_need(at.qn, at.hd, n, vm.rows, d, heads));
        if block == 0 {
            return Err(too_big("not even one row's attention fits the working-memory budget"));
        }
        let mut out = Vec::with_capacity(q.len());
        for qs in q.chunks(block * d) {
            let _hold = self.reserve(self.att_need(at.qn, at.hd, qs.len() / d, vm.rows, d, heads))?;
            out.extend(self.attend_buffers(qs, k, v, Some(&vm.kscale[layer * d..(layer + 1) * d]), &vm.vscale, layer, vm.rows, d, heads).await?);
        }
        Ok(out)
    }
}

impl Gpu {
    /// The whole in-context stage for `h` (`[rows][d]` row representations) against a resident model. Returns
    /// the first `out_n` decoder outputs per row.
    pub async fn in_context(&self, gw: &GpuWeights, vm: &VramModel, h: &[f32], out_n: usize) -> io::Result<Vec<f32>> {
        self.chain(gw, vm, h, None, out_n).await
    }

    /// All three stages on the card for already-normalised rows `x` (`[rows][features]`). Needs a model loaded
    /// with its column cache (a member with a single in-context problem).
    pub async fn predict(&self, gw: &GpuWeights, vm: &VramModel, x: &[f32], out_n: usize) -> io::Result<Vec<f32>> {
        assert!(!vm.col.is_empty() && vm.features + gw.n_cls <= gw.rope_positions);
        self.chain(gw, vm, &[], Some(x), out_n).await
    }

    /// Working memory one submission of `chain` borrows for `nq` rows (from raw rows if `raw`), and the size of
    /// its largest single buffer.
    fn chain_need(&self, gw: &GpuWeights, vm: &VramModel, nq: usize, raw: bool, out_n: usize) -> (usize, usize) {
        let (d, ff, heads, e, g) = (gw.d, gw.ff, gw.heads, gw.e, vm.features);
        let at = &self.stage.att_i16;
        let (np, _, _, _, partial) = self.att_cut(at.qn, at.hd, nq, vm.rows, heads);
        // held from start to end: the attention's slices and the row representations
        let base = self.bk(partial) + self.bk(nq * d);
        // the in-context stage
        let icl = self.work_need(nq, d, ff, np * at.qn) + self.bk(nq * heads * 64) + self.bk((nq * out_n).div_ceil(4) * 4);
        let (mut all, mut widest) = (icl, partial.max(nq * ff) * 4);
        if raw {
            // the rows as given and their cells, the column stage's and the row stage's working sets (the
            // stages share buffers of the same size, but that saving is not counted on: see `Pool`)
            all += self.bk(nq * vm.features) + self.bk(nq * g * e) + self.work_need(nq * g, e, 2 * e, nq * g) + self.row_stage_need(e, gw.n_cls, nq, g);
            widest = widest.max(nq * (g + gw.n_cls) * 2 * e * 4);
        }
        (base + all + back_bytes(nq * out_n), widest)
    }

    async fn chain(&self, gw: &GpuWeights, vm: &VramModel, h: &[f32], x: Option<&[f32]>, out_n: usize) -> io::Result<Vec<f32>> {
        let t0 = web_time::Instant::now();
        let (nq, width) = match x {
            Some(x) => (x.len() / vm.features, vm.features),
            None => (h.len() / gw.d, gw.d),
        };
        let input = x.unwrap_or(h);
        // Rows go in sub-batches whose working set fits the budget (all of them at once if they fit). Each is
        // recorded and handed to the card without waiting for the one before: they reuse the same buffers and
        // the card runs them in order, so the card is never idle between them.
        let raw = x.is_some();
        let fits = |n: usize| {
            let (need, widest) = self.chain_need(gw, vm, n, raw, out_n);
            if widest > self.max_binding {
                usize::MAX
            } else {
                need
            }
        };
        let sub = self.batch(nq, fits);
        if sub == 0 {
            return Err(too_big("not even one row fits the working-memory budget"));
        }
        self.new_call();
        let mut out = Vec::with_capacity(nq * out_n);
        let mut waiting: Vec<(Pending, Reserved)> = Vec::new();
        for r0 in (0..nq).step_by(sub) {
            let n = sub.min(nq - r0);
            let need = fits(n);
            let mut hold = match self.try_reserve(need)? {
                Some(r) => r,
                None => {
                    // the budget is promised elsewhere: give back this call's own results first, then wait
                    for (p, r) in waiting.drain(..) {
                        out.extend(self.take(p).await?);
                        drop(r);
                    }
                    self.reserve(need)?
                }
            };
            let p = self.record(gw, vm, &input[r0 * width..(r0 + n) * width], raw, n, out_n);
            self.settled();
            hold.shrink(back_bytes(n * out_n));
            waiting.push((p, hold));
        }
        let t1 = t0.elapsed().as_secs_f64() * 1e3;
        let batches = waiting.len();
        for (p, r) in waiting {
            out.extend(self.take(p).await?);
            drop(r);
        }
        trace(|| format!("gpu stage: {nq} rows in {batches} sub-batches: recorded and submitted {t1:.2} ms, answer back {:.2} ms", t0.elapsed().as_secs_f64() * 1e3));
        Ok(out)
    }

    /// One submission of `chain`: `nq` rows of `input` (raw rows if `raw`, row representations otherwise), all
    /// the way to their first `out_n` outputs, copied into a buffer to be read back.
    fn record(&self, gw: &GpuWeights, vm: &VramModel, input: &[f32], raw: bool, nq: usize, out_n: usize) -> Pending<'_> {
        let (d, ff, heads) = (gw.d, gw.ff, gw.heads);
        let wk = self.att_work(&self.stage.att_i16, nq, vm.rows, heads);
        let hb = if raw { self.storage(nq * d) } else { self.fill(input) };
        let st = &self.stage;
        let no = &st.dummy;
        let skip = Skip(std::env::var("BROWSERTABICL_SKIP").unwrap_or_default());
        let logits;
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            // every step goes into one compute pass: starting a pass per step costs far more than the steps do
            let mut pass = enc.begin_compute_pass(&Default::default());
            let pass = &mut pass;
            if raw {
                // the first two stages, when starting from raw rows
                let (g, e) = (vm.features, gw.e);
                let (xb, ct) = (self.fill(input), self.storage(nq * g * e));
                let cp = self.params([nq, g, 1, 1, e, e * 2, 0, 1, 1, 0, 0, 0]);
                self.run(pass, &skip, "c", &st.cells, &cp.sel[0], &gw.input, [&xb, no, no], &ct, nq * g * e);
                self.column_blocks(pass, &skip, gw, &vm.col, nq, vm.features, &ct);
                self.row_stage(pass, &skip, gw, nq, vm.features, &ct, &hb);
            }
            let w = self.work(nq, d, ff, wk.np * wk.qn);
            let hid = self.storage(nq * heads * 64);
            logits = self.storage((nq * out_n).div_ceil(4) * 4);
            // nq, n, parts, heads, d, ff, out_n, set_mod, set_div, t, pad0, pad1
            let cu = self.params([nq, vm.rows, 1, heads, d, ff, out_n, 1, 1, 0, 0, 0]);
            for l in 0..gw.layers {
                let b = &gw.blocks[l];
                let (kb, vb) = &vm.layers[l];
                self.head(pass, &skip, "i", &cu, b, &hb, &w, nq, d);
                self.run(pass, &skip, "iS", &st.sshid, &cu.sel[0], b, [&w.q0, no, no], &hid, nq * heads * 64);
                self.run(pass, &skip, "iS", &st.ssout, &cu.sel[0], b, [&hid, &w.q0, &vm.scales], &w.q, nq * d);
                self.attention(pass, &st.att_i16, &wk, 0..wk.np.div_ceil(wk.qb), nq, vm.rows, d, heads, l, &w.q, At(kb, 0), At(vb, 0), &vm.vscale, At(&w.att, 0), &skip.0);
                self.tail(pass, &skip, "i", &cu, b, &hb, &w, nq, d, ff);
            }
            let last = &gw.blocks[gw.layers];
            self.run(pass, &skip, "iN", &st.stats, &cu.sel[0], last, [&hb, no, no], &w.st, nq);
            self.run(pass, &skip, "iN", &st.norm2, &cu.sel[0], last, [&hb, &w.st, no], &w.xn, nq * d / 4);
            self.lin(pass, &skip, "iF", &cu, 4, last, &w.xn, no, &w.mid, nq, ff);
            self.run(pass, &skip, "iD", &st.dec, &cu.sel[0], last, [&w.mid, no, no], &logits, nq * out_n);
        }
        let back = self.back(nq * out_n);
        enc.copy_buffer_to_buffer(&logits, 0, &back.buf, 0, back_bytes(nq * out_n) as u64);
        self.send(enc, back, nq * out_n)
    }
}

/// Which groups of passes to leave out, for timing: the letters in `BROWSERTABICL_SKIP`. A stage as a whole is
/// c (columns), r (rows) or i (in-context); a kind of pass in every stage is N (norms), Q (query projection),
/// O (output projection), F and G (the feed-forward layers), S (query scaling), D (decoder); x and y are the
/// attention of the column and the row stage, a and m the in-context attention and its merge.
struct Skip(String);

/// The parameter block of a stage (the shader's `C`), once per matrix a `linear` pass can select.
struct Params {
    sel: [wgpu::Buffer; 6],
}

/// Working buffers of one stage for `rows` tokens: statistics, a normalised copy, queries before and after
/// their finishing step, attention output, the half-way state of a block and the wide middle of its
/// feed-forward network.
struct Work<'a> {
    st: Loan<'a>,
    xn: Loan<'a>,
    q0: Loan<'a>,
    q: Loan<'a>,
    att: Loan<'a>,
    h2: Loan<'a>,
    mid: Loan<'a>,
}

/// Below this many output numbers a matrix pass uses the one-output-per-invocation form (see `linear1`).
const FEW_OUTPUTS: usize = 1 << 16;

impl Gpu {
    fn params(&self, v: [usize; 12]) -> Params {
        Params {
            sel: std::array::from_fn(|sel| {
                let mut p = [0u32; 16];
                for (a, b) in p.iter_mut().zip(v) {
                    *a = b as u32;
                }
                p[12] = sel as u32;
                self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("call"), contents: bytemuck::cast_slice(&p), usage: wgpu::BufferUsages::UNIFORM })
            }),
        }
    }

    fn work(&self, rows: usize, d: usize, ff: usize, query_rows: usize) -> Work<'_> {
        let make = |floats: usize| self.storage(floats);
        Work { st: make(rows * 4), xn: make(rows * d), q0: make(rows * d), q: make(query_rows * d), att: make(rows * d), h2: make(rows * d), mid: make(rows * ff) }
    }

    /// One step of a stage, unless one of the groups it belongs to (the letters of `tag`) is being skipped.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self, pass: &mut wgpu::ComputePass, skip: &Skip, tag: &str, pipe: &wgpu::ComputePipeline, cu: &wgpu::Buffer, block: &(wgpu::Buffer, wgpu::Buffer), x: [&wgpu::Buffer; 3], o: &wgpu::Buffer,
        count: usize,
    ) {
        if !tag.chars().any(|t| skip.0.contains(t)) {
            self.step(pass, pipe, [cu, &block.1, &block.0, x[0], x[1], x[2], o].map(|b| At(b, 0)), count)
        }
    }

    /// Matrix `sel` of `block` (see the shader's `lin_w`) applied to `rows` rows of `x0`, giving `wide` numbers
    /// per row in `o`; `x1` is what the matrices that close a residual branch add.
    #[allow(clippy::too_many_arguments)]
    fn lin(
        &self, pass: &mut wgpu::ComputePass, skip: &Skip, tag: &str, p: &Params, sel: usize, block: &(wgpu::Buffer, wgpu::Buffer), x0: &wgpu::Buffer, x1: &wgpu::Buffer, o: &wgpu::Buffer, rows: usize,
        wide: usize,
    ) {
        let st = &self.stage;
        if rows * wide <= FEW_OUTPUTS {
            self.run(pass, skip, tag, &st.linear1, &p.sel[sel], block, [x0, x1, &st.dummy], o, rows * wide);
        } else {
            self.run(pass, skip, tag, &st.linear, &p.sel[sel], block, [x0, x1, &st.dummy], o, rows.div_ceil(4) * wide / 4);
        }
    }

    /// The start of a block for the `rows` tokens in `h`: their first layer norm (into `w.xn`) and their query
    /// projection (into `w.q0`). `stage` is the skip letter of the stage; the steps add their own (N, Q).
    #[allow(clippy::too_many_arguments)]
    fn head(&self, pass: &mut wgpu::ComputePass, skip: &Skip, stage: &str, p: &Params, block: &(wgpu::Buffer, wgpu::Buffer), h: &wgpu::Buffer, w: &Work, rows: usize, d: usize) {
        let st = &self.stage;
        let no = &st.dummy;
        let tag = |kind: &str| format!("{stage}{kind}");
        self.run(pass, skip, &tag("N"), &st.stats, &p.sel[0], block, [h, no, no], &w.st, rows);
        self.run(pass, skip, &tag("N"), &st.norm1, &p.sel[0], block, [h, &w.st, no], &w.xn, rows * d / 4);
        self.lin(pass, skip, &tag("Q"), p, 0, block, &w.xn, no, &w.q0, rows, d);
    }

    /// The rest of a block once the attention output is in `w.att`: `h += out(att)`, then the feed-forward
    /// network. The steps' own skip letters are O, N, F and G.
    #[allow(clippy::too_many_arguments)]
    fn tail(&self, pass: &mut wgpu::ComputePass, skip: &Skip, stage: &str, p: &Params, block: &(wgpu::Buffer, wgpu::Buffer), h: &wgpu::Buffer, w: &Work, rows: usize, d: usize, ff: usize) {
        let st = &self.stage;
        let no = &st.dummy;
        let tag = |kind: &str| format!("{stage}{kind}");
        self.lin(pass, skip, &tag("O"), p, 3, block, &w.att, h, &w.h2, rows, d);
        self.run(pass, skip, &tag("N"), &st.stats, &p.sel[0], block, [&w.h2, no, no], &w.st, rows);
        self.run(pass, skip, &tag("N"), &st.norm2, &p.sel[0], block, [&w.h2, &w.st, no], &w.xn, rows * d / 4);
        self.lin(pass, skip, &tag("F"), p, 4, block, &w.xn, no, &w.mid, rows, ff);
        self.lin(pass, skip, &tag("G"), p, 5, block, &w.mid, &w.h2, h, rows, d);
    }

    /// The column stage's cached-attention blocks for `nq` rows of `g` cells each (`ct`, `[nq][g][e]`, updated
    /// in place): every cell against its feature group's inducing points, whose keys and values are in `kv`
    /// (per block, `[g][inducing][e]`).
    #[allow(clippy::too_many_arguments)]
    fn column_blocks(&self, pass: &mut wgpu::ComputePass, skip: &Skip, gw: &GpuWeights, kv: &[(wgpu::Buffer, wgpu::Buffer)], nq: usize, g: usize, ct: &wgpu::Buffer) {
        let (e, cells) = (gw.e, nq * g);
        let w = self.work(cells, e, 2 * e, cells);
        // for the few-keys attention: `parts` queries share each key set, pad0 apart, set s starting at s * pad1
        let p = self.params([cells, gw.inds, nq, gw.col_heads, e, e * 2, 0, g, 1, 0, g, 1]);
        for (b, (kb, vb)) in gw.col.iter().zip(kv) {
            self.column_block(pass, skip, gw, b, kb, vb, nq, g, ct, &w, &p);
        }
    }

    /// One of those blocks, with the working buffers and parameters `column_blocks` makes.
    #[allow(clippy::too_many_arguments)]
    fn column_block(
        &self, pass: &mut wgpu::ComputePass, skip: &Skip, gw: &GpuWeights, b: &(wgpu::Buffer, wgpu::Buffer), kb: &wgpu::Buffer, vb: &wgpu::Buffer, nq: usize, g: usize, ct: &wgpu::Buffer, w: &Work,
        p: &Params,
    ) {
        let (e, cells) = (gw.e, nq * g);
        self.head(pass, skip, "c", p, b, ct, w, cells, e);
        self.run(pass, skip, "cx", &gw.col_few, &p.sel[0], b, [&w.q0, kb, vb], &w.att, g * gw.col_heads * nq.div_ceil(4));
        self.tail(pass, skip, "c", p, b, ct, w, cells, e, 2 * e);
    }

    /// The row stage for `nq` rows: each row's class tokens and its `g` cells (`ct`, `[nq][g][e]`) attend to one
    /// another; the class tokens at the end, normalised and laid end to end, go to `out` (`[nq][cls * e]`).
    #[allow(clippy::too_many_arguments)]
    fn row_stage(&self, pass: &mut wgpu::ComputePass, skip: &Skip, gw: &GpuWeights, nq: usize, g: usize, ct: &wgpu::Buffer, out: &wgpu::Buffer) {
        let st = &self.stage;
        let no = &st.dummy;
        let (e, cls) = (gw.e, gw.n_cls);
        let tk = g + cls;
        let toks = nq * tk;
        let w = self.work(toks, e, 2 * e, toks);
        let make = |floats: usize| self.storage(floats);
        let (rt, rk, rv, top) = (make(toks * e), make(toks * e), make(toks * e), make(nq * cls * e));
        let p = self.params([toks, tk, tk, gw.row_heads, e, e * 2, 0, nq.max(1), tk, tk, 1, tk]);
        let place = self.params([nq, 0, 1, gw.row_heads, e, e * 2, cls, 1, 1, tk, 0, 0]);
        // the same for the last block, where only the class tokens' outputs are wanted: they are the queries,
        // cls to a row, against all of their row's tokens
        let few = nq * cls;
        let pf = self.params([few, tk, cls, gw.row_heads, e, e * 2, 0, nq.max(1), cls, cls, 1, cls]);
        let placef = self.params([nq, 0, 1, gw.row_heads, e, e * 2, cls, 1, 1, cls, 0, 0]);
        self.run(pass, skip, "r", &st.place, &place.sel[0], &gw.row[0], [ct, &gw.cls, no], &rt, toks * e);
        let blocks = gw.row.len() - 1;
        for (bi, b) in gw.row[..blocks].iter().enumerate() {
            let last = bi + 1 == blocks;
            // every token's keys and values (rotated keys also take the 1 / sqrt(head width))
            self.run(pass, skip, "r", &st.stats, &p.sel[0], b, [&rt, no, no], &w.st, toks);
            self.run(pass, skip, "r", &st.norm1, &p.sel[0], b, [&rt, &w.st, no], &w.xn, toks * e / 4);
            self.lin(pass, skip, "r", &p, 1, b, &w.xn, no, &w.q0, toks, e);
            self.run(pass, skip, "r", &st.ropek, &p.sel[0], b, [&w.q0, &gw.rope, no], &rk, toks * e);
            self.lin(pass, skip, "r", &p, 2, b, &w.xn, no, &rv, toks, e);
            if !last {
                self.lin(pass, skip, "r", &p, 0, b, &w.xn, no, &w.q0, toks, e);
                self.run(pass, skip, "r", &st.ropeq, &p.sel[0], b, [&w.q0, &gw.rope, no], &w.q, toks * e);
                self.run(pass, skip, "ry", &gw.row_few, &p.sel[0], b, [&w.q, &rk, &rv], &w.att, nq * gw.row_heads * tk.div_ceil(4));
                self.tail(pass, skip, "r", &p, b, &rt, &w, toks, e, 2 * e);
            } else {
                // pull the class tokens (and their normalised copies) out, and finish the block for them alone
                self.run(pass, skip, "r", &st.first, &place.sel[0], b, [&w.xn, no, no], &w.h2, few * e);
                self.run(pass, skip, "r", &st.first, &place.sel[0], b, [&rt, no, no], &top, few * e);
                self.lin(pass, skip, "r", &pf, 0, b, &w.h2, no, &w.q0, few, e);
                self.run(pass, skip, "r", &st.ropeq, &pf.sel[0], b, [&w.q0, &gw.rope, no], &w.q, few * e);
                self.run(pass, skip, "ry", &gw.row_few, &pf.sel[0], b, [&w.q, &rk, &rv], &w.att, nq * gw.row_heads * cls.div_ceil(4));
                self.tail(pass, skip, "r", &pf, b, &top, &w, few, e, 2 * e);
            }
        }
        self.run(pass, skip, "r", &st.stats, &pf.sel[0], &gw.row[0], [&top, no, no], &w.st, few);
        self.run(pass, skip, "r", &st.clsnorm, &placef.sel[0], &gw.row[blocks], [&top, &w.st, no], out, nq * cls * e);
    }
}

/// What the tier holds for one problem of one fitted member.
enum Slot {
    /// Its place is taken and its bytes are counted; it is being uploaded.
    Loading,
    Ready(Arc<VramModel>),
    /// Larger than its tenant's whole budget: it stays with the CPU (until `evict` forgets this).
    TooBig,
}

/// Graphics memory actually taken by fitted models loaded through the tier, in all and per tenant, counted from
/// the moment a load is started until the model's buffers are released. That includes models already evicted
/// from the tier but still in use by a prediction; so this, not the tier's list, is what the budgets are
/// checked against before anything new is loaded.
#[derive(Default)]
struct Live {
    bytes: usize,
    by_tenant: HashMap<Arc<str>, usize>,
}

/// A model's share of `Live`, given back when the model is released.
struct Lease {
    live: Arc<Mutex<Live>>,
    tenant: Arc<str>,
    bytes: usize,
}

impl Drop for Lease {
    fn drop(&mut self) {
        let mut l = self.live.lock().unwrap();
        l.bytes -= self.bytes;
        if let Some(t) = l.by_tenant.get_mut(&self.tenant) {
            *t -= self.bytes;
        }
    }
}

/// Graphics memory shared by many tenants, with the same rules as the in-memory tier (`warm::Warm`, whose
/// bookkeeping it shares): every entry belongs to one tenant and model, each tenant has a byte budget within the
/// overall one, the least recently used entry makes room, and an entry in use is not released until the
/// prediction using it has finished. On top of that the budgets are hard limits on what is really on the card:
/// a load starts only when the bytes of everything loaded and not yet released, evicted or not, leave room for
/// it; otherwise that prediction (and only that one) is answered by the CPU. A model larger than its tenant's
/// budget is never loaded and is always answered by the CPU.
///
/// What the card can hold at worst, then: `budget` of fitted models, plus the card's working-memory budget
/// (`Gpu::set_work_budget`, 1.5 GB unless set), plus each set of weights in use (about 80 MB for prediction).
pub struct VramTier {
    pub gpu: Arc<Gpu>,
    budget: usize,
    tenant_budget: usize,
    state: Mutex<Ledger<Slot>>,
    live: Arc<Mutex<Live>>,
    /// Predictions the card was asked for but the CPU answered because the card failed, and the last reason.
    pub fallbacks: AtomicU64,
    last_error: Mutex<Option<String>>,
}

impl VramTier {
    pub fn new(gpu: Arc<Gpu>, budget: usize, tenant_budget: usize) -> Arc<VramTier> {
        Arc::new(VramTier { gpu, budget, tenant_budget: tenant_budget.min(budget), state: Mutex::default(), live: Arc::default(), fallbacks: AtomicU64::new(0), last_error: Mutex::default() })
    }

    /// Bytes of graphics memory the tier holds now, in all and by `tenant` (models being loaded included).
    pub fn held(&self, tenant: &str) -> (usize, usize) {
        self.state.lock().unwrap().held(tenant)
    }

    /// Bytes of `tenant`'s models that are loaded and ready (not still being loaded).
    pub fn ready(&self, tenant: &str) -> usize {
        self.state.lock().unwrap().iter().filter(|(k, v, _)| &*k.0 == tenant && matches!(v, Slot::Ready(_))).map(|(_, _, n)| n).sum()
    }

    /// Bytes of fitted models really on the card now, in all and of `tenant`: what the tier holds plus anything
    /// evicted that a prediction is still using. Never more than the budgets.
    pub fn on_card(&self, tenant: &str) -> (usize, usize) {
        let l = self.live.lock().unwrap();
        (l.bytes, l.by_tenant.get(tenant).copied().unwrap_or(0))
    }

    /// Why the card last failed a prediction (which the CPU then answered), if it ever did.
    pub fn last_error(&self) -> Option<String> {
        self.last_error.lock().unwrap().clone()
    }

    /// Release everything `tenant` holds (for `model` only, if given). Models in use by a prediction are
    /// released when it finishes.
    pub fn evict(&self, tenant: &str, model: Option<&str>) {
        let gone = self.state.lock().unwrap().evict(tenant, model);
        drop(gone);
    }

    /// One tenant's use of the tier for one fitted model, to put in `PredictOpts::accel`.
    pub fn handle(self: &Arc<VramTier>, tenant: &str, model: &str, bundle: Arc<Bundle>, weights: Arc<GpuWeights>) -> crate::predict::CardAttend {
        crate::predict::CardAttend(Arc::new(VramHandle { tier: self.clone(), tenant: tenant.into(), model: model.into(), bundle, weights }))
    }

    /// Load problem `node` of the member `prefix` of `bundle` now, without waiting for a first prediction to
    /// start it. `true` if it is resident afterwards; `false` if the budgets do not allow it (now, or ever).
    pub fn preload(&self, tenant: &str, model: &str, bundle: &Bundle, weights: &GpuWeights, prefix: &str, node: usize) -> io::Result<bool> {
        let key: Key = (tenant.into(), model.into(), prefix.to_string(), node as u64);
        if let Some(Slot::Ready(_)) = self.state.lock().unwrap().get(&key) {
            return Ok(true);
        }
        match self.claim(&key, bundle, weights, prefix, node)? {
            Some(lease) => self.finish(key, lease, bundle, weights, prefix, node),
            None => Ok(false),
        }
    }

    /// Take a place for a model in the tier and count its bytes, before anything is uploaded. `None` if there is
    /// no room within the budgets (it is not loaded then; a later prediction tries again, unless the model is
    /// larger than its tenant's whole budget).
    fn claim(&self, key: &Key, bundle: &Bundle, gw: &GpuWeights, prefix: &str, node: usize) -> io::Result<Option<Lease>> {
        let bytes = Gpu::model_bytes(gw, &Cache::open(bundle, prefix)?, node);
        let mut s = self.state.lock().unwrap();
        if s.contains(key) {
            return Ok(None);
        }
        // only models that are ready may make room; one being loaded keeps its place
        let gone = match s.admit(key.clone(), bytes, Slot::Loading, self.budget, self.tenant_budget, |v| matches!(v, Slot::Ready(_))) {
            Ok(gone) => gone,
            Err((_, gone)) => {
                if bytes > self.tenant_budget {
                    let _ = s.admit(key.clone(), 0, Slot::TooBig, self.budget, self.tenant_budget, |_| false);
                }
                drop(gone);
                return Ok(None);
            }
        };
        // what was evicted is released here unless a prediction is still using it
        drop(gone);
        let mut l = self.live.lock().unwrap();
        let mine = l.by_tenant.get(&key.0).copied().unwrap_or(0);
        if l.bytes + bytes > self.budget || mine + bytes > self.tenant_budget {
            // the room is promised to models still in use: not this time
            drop(l);
            s.remove(key);
            return Ok(None);
        }
        l.bytes += bytes;
        *l.by_tenant.entry(key.0.clone()).or_insert(0) += bytes;
        Ok(Some(Lease { live: self.live.clone(), tenant: key.0.clone(), bytes }))
    }

    /// Upload a claimed model and make it ready (unless it was evicted meanwhile).
    fn finish(&self, key: Key, lease: Lease, bundle: &Bundle, gw: &GpuWeights, prefix: &str, node: usize) -> io::Result<bool> {
        let loaded = Cache::open(bundle, prefix).and_then(|c| self.gpu.load(gw, &c, node));
        let mut s = self.state.lock().unwrap();
        match loaded {
            Ok(mut vm) => {
                debug_assert_eq!(vm.bytes, lease.bytes);
                vm.lease = Some(lease);
                match s.get(&key) {
                    Some(slot @ Slot::Loading) => {
                        *slot = Slot::Ready(Arc::new(vm));
                        Ok(true)
                    }
                    Some(Slot::Ready(_)) => Ok(true),
                    _ => Ok(false),
                }
            }
            Err(e) => {
                if let Some(Slot::Loading) = s.get(&key) {
                    s.remove(&key);
                }
                *self.last_error.lock().unwrap() = Some(e.to_string());
                Err(e)
            }
        }
    }
}

struct VramHandle {
    tier: Arc<VramTier>,
    tenant: Arc<str>,
    model: Arc<str>,
    bundle: Arc<Bundle>,
    weights: Arc<GpuWeights>,
}

impl AttendCached for VramHandle {
    fn rows<'a>(&'a self, prefix: &'a str, x: &'a [f32], out_n: usize) -> Later<'a, io::Result<Option<Vec<f32>>>> {
        Box::pin(async move {
            match self.resident(prefix, 0) {
                Some(vm) if !vm.col.is_empty() && vm.features + self.weights.n_cls <= self.weights.rope_positions => self.or_cpu(self.tier.gpu.predict(&self.weights, &vm, x, out_n).await),
                _ => Ok(None),
            }
        })
    }

    fn stage<'a>(&'a self, prefix: &'a str, node: usize, h: &'a [f32], _d: usize, out_n: usize) -> Later<'a, io::Result<Option<Vec<f32>>>> {
        Box::pin(async move {
            match self.resident(prefix, node) {
                Some(vm) => self.or_cpu(self.tier.gpu.in_context(&self.weights, &vm, h, out_n).await),
                None => Ok(None),
            }
        })
    }

    fn attend<'a>(&'a self, prefix: &'a str, node: usize, layer: usize, q: &'a [f32], d: usize, heads: usize) -> Later<'a, io::Result<Option<Vec<f32>>>> {
        Box::pin(async move {
            match self.resident(prefix, node) {
                Some(vm) => self.or_cpu(self.tier.gpu.attend_resident(&vm, layer, q, d, heads).await),
                None => Ok(None),
            }
        })
    }
}

impl VramHandle {
    /// A card that fails (out of memory, lost) does not fail the prediction: the CPU answers it instead, and
    /// the tier counts it and keeps the reason.
    fn or_cpu(&self, r: io::Result<Vec<f32>>) -> io::Result<Option<Vec<f32>>> {
        match r {
            Ok(v) => Ok(Some(v)),
            Err(e) => {
                self.tier.fallbacks.fetch_add(1, Relaxed);
                trace(|| format!("gpu: the CPU answers instead: {e}"));
                *self.tier.last_error.lock().unwrap() = Some(e.to_string());
                Ok(None)
            }
        }
    }

    /// The model if it is resident; otherwise start loading it (once) and say no for now. Natively the load
    /// runs on a thread of its own and the CPU answers meanwhile; a browser has no thread to spare, so there it
    /// is loaded right here and this prediction already uses it.
    fn resident(&self, prefix: &str, node: usize) -> Option<Arc<VramModel>> {
        let key: Key = (self.tenant.clone(), self.model.clone(), prefix.to_string(), node as u64);
        match self.tier.state.lock().unwrap().get(&key) {
            Some(Slot::Ready(vm)) => return Some(vm.clone()),
            Some(_) => return None,
            None => {}
        }
        let lease = self.tier.claim(&key, &self.bundle, &self.weights, prefix, node).ok()??;
        #[cfg(target_arch = "wasm32")]
        {
            let ready = self.tier.finish(key.clone(), lease, &self.bundle, &self.weights, prefix, node).unwrap_or(false);
            match self.tier.state.lock().unwrap().get(&key) {
                Some(Slot::Ready(vm)) if ready => Some(vm.clone()),
                _ => None,
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (tier, bundle, weights, prefix) = (self.tier.clone(), self.bundle.clone(), self.weights.clone(), prefix.to_string());
            std::thread::spawn(move || {
                let _ = tier.finish(key, lease, &bundle, &weights, &prefix, node);
            });
            None
        }
    }
}
