// Run the WebAssembly build under Node against the reference's numbers.
//
// With the generated fixtures (fixtures/, tools/dump_*.py) it uses the full-precision weights and tables there.
// Without them (as in CI) it uses the small committed data: the tiny model in tests/data (read straight from its
// PyTorch checkpoint, exact comparison) and the real 16-bit weights in docs/weights on a small table. Build first
// with tools/build-wasm.sh.
const fs = require("fs"), path = require("path");
const { Weights, combineProba, defaultTemperature, toBundle, calibrate, memoryBytes } = require("../pkg/node/tabicl_anywhere_wasm.js");
const root = path.join(__dirname, "..");
const fx = path.join(root, "fixtures"), small = path.join(root, "tests", "data");
const full = fs.existsSync(path.join(fx, "classifier-v2.tirs"));
function bundle(file) {
  const b = fs.readFileSync(file), n = Number(b.readBigUInt64LE(8)), out = {};
  for (const line of b.subarray(16, 16 + n).toString().trim().split("\n")) {
    const [name, dtype, off, len] = line.split(" ");
    const raw = b.buffer.slice(b.byteOffset + +off, b.byteOffset + +off + +len);
    out[name] = dtype === "float64" ? new Float64Array(raw) : dtype === "int32" ? new Int32Array(raw) : new Uint8Array(raw);
  }
  return out;
}
function fail(msg) { console.error(msg); process.exit(1); }
function worstOf(a, b) { let w = 0; for (let i = 0; i < a.length; i++) w = Math.max(w, Math.abs(a[i] - b[i])); return w; }
// the web page's copy of the weights: parts listed in a .json beside them
function docsWeights(kind) {
  const dir = path.join(root, "docs", "weights"), list = JSON.parse(fs.readFileSync(path.join(dir, `${kind}-v2-half.json`), "utf8"));
  const bytes = Buffer.concat(list.parts.map(p => fs.readFileSync(path.join(dir, p.file))));
  if (bytes.length !== list.bytes) fail(`docs/weights/${kind}: parts add up to ${bytes.length} bytes, the list says ${list.bytes}`);
  return bytes;
}

// Fit and predict a table fixture, compare with the reference within `tol`, then check that fitting one member
// at a time and combining (what a pool of browser workers does) gives exactly the same numbers.
function classify(weights, label, file, members, tol, kv = "f32") {
  const d = bundle(file);
  const rows = d.y_train.length, features = d.x_train.length / rows;
  const labels = [...new Set(d.y_train)].sort((a, b) => a - b);
  const y = Uint32Array.from(d.y_train, v => labels.indexOf(v));
  let t = Date.now();
  const fitted = weights.fitClassifier(d.x_train, y, rows, features, members, kv);
  const fitMs = Date.now() - t;
  const n = d.x_test.length / features;
  t = Date.now();
  const proba = weights.predictProba(fitted, d.x_test, n);
  const worst = worstOf(proba, d.proba);
  console.log(`${label}: ${rows} x ${features}, ${members} members: fit ${fitMs} ms, predict ${n} rows ${Date.now() - t} ms, worst probability difference from the reference ${worst.toExponential(2)} (allowed ${tol})`);
  if (!(worst < tol)) fail("MISMATCH");
  const sums = new Float64Array(proba.length);
  for (let m = 0; m < members; m++) {
    const part = weights.predictSums(weights.fitClassifierMember(d.x_train, y, rows, features, members, m, kv, ""), d.x_test, n);
    for (let i = 0; i < sums.length; i++) sums[i] += part[i];
  }
  const apart = worstOf(combineProba(sums, proba.length / n, members, defaultTemperature()), proba);
  console.log(`${label}: one member at a time, combined: differs from the whole-ensemble path by ${apart} (must be 0)`);
  if (apart !== 0) fail("MISMATCH between the one-member and whole-ensemble paths");
}

function regress(weights, label, file, members, tol, kv) {
  const d = bundle(file);
  const rows = d.y_train.length, features = d.x_train.length / rows, n = d.x_test.length / features;
  const out = weights.predictRegression(weights.fitRegressor(d.x_train, d.y_train, rows, features, members, kv), d.x_test, n);
  let worst = 0;
  for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(out[4 * i] - d.mean[i]));
  console.log(`${label}: worst difference in the predicted mean ${worst.toExponential(2)} (allowed ${tol})`);
  if (!(worst < tol)) fail("MISMATCH");
}

if (full) {
  let t = Date.now();
  const weights = new Weights(fs.readFileSync(path.join(fx, "classifier-v2.tirs")));
  console.log(`weights loaded in ${Date.now() - t} ms`);
  classify(weights, "table-b", path.join(fx, "table-b.tirs"), 3, 2e-4);
  classify(weights, "table-a", path.join(fx, "table-a.tirs"), 8, 2e-4);
  // Half-size weights (16-bit floats): made here from the full file, loaded, and compared with the reference.
  const fullBytes = fs.readFileSync(path.join(fx, "classifier-v2.tirs"));
  const half = toBundle(fullBytes, true);
  const onDisk = path.join(fx, "classifier-v2-half.tirs");
  const same = fs.existsSync(onDisk) ? Buffer.compare(Buffer.from(half), fs.readFileSync(onDisk)) === 0 : null;
  console.log(`half-size weights: ${(fullBytes.length / 1e6).toFixed(1)} MB -> ${(half.length / 1e6).toFixed(1)} MB${same === null ? "" : same ? " (identical to fixtures/classifier-v2-half.tirs)" : " (DIFFERS from fixtures/classifier-v2-half.tirs)"}`);
  if (same === false) fail("MISMATCH");
  classify(new Weights(half), "table-a, half weights, 16-bit fitted model", path.join(fx, "table-a.tirs"), 8, 2e-3, "i16");
  const ck = path.join(root, "ref", "ckpt", "tabicl-regressor-v2-20260212.ckpt");
  if (fs.existsSync(ck)) {
    t = Date.now();
    const reg = new Weights(fs.readFileSync(ck));
    console.log(`regression checkpoint converted and loaded in ${Date.now() - t} ms; isRegression = ${reg.isRegression}`);
    regress(reg, "reg-b (16-bit cache)", path.join(fx, "reg-b.tirs"), 1, 2e-2, "i16");
  }
} else {
  console.log("fixtures/ not generated: using the small committed data (tests/data, docs/weights)");
  // the tiny model, from its PyTorch checkpoint, at full precision: only rounding separates it from the reference
  const tiny = new Weights(fs.readFileSync(path.join(small, "tiny-clf.ckpt")));
  classify(tiny, "tiny-table", path.join(small, "tiny-table.tirs"), 8, 1e-5);
  classify(tiny, "tiny-many (14 classes)", path.join(small, "tiny-many.tirs"), 2, 1e-5);
  const tinyReg = new Weights(fs.readFileSync(path.join(small, "tiny-reg.ckpt")));
  if (!tinyReg.isRegression) fail("the tiny regression checkpoint was not recognised as one");
  regress(tinyReg, "tiny-reg-b", path.join(small, "tiny-reg-b.tirs"), 1, 1e-4, "f32");
  // the real model at 16 bits against the reference's full-precision answers (see tests/reference.rs for why 5e-3)
  classify(new Weights(docsWeights("classifier")), "table-b, 16-bit weights", path.join(small, "table-b.tirs"), 3, 5e-3);
  regress(new Weights(docsWeights("regressor")), "reg-b, 16-bit weights", path.join(small, "reg-b.tirs"), 1, 0.04, "f32");
}
console.log(`engine memory ${(memoryBytes() / 1e6).toFixed(0)} MB; inner-loop speed here: ${(Math.max(...Array.from({ length: 16 }, () => calibrate(25))) / 1e9).toFixed(2)} billion multiply-adds a second (best of 16 short bursts)`);
