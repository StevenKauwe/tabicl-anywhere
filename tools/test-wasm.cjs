// Run the WebAssembly build under Node against the reference's numbers (fixtures made by tools/dump_table.py).
const fs = require("fs"), path = require("path");
const { Weights, combineProba, defaultTemperature, toBundle, calibrate, memoryBytes } = require("../pkg/node/browsertabicl_wasm.js");
const fx = path.join(__dirname, "..", "fixtures");
function bundle(file) {
  const b = fs.readFileSync(file), n = Number(b.readBigUInt64LE(8)), out = {};
  for (const line of b.subarray(16, 16 + n).toString().trim().split("\n")) {
    const [name, dtype, off, len] = line.split(" ");
    const raw = b.buffer.slice(b.byteOffset + +off, b.byteOffset + +off + +len);
    out[name] = dtype === "float64" ? new Float64Array(raw) : dtype === "int32" ? new Int32Array(raw) : new Uint8Array(raw);
  }
  return out;
}
let t = Date.now();
const weights = new Weights(fs.readFileSync(path.join(fx, "classifier-v2.tirs")));
console.log(`weights loaded in ${Date.now() - t} ms`);
for (const [name, members] of [["table-b", 3], ["table-a", 8]]) {
  const d = bundle(path.join(fx, name + ".tirs"));
  const rows = d.y_train.length, features = d.x_train.length / rows;
  const labels = [...new Set(d.y_train)].sort((a, b) => a - b);
  const y = Uint32Array.from(d.y_train, v => labels.indexOf(v));
  t = Date.now();
  const fitted = weights.fitClassifier(d.x_train, y, rows, features, members, "f32");
  const fitMs = Date.now() - t;
  const n = d.x_test.length / features;
  t = Date.now();
  const proba = weights.predictProba(fitted, d.x_test, n);
  const predMs = Date.now() - t;
  let worst = 0;
  for (let i = 0; i < proba.length; i++) worst = Math.max(worst, Math.abs(proba[i] - d.proba[i]));
  console.log(`${name}: ${rows} x ${features}, ${members} members: fit ${fitMs} ms, predict ${n} rows ${predMs} ms, model ${(fitted.length / 1e6).toFixed(0)} MB, worst probability difference from the reference ${worst.toExponential(2)}`);
  if (!(worst < 2e-4)) { console.error("MISMATCH"); process.exit(1); }
  // The same ensemble fitted and asked ONE member at a time (what a pool of browser workers does), then combined:
  // must be the very same numbers as the whole-ensemble calls above.
  const sums = new Float64Array(proba.length);
  let bytes = 0;
  for (let m = 0; m < members; m++) {
    const one = weights.fitClassifierMember(d.x_train, y, rows, features, members, m, "f32", "");
    bytes += one.length;
    const part = weights.predictSums(one, d.x_test, n);
    for (let i = 0; i < sums.length; i++) sums[i] += part[i];
  }
  const joined = combineProba(sums, proba.length / n, members, defaultTemperature());
  let apart = 0;
  for (let i = 0; i < proba.length; i++) apart = Math.max(apart, Math.abs(joined[i] - proba[i]));
  console.log(`${name}: one member at a time, combined: differs from the whole-ensemble path by ${apart} (must be 0); ${members} separate members take ${(bytes / 1e6).toFixed(0)} MB`);
  if (apart !== 0) { console.error("MISMATCH between the one-member and whole-ensemble paths"); process.exit(1); }
}
// Half-size weights (16-bit floats): made here from the full file, loaded, and compared with the reference.
{
  const full = fs.readFileSync(path.join(fx, "classifier-v2.tirs"));
  const half = toBundle(full, true);
  const onDisk = path.join(fx, "classifier-v2-half.tirs");
  const same = fs.existsSync(onDisk) ? Buffer.compare(Buffer.from(half), fs.readFileSync(onDisk)) === 0 : null;
  const hw = new Weights(half);
  const d = bundle(path.join(fx, "table-a.tirs"));
  const rows = d.y_train.length, features = d.x_train.length / rows, n = d.x_test.length / features;
  const labels = [...new Set(d.y_train)].sort((a, b) => a - b);
  const y = Uint32Array.from(d.y_train, v => labels.indexOf(v));
  const proba = hw.predictProba(hw.fitClassifier(d.x_train, y, rows, features, 8, "i16"), d.x_test, n);
  let worst = 0;
  for (let i = 0; i < proba.length; i++) worst = Math.max(worst, Math.abs(proba[i] - d.proba[i]));
  console.log(`half-size weights: ${(full.length / 1e6).toFixed(1)} MB -> ${(half.length / 1e6).toFixed(1)} MB${same === null ? "" : same ? " (identical to fixtures/classifier-v2-half.tirs)" : " (DIFFERS from fixtures/classifier-v2-half.tirs)"}; table-a with a 16-bit fitted model: worst probability difference from the reference ${worst.toExponential(2)}; engine memory ${(memoryBytes() / 1e6).toFixed(0)} MB`);
  if (!(worst < 2e-3) || same === false) { console.error("MISMATCH"); process.exit(1); }
  console.log(`inner-loop speed here: ${(Math.max(...Array.from({ length: 16 }, () => calibrate(25))) / 1e9).toFixed(2)} billion multiply-adds a second (best of 16 short bursts)`);
}
// the published checkpoint, read directly
const ck = path.join(__dirname, "..", "ref", "ckpt", "tabicl-regressor-v2-20260212.ckpt");
if (fs.existsSync(ck)) {
  t = Date.now();
  const reg = new Weights(fs.readFileSync(ck));
  console.log(`regression checkpoint converted and loaded in ${Date.now() - t} ms; isRegression = ${reg.isRegression}`);
  const d = bundle(path.join(fx, "reg-b.tirs"));
  const rows = d.y_train.length, features = d.x_train.length / rows, n = d.x_test.length / features;
  const fitted = reg.fitRegressor(d.x_train, d.y_train, rows, features, 1, "i16");
  const out = reg.predictRegression(fitted, d.x_test, n);
  let worst = 0;
  for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(out[4 * i] - d.mean[i]));
  console.log(`reg-b: worst difference in the predicted mean ${worst.toExponential(2)} (16-bit cache)`);
}
