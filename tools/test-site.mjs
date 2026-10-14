// Everything about the page in site/ that can be checked without a browser, under Node:
//   node tools/test-site.mjs
// It runs the page's own code (site/app.js, site/worker.js) against the real engine and weights.
// NOT covered here (needs a real browser): the page itself, browser storage, drag and drop, the clipboard,
// graphics-card detection, and whether a given host lets the page start workers.
import fs from "node:fs";
import path from "node:path";
import { Worker } from "node:worker_threads";
import { execFileSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";
import * as app from "../site/app.js";
import * as worker from "../site/worker.js";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..");
const wasmPath = path.join(root, "site/pkg/tabicl_anywhere_wasm_bg.wasm");
const scratch = process.env.SCRATCH || path.resolve(root, process.env.CARGO_TARGET_DIR || "target");
fs.mkdirSync(scratch, { recursive: true });
// Test data: the generated fixtures/ when present; otherwise the small committed copies in tests/data, and the
// half-size weights joined from the web page's own parts in docs/weights (so this runs in CI too).
const fx = (f) => {
  for (const dir of ["fixtures", "tests/data"]) if (fs.existsSync(path.join(root, dir, f))) return path.join(root, dir, f);
  const m = /^(classifier|regressor)-v2-half\.tirs$/.exec(f);
  if (!m) return null;
  const out = path.join(scratch, f);
  if (!fs.existsSync(out)) {
    const dir = path.join(root, "docs/weights"), list = JSON.parse(fs.readFileSync(path.join(dir, `${m[1]}-v2-half.json`), "utf8"));
    fs.writeFileSync(out, Buffer.concat(list.parts.map((p) => fs.readFileSync(path.join(dir, p.file)))));
  }
  return out;
};
let failures = 0;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failures++; };
const worst = (a, b) => { let w = a.length === b.length ? 0 : Infinity; for (let i = 0; i < a.length; i++) w = Math.max(w, Math.abs(a[i] - b[i])); return w; };

function bundle(file) {
  const b = fs.readFileSync(file), n = Number(b.readBigUInt64LE(8)), out = {};
  for (const line of b.subarray(16, 16 + n).toString().trim().split("\n")) {
    const [name, dtype, off, len] = line.split(" ");
    const raw = b.buffer.slice(b.byteOffset + +off, b.byteOffset + +off + +len);
    out[name] = dtype === "float64" ? new Float64Array(raw) : dtype === "int32" ? new Int32Array(raw) : new Uint8Array(raw);
  }
  return out;
}
const blobOf = (file) => new Blob([fs.readFileSync(file)]);
/** A table as CSV text: f0..fN and the target `y`; NaN becomes an empty cell. */
function csvOf(x, y, rows, features) {
  const lines = [[...Array.from({ length: features }, (_, j) => `f${j}`), ...(y ? ["y"] : [])].join(",")];
  for (let r = 0; r < rows; r++) lines.push([...Array.from(x.subarray(r * features, (r + 1) * features), (v) => (Number.isNaN(v) ? "" : String(v))), ...(y ? [String(y[r])] : [])].join(","));
  return lines.join("\n") + "\n";
}

// ---- 1. estimates: pure arithmetic ---------------------------------------------------------------------
check(app.levels(2) === 1 && app.levels(10) === 1 && app.levels(11) === 2 && app.levels(100) === 2 && app.levels(101) === 3, "levels(): one step up to 10 classes, two up to 100");
check(app.planWorkers({ members: 8, cores: 16, deviceMemoryGB: 8 }) === 8 && app.planWorkers({ members: 8, cores: 4, deviceMemoryGB: 8 }) === 3 && app.planWorkers({ members: 8, cores: 16, deviceMemoryGB: 1 }) === 2 && app.planWorkers({ members: 1, cores: 16 }) === 1, "planWorkers(): bounded by members, processor threads and memory");
{
  const iphone = "Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/141.0 Mobile/15E148 Safari/604.1";
  const g = (o) => app.deviceMemoryGuess(o);
  check(g({ deviceMemory: 8 }) === 8 && g({ userAgent: iphone }) === 1 && g({ userAgent: "Macintosh", platform: "MacIntel", maxTouchPoints: 5 }) === 1 && g({ userAgent: "Macintosh", platform: "MacIntel" }) === 4 && g({ mobile: true }) === 1, "deviceMemoryGuess(): what the browser reports; else 1 GB on a phone or tablet (an iPad calling itself a Mac), 4 GB otherwise");
  const small = app.estimate({ rows: 150, features: 4, classes: 3, members: 8, workers: 1, rate: 5e9 });
  const mid = app.estimate({ rows: 2000, features: 20, classes: 3, members: 8, workers: 1, rate: 5e9 });
  const onPhone = (e) => app.planWorkers({ members: 8, cores: 6, deviceMemoryGB: g({ userAgent: iphone }), perWorkerBytes: e.perWorkerBytes });
  check(onPhone(small) === 2 && onPhone(mid) === 1, `on an iPhone: 2 workers for the example table (${app.formatBytes(small.perWorkerBytes)} each), 1 for 2000 x 20 (${app.formatBytes(mid.perWorkerBytes)})`);
}
{
  // The engine's builds: the browser's own WebAssembly.validate is asked about tiny test modules. Node accepts
  // all three; Safari refuses relaxed SIMD (stood in for here by refusing the relaxed module's bytes).
  check(JSON.stringify(worker.usableBuilds()) === '["relaxed","simd","scalar"]', "the engine build probes: this Node accepts relaxed SIMD, SIMD and the plain module");
  const noRelaxed = (b) => b !== worker.PROBES.relaxed && WebAssembly.validate(b);
  const noSimd = (b) => b === worker.PROBES.scalar;
  check(worker.usableBuilds(noRelaxed)[0] === "simd" && worker.usableBuilds(noSimd)[0] === "scalar" && worker.usableBuilds(() => false).length === 0, "a browser without relaxed SIMD (Safari) gets the simd build, one without SIMD the scalar build");
  for (const [name, file] of Object.entries(worker.BUILDS)) check(fs.existsSync(path.join(root, "site/pkg", file)), `site/pkg holds the ${name} build (${file})`);
  const safari = "the engine file could not be loaded (WebAssembly.Module doesn't parse at byte 1359: relaxed simd instructions not supported, in function at index 4)";
  check(/could not load the engine's file/.test(app.engineTrouble(safari)) && /too old/.test(app.engineTrouble("the engine file could not be loaded (this browser accepts none of the engine's builds: it is too old for WebAssembly as this page uses it)")) && /out of memory/.test(app.engineTrouble("RangeError: Out of memory")), "the engine's start-up failures are explained in plain words");
}
{
  const huge = app.estimate({ rows: 200000, features: 100, members: 1, workers: 1, rate: 5e9 });
  const fine = app.estimate({ rows: 500, features: 100, classes: 3, members: 4, workers: 4, rate: 5e9 });
  check(!!huge.refuse && !fine.refuse, `estimate(): refuses what cannot fit in 4 GB (200,000 x 100 needs ${app.formatBytes(huge.perWorkerBytes)}), allows 500 x 100`);
}

// ---- 2. weights in parts: fetch, join, and notice a short download ----------------------------------------
const fileFetch = (cut) => async (url) => {
  const p = fileURLToPath(url);
  if (!fs.existsSync(p)) return new Response("", { status: 404 });
  let body = fs.readFileSync(p);
  if (cut && p.endsWith(cut)) body = body.subarray(0, body.length - 1000);
  return new Response(body, { headers: { "content-length": String(body.length) } });
};
// the page fetches weights/ beside itself: site/weights after tools/build-site.sh, else the committed docs/ copy
const pageDir = fs.existsSync(path.join(root, "site/weights/classifier-v2-half.json")) ? "site" : "docs";
const pageUrl = pathToFileURL(path.join(root, pageDir, "index.html")).href;
const weightBlobs = {};
for (const kind of ["classifier", "regressor"]) {
  const src = app.weightsSource("weights/", kind, pageUrl);
  let last = [0, 0], calls = 0;
  const { blob, convert } = await app.fetchWeights(src, { fetchFn: fileFetch(), onProgress: (d, t) => { last = [d, t]; calls++; } });
  const whole = fs.readFileSync(fx(`${kind}-v2-half.tirs`));
  const same = Buffer.compare(Buffer.from(await blob.arrayBuffer()), whole) === 0;
  check(same && !convert && last[0] === whole.length && last[1] === whole.length && calls >= 4, `${kind} weights: parts fetched and joined are byte-identical to fixtures/${kind}-v2-half.tirs (${app.formatBytes(whole.length)}, progress reported ${calls} times)`);
  weightBlobs[kind] = blob;
}
{
  const src = app.weightsSource("weights/", "classifier", pageUrl);
  const err = await app.fetchWeights(src, { fetchFn: fileFetch("classifier-v2-half.02.bin") }).then(() => null, (e) => e.message);
  check(err && /cut short/.test(err), `a part that arrives short is refused ("${err}")`);
  const missing = await app.fetchWeights(app.weightsSource("nowhere/", "classifier", pageUrl), { fetchFn: fileFetch() }).then(() => null, (e) => e.message);
  check(missing && /404/.test(missing), `a missing weights folder gives a plain error ("${missing.replace(root, "...")}")`);
  const hf = app.weightsSource("https://huggingface.co/jingang/TabICL/resolve/main", "regressor", pageUrl);
  check(hf.url === "https://huggingface.co/jingang/TabICL/resolve/main/tabicl-regressor-v2-20260212.ckpt" && !hf.parts, "a Hugging Face address selects the published checkpoint");
}

// ---- 3. the worker's requests, called directly ------------------------------------------------------------
const wasm = fs.readFileSync(wasmPath);
let nextId = 1;
const ask = async (op, args = {}) => {
  const r = await worker.handle({ id: nextId++, op, ...args }, wasm);
  if (!r.ok) throw Object.assign(new Error(r.error), { fatal: r.fatal });
  return r.result;
};
const hello = await ask("hello", { calibrateMs: 400 });
check(hello.rate > 1e8, `engine starts; this machine: ${(hello.rate / 1e9).toFixed(2)} billion multiply-adds a second on one thread`);
const rate = hello.rate;
check(await ask("fit", { kind: "classifier" }).then(() => false, (e) => /not loaded/.test(e.message) && !e.fatal), "asking for a fit before the weights are loaded is an ordinary error, not a crash");
check(await ask("describe", { text: "a,b\n1\n" }).then(() => false, (e) => /row 2 has 1 fields/.test(e.message)), "a ragged CSV is refused with the row number");

// 3a. a table with text columns and empty cells: the page's path against the command line on the same file
{
  let s = 7;
  const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  const mk = (n, withTarget) => {
    const lines = [withTarget ? "size,colour,weight,note,kind" : "weight,colour,size,note"];
    for (let i = 0; i < n; i++) {
      const size = rnd() * 10, colour = ["red", "green", "blue, dark"][Math.floor(rnd() * 3)], weight = rnd() * 5;
      const kind = size + (colour === "red" ? 3 : 0) + rnd() * 2 > 8 ? "big one" : weight > 2.5 ? "heavy" : "small";
      const c = (v) => (v.includes(",") ? `"${v}"` : v), sz = rnd() < 0.1 ? "" : size.toFixed(3), w = rnd() < 0.05 ? "NA" : weight.toFixed(3);
      lines.push(withTarget ? `${sz},${c(colour)},${w},x${i % 4},${kind}` : `${w},${c(i === 0 ? "violet" : colour)},${sz},x${i % 4}`);
    }
    return lines.join("\n") + "\n";
  };
  const train = mk(160, true), fresh = mk(25, false);
  const info = await ask("describe", { text: train });
  check(info.rows === 160 && info.columns.length === 5 && info.columns[1].numeric === false && info.columns[1].distinct === 3 && info.columns[0].numeric && info.columns[0].missing > 0 && info.columns[4].distinct === 3, "describe: text columns, distinct values and empty cells are reported");
  const table = await ask("table", { text: train, target: "kind", regression: false, estimators: 4 });
  check(table.rows === 160 && table.features === 4 && table.classes === 3 && table.members === 4 && table.labels.join("|") === "big one|heavy|small", "table: 160 rows, 4 feature columns, labels in sorted order");
  await ask("load", { kind: "classifier", blob: weightBlobs.classifier });
  const store = app.memoryStore();
  const direct = { call: (_rec, op, args) => ask(op, args), ready: async () => [{}] };
  const { entry } = await app.fitModel(direct, store, { kind: "classifier", weightsBlob: weightBlobs.classifier, table, name: "t", target: "kind" });
  const { x } = await ask("features", { schema: entry.schema, text: fresh });
  const res = await app.predictModel(direct, store, { model: entry, weightsBlob: weightBlobs.classifier, x, rows: 25 });
  const pageCsv = app.resultsCsv(entry, res, 25);
  {
    // "Fill in example rows" on a visitor's own table: rows picked from it, the target left out, quoting kept
    const [head, ...all] = app.parseCsv(train);
    const picked = app.pickRows(all, 7);
    const { text, actual } = app.rowsToPredict(head, picked, 4);
    const back = app.parseCsv(text);
    const { x: px } = await ask("features", { schema: entry.schema, text });
    check(head.length === 5 && all.length === 160 && back[0].join() === "size,colour,weight,note" && back.length === 8 && back.slice(1).every((r, i) => r.join() === picked[i].slice(0, 4).join()) && actual.every((a, i) => a === picked[i][4]) && px.length === 7 * entry.features,
      "example rows from a visitor's table: 7 rows, header without the target, quoted commas and empty cells survive, the engine reads them");
    const lines = app.sampleCsv().trim().split("\n"), fresh7 = Array.from({ length: 7 }, () => app.seedling(Math.random));
    check(lines.length === 151 && lines[0] === app.SEEDLING_COLUMNS.join(",") && fresh7.every((r) => r.length === 5 && /^(thrived|survived|died)$/.test(r[4])), "the example table and fresh example rows come from the same generator");
  }
  const bin = path.resolve(root, process.env.CARGO_TARGET_DIR || "target", "release/tabicl-anywhere");
  if (fs.existsSync(bin)) {
    const dir = fs.mkdtempSync(path.join(scratch, "site-test-"));
    fs.writeFileSync(path.join(dir, "train.csv"), train);
    fs.writeFileSync(path.join(dir, "new.csv"), fresh);
    execFileSync(bin, ["fit", fx("classifier-v2-half.tirs"), path.join(dir, "train.csv"), "--target", "kind", path.join(dir, "m.tirs"), "--kv", "i16", "--estimators", "4"], { stdio: "pipe" });
    const cli = execFileSync(bin, ["predict", fx("classifier-v2-half.tirs"), path.join(dir, "m.tirs"), path.join(dir, "new.csv")], { stdio: ["ignore", "pipe", "pipe"] }).toString();
    const rowsOf = (t) => t.trim().split("\n").map((l) => l.split(","));
    const [a, b] = [rowsOf(pageCsv), rowsOf(cli)];
    let diff = 0, labels = a.length === b.length && a[0].join() === b[0].join();
    for (let i = 1; labels && i < a.length; i++) { labels = a[i][0] === b[i][0]; for (let j = 1; j < a[i].length; j++) diff = Math.max(diff, Math.abs(a[i][j] - b[i][j])); }
    check(labels && diff < 1e-4, `CSV with text columns, quoted commas, empty and NA cells, an unseen category, reordered columns: page and command line give the same 25 labels; probabilities differ by at most ${diff.toExponential(1)}`);
    fs.rmSync(dir, { recursive: true });
  } else console.log(`skip command-line comparison (no ${bin})`);
}

// 3b. member-by-member through CSV, against the reference's probabilities (half weights, compact storage: what the page uses)
for (const name of ["table-b", "table-a"]) {
  if (!fx(name + ".tirs")) { console.log(`skip ${name} (fixtures/ not generated)`); continue; }
  const d = bundle(fx(name + ".tirs"));
  const rows = d.y_train.length, features = d.x_train.length / rows, n = d.x_test.length / features;
  const members = Object.keys(d).filter((k) => /^member\d+_method$/.test(k)).length;
  const table = await ask("table", { text: csvOf(d.x_train, d.y_train, rows, features), target: "y", regression: false, estimators: members });
  const store = app.memoryStore();
  const direct = { call: (_rec, op, args) => ask(op, args), ready: async () => [{}] };
  const { entry } = await app.fitModel(direct, store, { kind: "classifier", weightsBlob: weightBlobs.classifier, table, name, target: "y" });
  const { x } = await ask("features", { schema: entry.schema, text: csvOf(d.x_test, null, n, features) });
  const res = await app.predictModel(direct, store, { model: entry, weightsBlob: weightBlobs.classifier, x, rows: n });
  const bad = worst(res.proba, d.proba);
  const info = await ask("describe", { text: csvOf(d.x_train, d.y_train, rows, features) });
  const est = app.memberBytes({ rows, features: app.usefulColumns(info, features), classes: table.classes });
  check(bad < 2e-3, `${name} through the page's path (CSV in, ${members} members one at a time, half weights, compact storage): worst probability difference from the reference ${bad.toExponential(2)}`);
  check(Math.abs(est * members / entry.bytes - 1) < 0.02, `${name}: stored size estimated ${app.formatBytes(est * members)}, actual ${app.formatBytes(entry.bytes)}`);
}

// 3c. regression through the page's path
{
  const d = bundle(fx("reg-a.tirs"));
  const rows = d.y_train.length, features = d.x_train.length / rows, n = d.x_test.length / features;
  await ask("load", { kind: "regressor", blob: weightBlobs.regressor });
  const table = await ask("table", { text: csvOf(d.x_train, d.y_train, rows, features), target: "y", regression: true, estimators: 4 });
  const store = app.memoryStore();
  const direct = { call: (_rec, op, args) => ask(op, args), ready: async () => [{}] };
  const { entry } = await app.fitModel(direct, store, { kind: "regressor", weightsBlob: weightBlobs.regressor, table, name: "reg-a", target: "y" });
  const { x } = await ask("features", { schema: entry.schema, text: csvOf(d.x_test, null, n, features) });
  const res = await app.predictModel(direct, store, { model: entry, weightsBlob: weightBlobs.regressor, x, rows: n });
  const mean = Float64Array.from({ length: n }, (_, i) => res.values[4 * i]), median = Float64Array.from({ length: n }, (_, i) => res.values[4 * i + 1]);
  const ym = d.y_train.reduce((a, b) => a + b, 0) / rows, spread = Math.sqrt(d.y_train.reduce((a, b) => a + (b - ym) ** 2, 0) / rows);
  check(worst(mean, d.mean) < 0.01 * spread && worst(median, d.median) < 0.01 * spread, `reg-a through the page's path (4 members, half weights, compact storage): mean off by at most ${worst(mean, d.mean).toExponential(2)}, median ${worst(median, d.median).toExponential(2)} (targets spread ${spread.toFixed(1)})`);
  check(app.resultsCsv(entry, res, n).split("\n")[0] === "prediction,median,q05,q95", "regression results as CSV carry the command line's columns");
  const est = app.memberBytes({ rows, features: table.features });
  check(Math.abs(est * entry.members / entry.bytes - 1) < 0.02, `reg-a: stored size estimated ${app.formatBytes(est * entry.members)}, actual ${app.formatBytes(entry.bytes)}`);
}

// ---- 4. the pool of workers (Node worker threads standing in for Web Workers) ------------------------------
const makeWorker = () => {
  const w = new Worker(path.join(root, "tools/node-worker.mjs"), { workerData: { wasm: wasmPath, script: pathToFileURL(path.join(root, "site/worker.js")).href } });
  const face = { postMessage: (m, t) => w.postMessage(m, t), terminate: () => w.terminate() };
  w.on("message", (data) => face.onmessage && face.onmessage({ data }));
  w.on("error", (e) => face.onerror && face.onerror({ message: e.message }));
  return face;
};
{
  const silent = new app.Engine(() => ({ postMessage() {}, terminate() {} }), { bootTimeout: 300 });
  const t = Date.now();
  const err = await silent.spawn().then(() => null, (e) => e);
  check(err instanceof app.EngineError && /did not start/.test(err.message) && Date.now() - t < 2000, `an engine that never answers is reported, not waited for ("${err && err.message}")`);
  const broken = new app.Engine(() => { const f = { postMessage() {}, terminate() {} }; setTimeout(() => f.onerror({ message: "blocked" }), 10); return f; }, { bootTimeout: 5000 });
  const err2 = await broken.spawn().then(() => null, (e) => e);
  check(err2 instanceof app.EngineError && /blocked/.test(err2.message), `a worker whose script fails to load is reported ("${err2 && err2.message}")`);
}
{
  // a synthetic table at the page's comfortable limit, 500 rows x 100 columns, 3 classes, 4 members
  let s = 99;
  const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  const rows = 500, features = 100, members = 4;
  const x = Float64Array.from({ length: rows * features }, () => rnd() * 10 - 5);
  const y = Int32Array.from({ length: rows }, (_, r) => (x[r * features] + x[r * features + 1] > 1 ? 2 : x[r * features + 2] > 0 ? 1 : 0));
  const text = csvOf(x, y, rows, features), fresh = csvOf(x.subarray(0, 40 * features), null, 40, features);
  const run = async (workers) => {
    const engine = new app.Engine(makeWorker);
    const store = app.memoryStore();
    const first = await engine.first();
    const table = await engine.call(first, "table", { text, target: "y", regression: false, estimators: members });
    let ticks = 0;
    const { entry, stats } = await app.fitModel(engine, store, { kind: "classifier", weightsBlob: weightBlobs.classifier, table, workers, name: "pool", target: "y", onProgress: () => ticks++ });
    const { x: xt } = await engine.call(first, "features", { schema: entry.schema, text: fresh });
    const t = performance.now();
    const res = await app.predictModel(engine, store, { model: entry, weightsBlob: weightBlobs.classifier, x: xt, rows: 40, workers });
    const predictMs = performance.now() - t;
    const used = engine.workers.length;
    engine.stop();
    return { entry, stats, res, ticks, predictMs, used, table };
  };
  const one = await run(1), four = await run(4);
  check(four.used === 4 && one.used === 1 && four.ticks === members + 1, "the pool starts one worker per member and reports progress as each finishes");
  check(worst(one.res.proba, four.res.proba) === 0, "4 workers in parallel give exactly the probabilities 1 worker gives");
  const speed = one.stats.ms / four.stats.ms;
  const slow = (four.stats.memberMs.reduce((a, b) => a + b) / members) / (one.stats.memberMs.reduce((a, b) => a + b) / members);
  console.log(`     500 x 100, 4 members: 1 worker ${(one.stats.ms / 1000).toFixed(1)} s, 4 workers ${(four.stats.ms / 1000).toFixed(1)} s (${speed.toFixed(2)}x faster incl. loading weights; each member ${slow.toFixed(2)}x slower when 4 run at once; the page assumes ${app.crowding(4).toFixed(2)}x)`);
  console.log(`     predicting 40 rows: 1 worker ${one.predictMs.toFixed(0)} ms, 4 workers ${four.predictMs.toFixed(0)} ms`);
  check(speed > 1.5, "4 workers are clearly faster than 1");
  const table = { rows, features, classes: 3 };
  const e1 = app.estimate({ ...table, members, workers: 1, rate }), e4 = app.estimate({ ...table, members, workers: 4, rate });
  const memberS = one.stats.memberMs.reduce((a, b) => a + b) / members / 1000;
  console.log(`     estimate for this table: ${e1.memberSeconds.toFixed(1)} s a member (measured ${memberS.toFixed(1)} s); 4 workers ${e4.seconds.toFixed(1)} s in all (measured ${(four.stats.ms / 1000).toFixed(1)} s incl. loading weights); memory per worker ${app.formatBytes(e1.perWorkerBytes)} (measured ${app.formatBytes(four.stats.mem)}); storage ${app.formatBytes(e1.storageBytes)} (actual ${app.formatBytes(one.entry.bytes)})`);
  check(e1.memberSeconds / memberS > 0.4 && e1.memberSeconds / memberS < 2.5, "time estimate is within a factor of 2.5 of the measured fit");
  check(e1.perWorkerBytes >= four.stats.mem && e1.perWorkerBytes < 2 * four.stats.mem, "memory estimate is at least the measured peak and under twice it");
  check(Math.abs(e1.storageBytes / one.entry.bytes - 1) < 0.02, "storage estimate is within 2% of the fitted model's real size");
  // expiry and deletion
  const store = app.memoryStore(), cat = { weights: {}, models: [] };
  for (const [id, expires] of [["old", 1000], ["new", 0]]) { await store.put(`model-${id}-m0`, new Blob(["x"])); cat.models.push({ id, name: id, members: 1, expires }); }
  await store.put("model-orphan-m0", new Blob(["x"]));
  await app.writeCatalogue(store, cat);
  const removed = await app.sweep(store, cat, 5000);
  const back = await app.readCatalogue(store);
  check(removed.join() === "old" && back.models.length === 1 && back.models[0].id === "new" && (await store.keys()).sort().join() === "catalogue.json,model-new-m0", "expired models and stray pieces are removed from storage; the rest stay");
}

console.log(failures ? `\n${failures} check(s) FAILED` : "\nall checks passed");
process.exit(failures ? 1 : 0);
