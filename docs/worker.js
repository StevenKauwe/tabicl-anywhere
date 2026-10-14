// BrowserTabICL worker: one WebAssembly engine on its own thread.
//
// The page starts several of these. Each holds its own copy of the model weights and does one job at a time:
// read a CSV table, fit ONE ensemble member, or predict with one member. Nothing here touches the network
// except loading the engine file that sits next to this script.
//
// Messages in:  { id, op, ...arguments }
// Messages out: { id, ok: true, result }  or  { id, ok: false, error, fatal }
//               (fatal: this worker's engine is no longer usable, e.g. it ran out of memory; start a new one)
import init, * as engine from "./pkg/browsertabicl_wasm.js";

let started = null;
const held = {}; // "classifier" | "regressor" -> Weights

/** Start the engine once. `source` (optional) is the .wasm as bytes, a URL or a Response; the default is the file next to the script. */
export function boot(source) {
  if (!started) started = source ? init({ module_or_path: source }) : init();
  return started;
}

const mem = () => engine.memoryBytes();
const bytesOf = async (blob) => new Uint8Array(await blob.arrayBuffer());

function weights(kind) {
  if (!held[kind]) throw new Error(`the ${kind} model is not loaded in this worker`);
  return held[kind];
}

const ops = {
  /** Is the engine alive, and how fast is this machine? */
  hello({ calibrateMs = 0 }) {
    // The best of many short bursts. Short, because a browser only swaps in its fully optimised build of the
    // engine between calls (one long call would measure the slow first build); best, so that something else
    // briefly using the processor cannot drag the figure down.
    let rate = 0;
    for (let i = 0; i < 16 && calibrateMs > 0; i++) rate = Math.max(rate, engine.calibrate(calibrateMs / 16));
    return { result: { rate, mem: mem() } };
  },

  /** Column summary of CSV text. */
  describe({ text }) {
    return { result: JSON.parse(engine.describeCsv(text)) };
  },

  /** Read a training table: numbers, target, schema, and how many ensemble members it really supports. */
  table({ text, target, regression, estimators }) {
    const t = new engine.Table(text, target, regression);
    try {
      const x = t.x, y = regression ? t.yNumbers : t.yClasses;
      const result = { rows: t.rows, features: t.features, classes: t.classes, labels: JSON.parse(t.labels), schema: t.schema, members: t.members(estimators), x, y };
      return { result, transfer: [x.buffer, y.buffer] };
    } finally {
      t.free();
    }
  },

  /** Features of new rows, using the schema stored with a fitted model. */
  features({ schema, text }) {
    const x = engine.featuresFromCsv(schema, text);
    return { result: { x }, transfer: [x.buffer] };
  },

  /** Turn a downloaded checkpoint into the half-size weight file this page stores. */
  async convert({ blob }) {
    const out = engine.toBundle(await bytesOf(blob), true);
    return { result: { bytes: out }, transfer: [out.buffer] };
  },

  /** Load model weights (kept until another kind is loaded). */
  async load({ kind, blob }) {
    if (!held[kind]) {
      for (const k of Object.keys(held)) { held[k].free(); delete held[k]; }
      const w = new engine.Weights(await bytesOf(blob));
      if (w.isRegression !== (kind === "regressor")) { w.free(); throw new Error(`these weights are not a ${kind}`); }
      held[kind] = w;
    }
    return { result: { mem: mem() } };
  },

  /** Fit one member of the ensemble; returns the fitted member as bytes. */
  fit({ kind, x, y, rows, features, estimators, member, kv, schema }) {
    const t = performance.now();
    const w = weights(kind);
    const bytes = kind === "regressor"
      ? w.fitRegressorMember(x, y, rows, features, estimators, member, kv, schema || "")
      : w.fitClassifierMember(x, y, rows, features, estimators, member, kv, schema || "");
    return { result: { bytes, ms: performance.now() - t, mem: mem() }, transfer: [bytes.buffer] };
  },

  /** Predict with one fitted member: score sums (classifier) or mean/median/5%/95% per row (regressor). */
  async predict({ kind, blob, x, rows }) {
    const t = performance.now();
    const fitted = await bytesOf(blob);
    const out = kind === "regressor" ? weights(kind).predictRegression(fitted, x, rows) : weights(kind).predictSums(fitted, x, rows);
    return { result: { out, ms: performance.now() - t, mem: mem() }, transfer: [out.buffer] };
  },

  /** Members' score sums (already added together in member order) into probabilities. */
  combine({ sums, classes, members, temperature }) {
    const p = engine.combineProba(sums, classes, members, temperature ?? engine.defaultTemperature());
    return { result: { proba: p }, transfer: [p.buffer] };
  },
};

/** Run one message; never throws. */
export async function handle(msg, wasmSource) {
  try {
    await boot(wasmSource);
  } catch (e) {
    return { id: msg.id, ok: false, fatal: true, error: `the engine file could not be loaded (${e && e.message ? e.message : e})` };
  }
  try {
    const op = ops[msg.op];
    if (!op) throw new Error(`unknown request ${msg.op}`);
    const { result, transfer } = await op(msg);
    return { id: msg.id, ok: true, result, transfer };
  } catch (e) {
    if (globalThis.__BROWSERTABICL_DEBUG) console.error(e && e.stack);
    const text = String(e && e.message ? e.message : e);
    // The engine stops dead (rather than unwinding) when it cannot get memory or hits an internal error.
    const fatal = (typeof WebAssembly !== "undefined" && e instanceof WebAssembly.RuntimeError) || e instanceof RangeError || /unreachable|out of memory|memory access|allocation/i.test(text);
    return { id: msg.id, ok: false, fatal, error: fatal ? `the engine ran out of memory or stopped unexpectedly (${text})` : text };
  }
}

// Inside a real worker: answer messages. (Under Node the test harness imports `handle` instead.)
if (typeof self !== "undefined" && typeof self.postMessage === "function" && typeof document === "undefined") {
  self.onmessage = async (ev) => {
    const { transfer, ...reply } = await handle(ev.data, globalThis.__BROWSERTABICL_WASM);
    self.postMessage(reply, transfer || []);
  };
}
