// BrowserTabICL worker: one WebAssembly engine on its own thread.
//
// The page starts several of these. Each holds its own copy of the model weights and does one job at a time:
// read a CSV table, fit ONE ensemble member, or predict with one member. Nothing here touches the network
// except loading the engine file that sits next to this script.
//
// One of them may also be the graphics-card worker: after a `card` request it holds the browser's WebGPU device
// and does fit and predict requests marked `card: true` there, one member after another (a card does one job at
// a time anyway). A failure on the card is reported with `card: true` in the reply, so the page can tell it from
// a problem with the table and hand the job to the processor workers instead.
//
// Messages in:  { id, op, ...arguments }
// Messages out: { id, ok: true, result }  or  { id, ok: false, error, fatal }
//               (fatal: this worker's engine is no longer usable, e.g. it ran out of memory; start a new one)
import init, * as engine from "./pkg/tabicl_anywhere_wasm.js";

let started = null;
let build = null; // which of BUILDS is running
const held = {}; // "classifier" | "regressor" -> Weights
let card = null; // the engine's Card, once a `card` request has opened one
let pretendFailure = false; // for trying out the page's fallback: the next card job fails as a lost card would

// The engine comes in three builds of the same code (tools/build-wasm.sh), fastest first. A browser refuses a
// whole engine file if it holds even one instruction the browser does not know, so the build is chosen before
// anything is downloaded, by asking the browser whether it accepts two tiny test modules: one with an
// instruction from "relaxed SIMD" (Chrome and Edge among others; not Safari, so no browser on an iPhone or iPad), one
// with an instruction from plain SIMD (every current browser; Safari since iOS 16.4 and macOS 13.3). Only the
// chosen file is downloaded (the workers share it through the browser's cache).
export const BUILDS = {
  relaxed: "tabicl_anywhere_wasm_relaxed.wasm", // fused multiply-add: about 15% quicker to fit
  simd: "tabicl_anywhere_wasm_bg.wasm", // four numbers at a time
  scalar: "tabicl_anywhere_wasm_scalar.wasm", // one number at a time: about 3 times slower, for old browsers
};
const v128zero = [0xfd, 0x0c, ...new Array(16).fill(0)]; // v128.const 0
/** A module with one function, `() -> ()`, whose body (after its empty list of locals) is `code`. */
function probe(code) {
  const body = [0x00, ...code, 0x0b];
  return new Uint8Array([0x00, 0x61, 0x73, 0x6d, 1, 0, 0, 0, 1, 4, 1, 0x60, 0, 0, 3, 2, 1, 0, 0x0a, body.length + 2, 1, body.length, ...body]);
}
export const PROBES = {
  relaxed: probe([...v128zero, ...v128zero, ...v128zero, 0xfd, 0x85, 0x02, 0x1a]), // f32x4.relaxed_madd; drop
  simd: probe([...v128zero, 0x1a]), // v128.const; drop
  scalar: probe([]),
};

/** The builds this browser accepts, fastest first. `validate` stands in for WebAssembly.validate in tests. */
export function usableBuilds(validate = (b) => WebAssembly.validate(b)) {
  return Object.keys(BUILDS).filter((name) => {
    try { return validate(PROBES[name]); } catch { return false; }
  });
}

/**
 * Start the engine once. `source` (optional) is the .wasm as bytes, a URL or a Response (tests pass one);
 * otherwise the fastest build this browser accepts is loaded from ./pkg/. `engine=relaxed|simd|scalar` in the
 * address of this script forces one build (the page passes on its own `?engine=`, for trying them out).
 */
export function boot(source) {
  if (!started) started = source ? init({ module_or_path: source }).then(() => { build = "given"; }) : bootBest();
  return started;
}

async function bootBest() {
  let forced = null;
  try { forced = new URL(import.meta.url).searchParams.get("engine"); } catch { /* no address */ }
  const list = forced && BUILDS[forced] ? [forced] : usableBuilds();
  if (!list.length) throw new Error("this browser accepts none of the engine's builds: it is too old for WebAssembly as this page uses it");
  const tried = [];
  // normally the first works; the others are a safety net (each is only fetched if the one before failed)
  for (const name of list) {
    try {
      await init({ module_or_path: new URL(`./pkg/${BUILDS[name]}`, import.meta.url) });
      build = name;
      return;
    } catch (e) {
      tried.push(`${name} build: ${e && e.message ? e.message : e}`);
    }
  }
  throw new Error(tried.join("; "));
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
    return { result: { rate, mem: mem(), build } };
  },

  /** Open the graphics card (once) and measure it. Fails where the browser offers no WebGPU or no usable card. */
  async card({ workMB = 768, modelsMB = 1024, calibrateMs = 300, limitMB = 0, failNext = false }) {
    pretendFailure = failNext;
    if (typeof navigator === "undefined" || !navigator.gpu) throw new Error("this browser offers no graphics card to web pages (no WebGPU)");
    const t = performance.now();
    if (!card) {
      if (!(await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }))) throw new Error("the browser offers WebGPU but found no graphics card it can use");
      card = await engine.Card.open(workMB, modelsMB);
    }
    const opened = performance.now() - t;
    const rate = calibrateMs > 0 ? await card.calibrate(calibrateMs) : 0;
    if (limitMB > 0) card.setWorkMb(limitMB); // for trying out what happens when the card runs short
    // the browser's own words for the card (the engine sees only what WebGPU tells it, often nothing)
    let adapter = "", software = card.software;
    try {
      const a = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      const i = (a && a.info) || {};
      adapter = [i.vendor, i.architecture, i.description || i.device].filter(Boolean).join(" ");
      // a program standing in for a card (Chromium's "swiftshader", Mesa's "llvmpipe")
      software = software || !!(a && (a.isFallbackAdapter || i.isFallbackAdapter)) || /swiftshader|llvmpipe|software/i.test(adapter);
    } catch { /* not told */ }
    return { result: { name: card.name, adapter, software, rate, openMs: opened, ms: performance.now() - t } };
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

  /** Fit one member of the ensemble; returns the fitted member as bytes. With `card`, on the graphics card. */
  async fit({ kind, x, y, rows, features, estimators, member, kv, schema, card: onCard }) {
    const t = performance.now();
    const w = weights(kind);
    if (onCard) {
      const bytes = await onTheCard(() => kind === "regressor"
        ? w.fitRegressorMemberCard(theCard(), x, y, rows, features, estimators, member, kv, schema || "")
        : w.fitClassifierMemberCard(theCard(), x, y, rows, features, estimators, member, kv, schema || ""));
      return { result: { bytes, ms: performance.now() - t, mem: mem(), card: true }, transfer: [bytes.buffer] };
    }
    const bytes = kind === "regressor"
      ? w.fitRegressorMember(x, y, rows, features, estimators, member, kv, schema || "")
      : w.fitClassifierMember(x, y, rows, features, estimators, member, kv, schema || "");
    return { result: { bytes, ms: performance.now() - t, mem: mem() }, transfer: [bytes.buffer] };
  },

  /** Predict with one fitted member: score sums (classifier) or mean/median/5%/95% per row (regressor). */
  async predict({ kind, blob, x, rows, card: onCard, key }) {
    const t = performance.now();
    const fitted = await bytesOf(blob);
    if (onCard) {
      const r = await onTheCard(() => weights(kind).predictCard(theCard(), key, fitted, x, rows));
      return { result: { out: r.out, ms: performance.now() - t, mem: mem(), cardLayers: r.cardLayers, problem: r.problem }, transfer: [r.out.buffer] };
    }
    const out = kind === "regressor" ? weights(kind).predictRegression(fitted, x, rows) : weights(kind).predictSums(fitted, x, rows);
    return { result: { out, ms: performance.now() - t, mem: mem() }, transfer: [out.buffer] };
  },

  /** Members' score sums (already added together in member order) into probabilities. */
  combine({ sums, classes, members, temperature }) {
    const p = engine.combineProba(sums, classes, members, temperature ?? engine.defaultTemperature());
    return { result: { proba: p }, transfer: [p.buffer] };
  },
};

function theCard() {
  if (!card) throw new Error("the graphics card has not been opened in this worker");
  return card;
}

/** Run a job on the card; anything that goes wrong is marked as the card's failure. */
async function onTheCard(job) {
  try {
    if (pretendFailure) {
      pretendFailure = false;
      throw new Error("the device was lost (pretended, as asked for by the page address)");
    }
    return await job();
  } catch (e) {
    const err = new Error(`the graphics card failed (${e && e.message ? e.message : e})`);
    err.card = true;
    throw err;
  }
}

/** Run one message; never throws. */
export async function handle(msg, wasmSource) {
  try {
    await boot(wasmSource);
  } catch (e) {
    return { id: msg.id, ok: false, fatal: true, boot: true, error: `the engine file could not be loaded (${e && e.message ? e.message : e})` };
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
    return { id: msg.id, ok: false, fatal, card: !!(e && e.card), error: fatal ? `the engine ran out of memory or stopped unexpectedly (${text})` : text };
  }
}

// Inside a real worker: answer messages. (Under Node the test harness imports `handle` instead.)
if (typeof self !== "undefined" && typeof self.postMessage === "function" && typeof document === "undefined") {
  self.onmessage = async (ev) => {
    const { transfer, ...reply } = await handle(ev.data, globalThis.__BROWSERTABICL_WASM);
    self.postMessage(reply, transfer || []);
  };
}
