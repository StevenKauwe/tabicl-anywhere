// BrowserTabICL page logic. One module: the first half is plain logic with no page in it (sizes and times,
// fetching and joining the weight parts, browser storage, the pool of workers, fitting and predicting member
// by member) and is also run under Node by tools/test-site.mjs; the second half wires it to the page.

// =====================================================================================================
// Estimates
// =====================================================================================================

/** Above these the page warns (and still lets the visitor continue). */
export const SOFT_LIMITS = { rows: 500, columns: 100 };

/** Facts about the TabICL v2 network that fix how big a fitted model is. */
const NET = { layers: 12, width: 512, colBlocks: 3, colWidth: 128, inducing: 128, chunk: 256, nativeClasses: 10 };

/**
 * Work to fit one ensemble member, in units of one multiply-add of the engine's inner loop:
 *   levels * (cell * rows * columns + pair * rows^2 + row * rows) + fixed
 * The four numbers were fitted to WebAssembly fits of 21 tables from 50 x 3 to 3000 x 8 and 1000 x 50
 * (they reproduce those within about a third). They describe the program, not a machine: dividing by the speed
 * the worker's `hello` request measures in the visitor's own browser (best of 16 short `calibrate()` bursts) gives
 * seconds.
 */
export const WORK = { cell: 2.27e6, pair: 2.14e4, row: 2.79e7, fixed: 1.64e9 };

/** Memory one worker holds once the weights are loaded (measured: 169 MB classifier, 175 MB regressor). */
const WEIGHTS_MEMORY = 176e6;
/** WebAssembly (32-bit) cannot address more than 4 GB; leave headroom. */
const WASM_LIMIT = 3.6e9;

/** With more than 10 classes the model answers in several steps; each step stores the rows again. */
export function levels(classes) {
  let n = 1;
  for (let cap = NET.nativeClasses; cap < classes; cap *= NET.nativeClasses) n++;
  return n;
}

/** Bytes of one fitted ensemble member. `kv` is how the big part is stored: "i16" (2 bytes a number), "f32" (4) or "i8" (1). */
export function memberBytes({ rows, features, classes = 0, kv = "i16" }) {
  const width = { f32: 4, i16: 2, i8: 1 }[kv];
  const lv = levels(classes);
  // per layer: keys and values, 2 x rows x 512 numbers, stored in chunks of 256 rows; the compact formats add
  // 2 x 512 scale factors (4 bytes each) per chunk
  const chunks = Math.ceil(rows / NET.chunk);
  const layer = 2 * rows * NET.width * width + (kv === "f32" ? 0 : chunks * 8 * NET.width);
  const rowPart = NET.layers * (layer + NET.width * 4);
  // per column: 3 blocks x (keys + values) x 128 x 128 numbers, always 4 bytes each (about 393 KB)
  const colPart = features * NET.colBlocks * 2 * NET.colWidth * NET.inducing * 4;
  return lv * (rowPart + colPart) + 8192 + features * 256;
}

/** Columns the model will actually learn from: all but the target and any column holding a single value (those are dropped). */
export function usefulColumns(info, targetIndex) {
  return Math.max(1, info.columns.filter((c, i) => i !== targetIndex && c.distinct > 1).length);
}

/** How many workers to run: one per member, within the processor's threads and the machine's memory. */
export function planWorkers({ members, cores = 4, deviceMemoryGB = 4, perWorkerBytes = WEIGHTS_MEMORY }) {
  const byCores = Math.max(1, Math.min(8, cores > 2 ? cores - 1 : cores));
  const byMemory = Math.max(1, Math.floor((deviceMemoryGB * 1e9 * 0.5) / perWorkerBytes));
  return Math.max(1, Math.min(members, byCores, byMemory));
}

/** Running several workers at once slows each a little (they share memory bandwidth and processor cache). */
export const crowding = (workers) => 1 + 0.06 * (workers - 1);

/**
 * What fitting this table will cost. `rate` is multiply-adds per second from `calibrate()`; `adjust` is the
 * ratio of measured to estimated time remembered from earlier fits in this browser (1 if none).
 */
export function estimate({ rows, features, classes = 0, members = 1, workers = 1, kv = "i16", rate, adjust = 1 }) {
  const lv = levels(classes);
  const one = memberBytes({ rows, features, classes, kv });
  // while fitting, a worker holds: the weights, every cell's embedding (rows x columns x 128 numbers), the member
  // being written, and working arrays that grow with the rows; the page then holds the finished member too
  const perWorker = WEIGHTS_MEMORY + lv * rows * features * NET.colWidth * 4 + one + rows * 14000;
  const work = lv * (WORK.cell * rows * features + WORK.pair * rows * rows + WORK.row * rows) + WORK.fixed;
  const memberSeconds = rate > 0 ? (adjust * work) / rate : NaN;
  // plus about a second for each worker to start and read the weights (they do so side by side)
  const seconds = memberSeconds * Math.ceil(members / workers) * crowding(workers) + 1;
  let refuse = null;
  if (perWorker > WASM_LIMIT || one > 2e9) {
    refuse = `This table is too large to fit in a browser. Fitting it needs about ${formatBytes(perWorker)} in one piece, and a web page's engine cannot use more than 4 GB. Use fewer rows or columns, or the command-line version, which works from disk.`;
  }
  return { memberBytes: one, storageBytes: one * members, perWorkerBytes: perWorker, memoryBytes: (perWorker + one) * Math.min(workers, members), memberSeconds, seconds, levels: lv, refuse };
}

/**
 * Fitting one member on the graphics card, in two parts:
 *   card:  levels * pair * rows^2 + fixed                    in multiply-adds of the card
 *   host:  levels * (cell * rows * columns + row * rows)     in multiply-adds of one processor thread
 * The card does the network, and for tables of browser size most of its time is the fixed cost of a few
 * hundred small steps; the worker that drives it prepares the cells and packs what comes back, on one processor
 * thread, which is what grows with the table. Each part is divided by its own measured speed (`card` request:
 * the card's attention; `hello`: the processor's inner loop), so the numbers describe the program, not a
 * machine. Fitted to fits in Chromium on an RTX 3090 of tables from 50 x 3 to 2000 x 10 and 500 x 100 (within
 * about a third, except the smallest tables, which come out quicker than this says; see site/README.md).
 */
export const CARD_WORK = { pair: 380, fixed: 5.0e9 };
export const CARD_HOST = { cell: 1.0e5, row: 5.8e5 };
/** Seconds, once per page and model, for the card worker to read the weights and put them on the card. */
const CARD_START = 0.7;

/**
 * What fitting costs on the graphics card: members one after another. `cardRate` from the `card` request,
 * `rate` the processor's; `adjust` as in `estimate`, kept separately for the card; `started`: the card worker
 * already holds these weights.
 */
export function estimateCard({ rows, features, classes = 0, members = 1, cardRate, rate, adjust = 1, started = false }) {
  const lv = levels(classes);
  const card = lv * CARD_WORK.pair * rows * rows + CARD_WORK.fixed;
  const host = lv * (CARD_HOST.cell * rows * features + CARD_HOST.row * rows);
  const memberSeconds = cardRate > 0 && rate > 0 ? adjust * (card / cardRate + host / rate) : NaN;
  return { memberSeconds, seconds: memberSeconds * members + (started ? 0 : CARD_START) };
}

/**
 * The most members (8, 4, 2 or 1) that fit in about half a minute here; fewer for big tables. With `card`
 * (the arguments of `estimateCard` other than the table), by the card's estimate.
 */
export function defaultMembers(table, { cores, deviceMemoryGB, rate, adjust, card = null }) {
  for (const members of [8, 4, 2]) {
    const probe = estimate({ ...table, members, workers: 1, rate, adjust });
    if (probe.refuse) continue;
    if (card) {
      if (estimateCard({ ...table, members, rate, ...card }).seconds <= 30) return members;
      continue;
    }
    const workers = planWorkers({ members, cores, deviceMemoryGB, perWorkerBytes: probe.perWorkerBytes });
    const e = estimate({ ...table, members, workers, rate, adjust });
    if (e.seconds <= 30) return members;
  }
  return 1;
}

/**
 * Whether to predict on the graphics card (in "auto"): it costs about a tenth of a second per member to start,
 * so it pays from about 10,000 training cells (rows x columns) or 1,000 new rows (measured in Chromium on an
 * RTX 3090; below that the processor workers answer sooner).
 */
export function cardForPrediction({ choice = "auto", card = null, trainRows, features, rows }) {
  if (!card || choice === "processor") return false;
  if (choice === "card") return true;
  return !card.software && (trainRows * features >= 10000 || rows >= 1000);
}

/**
 * Where to fit: the graphics card or the processor, and why, in a sentence for the page. `choice` is the
 * visitor's ("auto", "card", "processor"); `card` describes the card (`null`: none usable, with `cardWhy`).
 */
export function chooseDevice({ choice = "auto", card = null, cardWhy = "", kind, classes = 0, cardSeconds, cpuSeconds }) {
  if (!card) return { useCard: false, why: cardWhy };
  if (choice === "processor") return { useCard: false, why: "you chose the processor" };
  if (choice === "card") return { useCard: true, why: "you chose the graphics card" };
  if (card.software) return { useCard: false, why: "this browser offers only a software stand-in for a graphics card, which is slower than the processor" };
  if (kind === "classifier" && classes > NET.nativeClasses) return { useCard: false, why: `with more than ${NET.nativeClasses} classes part of the work stays on a single processor thread beside the card, so the processor workers are quicker` };
  if (cardSeconds < cpuSeconds) return { useCard: true, why: "it should be quicker than the processor" };
  return { useCard: false, why: "for this table the processor should be as quick or quicker" };
}

export function formatBytes(n) {
  if (!(n >= 0)) return "?";
  if (n < 1e6) return `${Math.max(1, Math.round(n / 1e3))} KB`;
  if (n < 1e9) return `${n < 1e7 ? (n / 1e6).toFixed(1) : Math.round(n / 1e6)} MB`;
  return `${(n / 1e9).toFixed(1)} GB`;
}

export function formatDuration(s) {
  if (!(s >= 0)) return "an unknown time";
  if (s < 1.5) return "about a second";
  if (s < 90) return `about ${s < 20 ? Math.round(s) : Math.round(s / 5) * 5} seconds`;
  if (s < 5400) return `about ${Math.round(s / 60)} minutes`;
  return `about ${(s / 3600).toFixed(1)} hours`;
}

// =====================================================================================================
// Weights: one configurable location
// =====================================================================================================

const CHECKPOINT = { classifier: "tabicl-classifier-v2-20260212.ckpt", regressor: "tabicl-regressor-v2-20260212.ckpt" };

/**
 * Where the weights for `kind` come from, given the page's one setting `base`:
 *  - a folder (the default, "weights/"): our own half-size copy, cut into parts of at most 14 MB and listed in
 *    `<kind>-v2-half.json` (made by `tabicl-anywhere split`);
 *  - an address on huggingface.co: the checkpoint as published (110 MB), converted to half size in the browser.
 */
export function weightsSource(base, kind, pageUrl) {
  const root = new URL(base.endsWith("/") ? base : base + "/", pageUrl);
  if (/(^|\.)huggingface\.co$/.test(root.hostname)) return { url: new URL(CHECKPOINT[kind], root).href, parts: false, id: `hf-${CHECKPOINT[kind]}` };
  return { url: new URL(`${kind}-v2-half.json`, root).href, parts: true, id: `parts-${kind}-v2-half` };
}

async function readBody(response, onBytes) {
  if (!response.body || !response.body.getReader) {
    const all = new Uint8Array(await response.arrayBuffer());
    onBytes(all.length);
    return [all];
  }
  const reader = response.body.getReader(), chunks = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return chunks;
    chunks.push(value);
    onBytes(value.length);
  }
}

/**
 * Download the weights. Returns `{ blob, convert }`; `convert` is true when the blob is a published checkpoint
 * that still has to be turned into a weight file. `onProgress(done, total)` is called as bytes arrive.
 */
export async function fetchWeights(source, { fetchFn = fetch, onProgress = () => {} } = {}) {
  const get = async (url) => {
    let r;
    try {
      r = await fetchFn(url);
    } catch (e) {
      throw new Error(`could not reach ${url} (${e.message || e})`);
    }
    if (!r.ok) throw new Error(`${url} answered ${r.status}`);
    return r;
  };
  if (!source.parts) {
    const r = await get(source.url);
    const total = Number(r.headers.get("content-length")) || 0;
    let done = 0;
    const chunks = await readBody(r, (n) => onProgress((done += n), total));
    return { blob: new Blob(chunks), convert: true };
  }
  const list = await (await get(source.url)).json();
  if (list.format !== "browsertabicl-parts-1" || !Array.isArray(list.parts)) throw new Error(`${source.url} is not a list of weight parts`);
  let done = 0;
  const chunks = [];
  for (const part of list.parts) {
    let got = 0;
    const body = await readBody(await get(new URL(part.file, source.url).href), (n) => { got += n; onProgress((done += n), list.bytes); });
    if (got !== part.bytes) throw new Error(`${part.file} arrived as ${got} bytes, expected ${part.bytes}; the download was cut short`);
    chunks.push(...body);
  }
  const blob = new Blob(chunks);
  const magic = new TextDecoder().decode(await blob.slice(0, 8).arrayBuffer());
  if (blob.size !== list.bytes || magic !== "TIRSBNDL") throw new Error("the downloaded parts do not add up to a weight file");
  return { blob, convert: false };
}

// =====================================================================================================
// Storage in the browser: named blobs, plus one small catalogue of what is stored
// =====================================================================================================

/** Kept only until the page closes: used when the browser offers no storage (some private windows). */
export function memoryStore() {
  const m = new Map();
  return {
    kind: "memory",
    async put(key, data) { m.set(key, data instanceof Blob ? data : new Blob([data])); },
    async get(key) { return m.get(key) || null; },
    async del(key) { m.delete(key); },
    async keys() { return [...m.keys()]; },
  };
}

/** The browser's private file system for this site: best for large files. */
async function fileStore() {
  const root = await navigator.storage.getDirectory();
  const dir = await root.getDirectoryHandle("browsertabicl", { create: true });
  const store = {
    kind: "files",
    async put(key, data) {
      const w = await (await dir.getFileHandle(key, { create: true })).createWritable();
      await w.write(data);
      await w.close();
    },
    async get(key) {
      try {
        return await (await dir.getFileHandle(key)).getFile();
      } catch (e) {
        if (e && e.name === "NotFoundError") return null;
        throw e;
      }
    },
    async del(key) { try { await dir.removeEntry(key); } catch (e) { if (!e || e.name !== "NotFoundError") throw e; } },
    async keys() { const out = []; for await (const name of dir.keys()) out.push(name); return out; },
  };
  await store.put("probe", new Blob(["ok"])); // some browsers offer the folder but not writing from a page
  await store.del("probe");
  return store;
}

/** IndexedDB: the fallback where the private file system cannot be written from a page. */
async function databaseStore() {
  const db = await new Promise((resolve, reject) => {
    const open = indexedDB.open("browsertabicl", 1);
    open.onupgradeneeded = () => open.result.createObjectStore("blobs");
    open.onsuccess = () => resolve(open.result);
    open.onerror = () => reject(open.error);
    open.onblocked = () => reject(new Error("storage is blocked by another tab"));
  });
  const run = (mode, f) => new Promise((resolve, reject) => {
    const tx = db.transaction("blobs", mode);
    const req = f(tx.objectStore("blobs"));
    tx.oncomplete = () => resolve(req.result);
    tx.onerror = tx.onabort = () => reject(tx.error || req.error);
  });
  const store = {
    kind: "database",
    put: (key, data) => run("readwrite", (s) => s.put(data instanceof Blob ? data : new Blob([data]), key)),
    get: async (key) => (await run("readonly", (s) => s.get(key))) || null,
    del: (key) => run("readwrite", (s) => s.delete(key)),
    keys: () => run("readonly", (s) => s.getAllKeys()),
  };
  await store.put("probe", new Blob(["ok"]));
  await store.del("probe");
  return store;
}

export async function openStore() {
  try {
    if (navigator.storage && navigator.storage.getDirectory) return await fileStore();
  } catch { /* fall through */ }
  try {
    if (typeof indexedDB !== "undefined") return await databaseStore();
  } catch { /* fall through */ }
  return memoryStore();
}

const CATALOGUE = "catalogue.json";

/** `{ weights: { kind: { id, bytes, saved } }, models: [ ... ] }` */
export async function readCatalogue(store) {
  try {
    const blob = await store.get(CATALOGUE);
    const c = blob ? JSON.parse(await blob.text()) : {};
    return { weights: c.weights || {}, models: Array.isArray(c.models) ? c.models : [] };
  } catch {
    return { weights: {}, models: [] };
  }
}

export const writeCatalogue = (store, c) => store.put(CATALOGUE, new Blob([JSON.stringify(c)]));
const memberKey = (id, i) => `model-${id}-m${i}`;

export async function deleteModel(store, catalogue, id) {
  const m = catalogue.models.find((x) => x.id === id);
  catalogue.models = catalogue.models.filter((x) => x.id !== id);
  await writeCatalogue(store, catalogue);
  for (let i = 0; i < (m ? m.members : 0); i++) await store.del(memberKey(id, i));
}

/** Remove fitted models whose expiry has passed, and stored pieces no catalogue entry owns. Returns the names removed. */
export async function sweep(store, catalogue, now = Date.now()) {
  const gone = catalogue.models.filter((m) => m.expires && m.expires < now);
  for (const m of gone) await deleteModel(store, catalogue, m.id);
  const owned = new Set(catalogue.models.flatMap((m) => Array.from({ length: m.members }, (_, i) => memberKey(m.id, i))));
  for (const key of await store.keys()) if (key.startsWith("model-") && !owned.has(key)) await store.del(key);
  return gone.map((m) => m.name);
}

// =====================================================================================================
// The pool of workers
// =====================================================================================================

export class EngineError extends Error {}

/**
 * A set of workers, each one engine. `makeWorker()` returns something with `postMessage`, `terminate` and
 * settable `onmessage` / `onerror` (a Web Worker). Starting a worker that never answers fails after
 * `bootTimeout` ms with a plain message instead of hanging.
 */
export class Engine {
  constructor(makeWorker, { bootTimeout = 30000 } = {}) {
    this.makeWorker = makeWorker;
    this.bootTimeout = bootTimeout;
    this.workers = [];
    this.next = 1;
  }

  async spawn(calibrateMs = 0, { pool = true } = {}) {
    let w;
    try {
      w = await this.makeWorker();
    } catch (e) {
      throw new EngineError(`this browser would not start a background worker (${e.message || e})`);
    }
    const rec = { w, pending: new Map(), loaded: null, dead: false };
    const fail = (why) => {
      rec.dead = true;
      for (const p of rec.pending.values()) { const e = new EngineError(why); e.card = rec === this.card; p.reject(e); }
      rec.pending.clear();
      this.workers = this.workers.filter((r) => r !== rec);
      if (rec === this.card) this.card = null;
      try { w.terminate(); } catch { /* already gone */ }
    };
    rec.fail = fail;
    w.onmessage = (ev) => {
      const m = ev.data, p = rec.pending.get(m.id);
      if (!p) return;
      rec.pending.delete(m.id);
      if (m.ok) return p.resolve(m.result);
      const err = new EngineError(m.error);
      err.fatal = !!m.fatal;
      err.card = !!m.card || (m.fatal && rec === this.card);
      p.reject(err);
      if (m.fatal) fail(m.error);
    };
    w.onerror = (ev) => fail(`a background worker stopped (${(ev && ev.message) || "its script could not be loaded"})`);
    w.onmessageerror = () => fail("a message to a background worker could not be delivered");
    if (pool) this.workers.push(rec);
    let timer;
    const slow = new Promise((_, reject) => { timer = setTimeout(() => reject(new EngineError(`the engine did not start within ${Math.max(1, Math.round(this.bootTimeout / 1000))} seconds`)), this.bootTimeout); });
    try {
      rec.hello = await Promise.race([this.call(rec, "hello", { calibrateMs }), slow]);
    } catch (e) {
      fail(e.message);
      throw e;
    } finally {
      clearTimeout(timer);
    }
    return rec;
  }

  call(rec, op, args = {}, transfer = []) {
    if (rec.dead) return Promise.reject(new EngineError("that worker has stopped"));
    return new Promise((resolve, reject) => {
      const id = this.next++;
      rec.pending.set(id, { resolve, reject });
      try {
        rec.w.postMessage({ id, op, ...args }, transfer);
      } catch (e) {
        rec.pending.delete(id);
        reject(new EngineError(`could not hand the job to a worker (${e.message || e})`));
      }
    });
  }

  /** At least one worker (the first is used for reading tables and small jobs). */
  async first() {
    return this.workers[0] || (await this.spawn());
  }

  /** Exactly up to `n` workers, each with the `kind` weights loaded. */
  async ready(n, kind, weightsBlob) {
    while (this.workers.length < n) await this.spawn();
    const use = this.workers.slice(0, n);
    await Promise.all(use.map(async (rec) => {
      if (rec.loaded !== kind) {
        await this.call(rec, "load", { kind, blob: weightsBlob });
        rec.loaded = kind;
      }
    }));
    return use;
  }

  /**
   * The graphics-card worker: a worker of its own, outside the pool, with the browser's card opened and
   * measured. Rejects (and keeps no worker) where there is no usable card.
   */
  async openCard(options = {}) {
    if (this.card) return this.card;
    if (!this.opening) {
      this.opening = (async () => {
        const rec = await this.spawn(0, { pool: false });
        try {
          rec.info = await this.call(rec, "card", options);
        } catch (e) {
          rec.fail(e.message);
          throw e;
        }
        this.card = rec;
        return rec;
      })().finally(() => { this.opening = null; });
    }
    return this.opening;
  }

  /** The card worker with the `kind` weights loaded. */
  async readyCard(kind, weightsBlob) {
    const rec = this.card;
    if (!rec) throw Object.assign(new EngineError("the graphics card is not available"), { card: true });
    if (rec.loaded !== kind) {
      await this.call(rec, "load", { kind, blob: weightsBlob });
      rec.loaded = kind;
    }
    return rec;
  }

  stopCard() {
    if (this.card) this.card.fail("stopped");
  }

  /** Stop workers beyond the first `keep` (frees their memory). */
  trim(keep = 1) {
    for (const rec of this.workers.slice(keep)) rec.fail("stopped to free memory");
  }

  stop() {
    this.trim(0);
    this.stopCard();
  }
}

/** Run jobs `list` (numbers) over `workers`, each worker taking the next job when it finishes one. */
async function shareOut(workers, list, run) {
  let next = 0, failed = null;
  await Promise.all(workers.map(async (rec) => {
    while (next < list.length && !failed) {
      const i = list[next++];
      try {
        await run(rec, i);
      } catch (e) {
        failed = failed || e;
      }
    }
  }));
  if (failed) throw failed;
}

/**
 * Fit a model: one ensemble member per worker at a time, each stored as it finishes.
 * `table` is what the worker's `table` request returned. Returns the catalogue entry (not yet saved in it).
 */
export async function fitModel(engine, store, { kind, weightsBlob, table, kv = "i16", workers = 1, card = false, onCardTrouble = () => {}, name, target, expiresDays = 30, onProgress = () => {}, now = Date.now() }) {
  const id = `${now.toString(36)}${Math.floor(Math.random() * 1e6).toString(36)}`;
  const members = table.members;
  const started = performance.now();
  const stats = { memberMs: [], cardMemberMs: [], mem: 0, onCard: 0, cardTrouble: null };
  let done = 0, bytes = 0;
  const one = async (rec, i, onCard) => {
    const r = await engine.call(rec, "fit", { kind, x: table.x, y: table.y, rows: table.rows, features: table.features, estimators: members, member: i, kv, schema: table.schema, card: onCard });
    await store.put(memberKey(id, i), new Blob([r.bytes]));
    bytes += r.bytes.length;
    (onCard ? stats.cardMemberMs : stats.memberMs).push(r.ms);
    if (onCard) stats.onCard++;
    stats.mem = Math.max(stats.mem, r.mem);
    onProgress({ done: ++done, members, phase: "fitting", onCard });
  };
  try {
    let rest = Array.from({ length: members }, (_, i) => i);
    onProgress({ done, members, phase: "fitting" });
    if (card) {
      // on the card, one member after another; if the card fails, what is left goes to the processor
      try {
        const rec = await engine.readyCard(kind, weightsBlob);
        while (rest.length) {
          await one(rec, rest[0], true);
          rest = rest.slice(1);
        }
      } catch (e) {
        if (!e.card) throw e;
        stats.cardTrouble = e.message;
        onCardTrouble(e);
      }
    }
    if (rest.length) {
      const pool = await engine.ready(Math.min(workers, rest.length), kind, weightsBlob);
      await shareOut(pool, rest, (rec, i) => one(rec, i, false));
    }
  } catch (e) {
    for (let i = 0; i < members; i++) await store.del(memberKey(id, i)).catch(() => {});
    throw e;
  }
  return {
    entry: {
      id, name, kind, target, kv, members, bytes, rows: table.rows, features: table.features, classes: table.classes, labels: table.labels, schema: table.schema,
      created: now, expires: expiresDays > 0 ? now + expiresDays * 864e5 : 0,
    },
    stats: { ...stats, ms: performance.now() - started, workers: Math.min(workers, Math.max(1, members - stats.onCard)) },
  };
}

/**
 * Predict for new rows `x` (`rows * features` numbers from the worker's `features` request).
 * Classifier: `{ proba, classes }` (`rows * classes`). Regressor: `{ values }`, four per row: mean, median, 5%, 95%.
 * Members' answers are added in member order, which makes the result identical to the one-call ensemble.
 */
export async function predictModel(engine, store, { model, weightsBlob, x, rows, workers = 1, card = false, onCardTrouble = () => {}, onProgress = () => {} }) {
  const parts = new Array(model.members);
  const stats = { onCard: 0, cardLayers: 0, cardTrouble: null, cardProblem: null };
  let done = 0, last = null;
  const one = async (rec, i, onCard) => {
    const blob = await store.get(memberKey(model.id, i));
    if (!blob) throw new EngineError("part of this fitted model is missing from the browser's storage; fit it again");
    const r = await engine.call(rec, "predict", { kind: model.kind, blob, x, rows, card: onCard, key: memberKey(model.id, i) });
    parts[i] = r.out;
    if (onCard) { stats.onCard++; stats.cardLayers += r.cardLayers; stats.cardProblem = stats.cardProblem || r.problem; }
    last = rec;
    onProgress({ done: ++done, members: model.members, onCard });
  };
  let rest = Array.from({ length: model.members }, (_, i) => i);
  if (card) {
    try {
      const rec = await engine.readyCard(model.kind, weightsBlob);
      while (rest.length) {
        await one(rec, rest[0], true);
        rest = rest.slice(1);
      }
    } catch (e) {
      if (!e.card) throw e;
      stats.cardTrouble = e.message;
      onCardTrouble(e);
    }
  }
  if (rest.length) {
    const pool = await engine.ready(Math.min(workers, rest.length), model.kind, weightsBlob);
    await shareOut(pool, rest, (rec, i) => one(rec, i, false));
  }
  const total = new Float64Array(parts[0].length);
  for (const p of parts) for (let j = 0; j < total.length; j++) total[j] += p[j];
  if (model.kind === "regressor") {
    const inv = 1 / model.members;
    for (let j = 0; j < total.length; j++) total[j] *= inv;
    return { values: total, stats };
  }
  const { proba } = await engine.call(last && !last.dead ? last : await engine.first(), "combine", { sums: total, classes: model.classes, members: model.members });
  return { proba, classes: model.classes, stats };
}

// =====================================================================================================
// Results as CSV
// =====================================================================================================

const csvCell = (s) => (/[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s);

/** The same columns the command line prints. */
export function resultsCsv(model, result, rows) {
  const lines = [];
  if (model.kind === "regressor") {
    lines.push("prediction,median,q05,q95");
    for (let r = 0; r < rows; r++) lines.push(Array.from(result.values.subarray(4 * r, 4 * r + 4), (v) => v.toFixed(6)).join(","));
  } else {
    const c = model.classes;
    lines.push(["prediction", ...model.labels.map((l) => csvCell(`p(${l})`))].join(","));
    for (let r = 0; r < rows; r++) {
      const p = result.proba.subarray(r * c, (r + 1) * c);
      let best = 0;
      for (let j = 1; j < c; j++) if (p[j] > p[best]) best = j;
      lines.push([csvCell(model.labels[best]), ...Array.from(p, (v) => v.toFixed(6))].join(","));
    }
  }
  return lines.join("\n") + "\n";
}

// =====================================================================================================
// The page
// =====================================================================================================

if (typeof document !== "undefined") startPage();

function startPage() {
  const $ = (id) => document.getElementById(id);
  const el = (tag, props = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (k === "class") n.className = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else if (v === true) n.setAttribute(k, "");
      else if (v !== false && v != null) n.setAttribute(k, v);
    }
    for (const kid of kids.flat()) if (kid != null) n.append(kid);
    return n;
  };
  const say = (node, text, tone = "") => { node.textContent = text; node.dataset.tone = tone; node.hidden = !text; };
  const local = {
    get(k, d) { try { const v = localStorage.getItem("browsertabicl." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem("browsertabicl." + k, JSON.stringify(v)); } catch { /* storage refused */ } },
  };

  const app = $("app");
  const weightsBase = app.dataset.weights || "weights/";
  const framed = (() => { try { return window.self !== window.top; } catch { return true; } })();
  const cores = navigator.hardwareConcurrency || 4;
  const deviceMemoryGB = navigator.deviceMemory || 4;

  const state = {
    store: null, catalogue: { weights: {}, models: [] }, engine: null, rate: 0, adjust: local.get("adjust", 1), adjustCard: local.get("adjustCard", 1),
    card: null, cardWhy: "no graphics card has been looked for yet",
    csv: null, csvName: "", info: null, weightBlobs: {}, downloading: {}, busy: false, selected: null, lastResult: null, membersTouched: false,
  };

  // ---- engine -------------------------------------------------------------------------------------------

  // Three ways to start the engine, tried in order: a worker from the file next to the page; a worker from a
  // one-line blob that loads that file (for hosts that only allow those); and, last, no worker at all: the engine
  // runs inside the page itself, one job at a time, and the page cannot respond while it works.
  const WAYS = ["file", "blob", "page"];
  let way = 0;
  function makeWorker() {
    const url = new URL("worker.js", import.meta.url);
    if (WAYS[way] === "file") return new Worker(url, { type: "module" });
    if (WAYS[way] === "blob") {
      const blob = new Blob([`import ${JSON.stringify(url.href)};`], { type: "text/javascript" });
      return new Worker(URL.createObjectURL(blob), { type: "module" });
    }
    const face = {
      terminate() {},
      postMessage(message) {
        import(url.href).then((mod) => mod.handle(message)).then(
          ({ transfer, ...reply }) => face.onmessage && face.onmessage({ data: reply }),
          (e) => face.onerror && face.onerror({ message: e.message || String(e) }),
        );
      },
    };
    return face;
  }
  const inPage = () => WAYS[way] === "page";

  function engineFailed(e) {
    $("engine-error").hidden = false;
    $("engine-error-detail").textContent = e.message || String(e);
    say($("hw-engine"), "The prediction engine did not start.", "bad");
    for (const b of document.querySelectorAll("[data-needs-engine]")) b.disabled = true;
  }

  async function startEngine() {
    if (typeof Worker === "undefined" || typeof WebAssembly === "undefined") throw new EngineError("this browser has no WebAssembly or no background workers");
    if (state.engine) state.engine.stop(); // a fresh engine: the old one's workers (the card's too) go
    state.engine = new Engine(makeWorker, { bootTimeout: 30000 });
    let first;
    const tried = [];
    for (;;) {
      try {
        first = await state.engine.spawn(400);
        break;
      } catch (e) {
        tried.push(e.message);
        if (way + 1 >= WAYS.length) { way = 0; throw new EngineError(tried.join("; then: ")); }
        way++; // remembered: later workers start the way that worked
      }
    }
    state.rate = first.hello.rate;
    const speed = `measured speed ${(state.rate / 1e9).toFixed(1)} billion operations a second per thread`;
    if (inPage()) say($("hw-engine"), `Engine running on the processor, inside the page: this host does not allow background workers, so one processor thread is used and the page will not respond while it is fitting or predicting (${speed}).`, "warn");
    else say($("hw-engine"), `Engine running on the processor: ${cores} processor thread${cores === 1 ? "" : "s"} available, ${speed}.`, "ok");
  }

  // The graphics card: a worker of its own holding the browser's WebGPU device. state.card is null until it
  // is ready (or if there is none); state.cardWhy says why not, in words for the page.
  const cardName = (info) => info.adapter || info.name.replace(/\s*\(BrowserWebGpu\)\s*$/i, "").trim() || "details hidden by the browser";

  async function startCard() {
    const out = $("hw-gpu");
    state.card = null;
    try {
      if (!navigator.gpu) {
        state.cardWhy = "this browser does not offer a graphics card to web pages (no WebGPU)";
        return say(out, "Graphics card: this browser does not offer one to web pages (no WebGPU). Nothing is lost: everything runs on the processor.");
      }
      if (inPage()) {
        state.cardWhy = "this host does not allow background workers, which the graphics card needs here";
        return say(out, "Graphics card: not used, because this host does not allow background workers. Everything runs on the processor.");
      }
      say(out, "Graphics card: getting it ready.");
      // (?card-work-mb=N in the address cuts the card's working memory to N MB once it is measured, to try what
      // happens when it runs short)
      // (and ?card-fail makes the card's first job fail as if the card had been lost)
      const query = new URLSearchParams(location.search);
      const limitMB = Number(query.get("card-work-mb")) || 0;
      const rec = await state.engine.openCard({ workMB: 768, modelsMB: 1024, calibrateMs: 300, limitMB, failNext: query.has("card-fail") });
      const info = rec.info;
      state.card = { name: cardName(info), software: info.software, rate: info.rate };
      const speed = `measured speed ${(info.rate / 1e9).toFixed(info.rate < 1e10 ? 1 : 0)} billion operations a second`;
      if (info.software) say(out, `Graphics card: the browser offers only a software stand-in (${state.card.name}, ${speed}). It is slower than the processor, so the processor is used unless you choose otherwise under "More options".`, "warn");
      else say(out, `Graphics card: ${state.card.name}, ready (${speed}). It is used whenever it should be quicker; the estimate below says which is used.`, "ok");
    } catch (e) {
      state.cardWhy = `the graphics card could not be started (${e.message || e})`;
      say(out, `Graphics card: found, but it could not be started here (${e.message || e}). Everything runs on the processor.`, "warn");
    }
    if (state.info) refreshEstimate();
  }

  /** The card failed during a job: say so, stop using it, and carry on with the processor. */
  function cardTrouble(e) {
    const what = String(e.message || e).replace(/^the graphics card failed \((.*)\)$/, "$1");
    state.card = null;
    state.cardWhy = `the graphics card stopped working (${what})`;
    state.engine.stopCard();
    say($("hw-gpu"), `Graphics card: it stopped working here (${what}). The processor is used from now on; reloading the page tries the card again.`, "warn");
  }

  const cardOpts = () => ({ cardRate: state.card.rate, adjust: state.adjustCard, started: !!(state.engine.card && state.engine.card.loaded) });

  // ---- storage ------------------------------------------------------------------------------------------

  async function showStorage() {
    const kinds = { files: "this browser's private file storage", database: "this browser's database storage", memory: "memory only: this browser offers this page no storage, so everything is lost when the tab closes" };
    let text = `Kept in ${kinds[state.store.kind]}.`;
    try {
      if (navigator.storage && navigator.storage.estimate) {
        const { usage, quota } = await navigator.storage.estimate();
        if (usage != null) text += ` Using ${formatBytes(usage)}${quota ? ` of about ${formatBytes(quota)} the browser allows this site` : ""}.`;
        state.quotaLeft = quota ? quota - usage : null;
      }
    } catch { /* not available */ }
    if (state.persisted === true) text += " The browser has agreed not to clear it on its own.";
    else if (state.persisted === false) text += " The browser may clear it if the device runs short of space.";
    say($("storage-line"), text, state.store.kind === "memory" ? "warn" : "");
  }

  async function askPersist() {
    try {
      if (navigator.storage && navigator.storage.persist) state.persisted = (await navigator.storage.persisted()) || (await navigator.storage.persist());
    } catch { /* not available */ }
    showStorage();
  }

  // ---- weights ------------------------------------------------------------------------------------------

  const kindWord = { classifier: "classification", regressor: "number prediction" };

  function showWeights() {
    for (const kind of ["classifier", "regressor"]) {
      const have = state.catalogue.weights[kind] || (state.weightBlobs[kind] && { bytes: state.weightBlobs[kind].size });
      const row = $(`weights-${kind}`);
      row.querySelector(".state").textContent = have ? `in this browser (${formatBytes(have.bytes)})` : state.downloading[kind] ? "downloading" : "not downloaded yet";
      row.dataset.have = have ? "yes" : "no";
      row.querySelector(".get").hidden = !!have || !!state.downloading[kind];
      row.querySelector(".drop").hidden = !have;
    }
  }

  /** The weights for `kind` as a blob: from memory, from storage, or downloaded (once) with progress. */
  function getWeights(kind) {
    if (state.weightBlobs[kind]) return Promise.resolve(state.weightBlobs[kind]);
    if (!state.downloading[kind]) {
      state.downloading[kind] = (async () => {
        const source = weightsSource(weightsBase, kind, document.baseURI);
        const key = `weights-${kind}`;
        const saved = state.catalogue.weights[kind];
        if (saved && saved.id === source.id) {
          const blob = await state.store.get(key);
          if (blob && blob.size === saved.bytes) return (state.weightBlobs[kind] = blob);
        }
        const bar = $(`weights-${kind}`).querySelector("progress"), note = $(`weights-${kind}`).querySelector(".note");
        bar.hidden = false;
        showWeights();
        let { blob, convert } = await fetchWeights(source, {
          onProgress(done, total) {
            if (total) { bar.max = total; bar.value = done; } else bar.removeAttribute("value");
            note.textContent = `${formatBytes(done)}${total ? ` of ${formatBytes(total)}` : ""}`;
          },
        });
        if (convert) {
          note.textContent = "Preparing the model";
          bar.removeAttribute("value");
          const { bytes } = await state.engine.call(await state.engine.first(), "convert", { blob });
          blob = new Blob([bytes]);
        }
        try {
          await state.store.put(key, blob);
          // read it back from storage so workers share the stored file instead of a copy held by the page
          blob = (await state.store.get(key)) || blob;
          state.catalogue.weights[kind] = { id: source.id, bytes: blob.size, saved: Date.now() };
          await writeCatalogue(state.store, state.catalogue);
          note.textContent = "";
        } catch (e) {
          note.textContent = `Downloaded, but the browser would not store it (${e.message || e}); it will be downloaded again next time.`;
        }
        bar.hidden = true;
        askPersist();
        return (state.weightBlobs[kind] = blob);
      })().catch((e) => {
        const row = $(`weights-${kind}`);
        row.querySelector("progress").hidden = true;
        row.querySelector(".note").textContent = `Download failed: ${e.message || e}`;
        throw new Error(`The ${kindWord[kind]} model could not be downloaded: ${e.message || e}`);
      }).finally(() => { delete state.downloading[kind]; showWeights(); });
    }
    return state.downloading[kind];
  }

  async function dropWeights(kind) {
    delete state.weightBlobs[kind];
    delete state.catalogue.weights[kind];
    await state.store.del(`weights-${kind}`).catch(() => {});
    await writeCatalogue(state.store, state.catalogue).catch(() => {});
    state.engine && state.engine.trim(1);
    showWeights();
    showStorage();
  }

  // ---- the training table -------------------------------------------------------------------------------

  function sampleCsv() {
    // An invented table, clearly labelled as an example in the page: seedlings grown under different conditions.
    let s = 20260212;
    const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
    const soils = ["clay", "loam", "sand"];
    const lines = ["light_hours,water_ml,soil,temperature_c,outcome"];
    for (let i = 0; i < 150; i++) {
      const light = 4 + rnd() * 10, water = 20 + rnd() * 180, soil = soils[Math.floor(rnd() * 3)], temp = 12 + rnd() * 18;
      const score = (light - 4) / 10 + (1 - Math.abs(water - 110) / 110) + (soil === "loam" ? 0.5 : soil === "sand" ? 0.1 : 0.25) + (1 - Math.abs(temp - 22) / 12) + (rnd() - 0.5) * 0.6;
      const outcome = score > 2.25 ? "thrived" : score > 1.6 ? "survived" : "died";
      lines.push(`${light.toFixed(1)},${rnd() < 0.04 ? "" : water.toFixed(0)},${soil},${temp.toFixed(1)},${outcome}`);
    }
    return lines.join("\n") + "\n";
  }

  async function useCsv(text, name, isExample = false) {
    say($("table-error"), "");
    try {
      if (!text.trim()) throw new Error("that file is empty");
      const info = await state.engine.call(await state.engine.first(), "describe", { text });
      if (info.columns.length < 2) throw new Error("the table needs at least two columns: one to predict and one to predict it from");
      if (info.rows < 2) throw new Error("the table needs a header row and at least two rows of data");
      Object.assign(state, { csv: text, csvName: name, info, membersTouched: false });
      $("table-name").textContent = `${name}${isExample ? " (an invented example, not your data)" : ""}: ${info.rows.toLocaleString()} rows, ${info.columns.length} columns`;
      const body = $("columns-body");
      body.replaceChildren(...info.columns.map((c) => el("tr", {},
        el("td", {}, c.name || "(no name)"),
        el("td", {}, c.numeric ? "numbers" : "text"),
        el("td", { class: "num" }, String(c.distinct)),
        el("td", { class: "num" }, c.missing ? String(c.missing) : ""),
        el("td", { class: "examples" }, c.examples.join(", ")),
      )));
      const target = $("target");
      target.replaceChildren(...info.columns.map((c, i) => el("option", { value: String(i) }, c.name || `(column ${i + 1})`)));
      target.value = String(info.columns.length - 1);
      $("model-name").value = name.replace(/\.[^.]+$/, "");
      $("table-details").hidden = false;
      targetChanged();
    } catch (e) {
      $("table-details").hidden = true;
      say($("table-error"), `That table could not be read: ${e.message || e}.`, "bad");
    }
  }

  const targetColumn = () => state.info.columns[Number($("target").value)];

  function targetChanged() {
    const c = targetColumn();
    const numberOk = c.numeric && c.missing === 0;
    $("kind-regressor").disabled = !numberOk;
    // few distinct values, or text: classes. Many distinct numbers: a number to predict.
    const kind = numberOk && c.distinct > 10 ? "regressor" : "classifier";
    $(`kind-${kind}`).checked = true;
    state.membersTouched = false;
    refreshEstimate();
  }

  const currentKind = () => ($("kind-regressor").checked ? "regressor" : "classifier");

  function currentPlan() {
    const c = targetColumn();
    const kind = currentKind();
    const table = { rows: state.info.rows, features: usefulColumns(state.info, Number($("target").value)), classes: kind === "classifier" ? c.distinct : 0, kv: $("precision").value };
    const env = { cores, deviceMemoryGB, rate: state.rate, adjust: state.adjust };
    const choice = $("device").value;
    // decided first for the default number of members, then again for the number chosen
    const decide = (members) => {
      const probe = estimate({ ...table, members, workers: 1, ...env });
      const workers = inPage() ? 1 : planWorkers({ members, cores, deviceMemoryGB, perWorkerBytes: probe.perWorkerBytes });
      const est = estimate({ ...table, members, workers, ...env });
      const estCard = state.card ? estimateCard({ ...table, members, rate: state.rate, ...cardOpts() }) : null;
      const device = chooseDevice({ choice, card: state.card, cardWhy: state.cardWhy, kind, classes: table.classes, cardSeconds: estCard ? estCard.seconds : Infinity, cpuSeconds: est.seconds });
      return { workers, est, estCard, ...device };
    };
    if (!state.membersTouched) {
      const onCard = decide(8).useCard && state.card;
      $("members").value = String(defaultMembers(table, inPage() ? { ...env, cores: 1 } : { ...env, card: onCard ? cardOpts() : null }));
    }
    const members = Number($("members").value);
    return { kind, table, members, ...decide(members), target: c };
  }

  function refreshEstimate() {
    if (!state.info) return;
    const { kind, table, members, workers, est, estCard, useCard, why, target } = currentPlan();
    const lines = [];
    const problems = [];
    const s = members === 1 ? "" : "s";
    if (useCard) lines.push(`Fitting ${members} member${s} on the graphics card (${state.card.name}), one after another, should take ${formatDuration(estCard.seconds)} and use about ${formatBytes(WEIGHTS_MEMORY + est.memberBytes)} of memory and up to 0.8 GB of graphics memory. The graphics card is used because ${why}; on the processor it would take ${formatDuration(est.seconds)}.`);
    else lines.push(`Fitting ${members} member${s} on ${workers} processor thread${workers === 1 ? "" : "s"} should take ${formatDuration(est.seconds)} and use about ${formatBytes(est.memoryBytes)} of memory while it runs. The processor is used because ${why}${estCard ? `; the graphics card would take ${formatDuration(estCard.seconds)}` : ""}.`);
    lines.push(`The fitted model will take about ${formatBytes(est.storageBytes)} of browser storage.`);
    if (!state.catalogue.weights[kind] && !state.weightBlobs[kind]) lines.push(`First the ${kindWord[kind]} model is downloaded once (about ${kind === "classifier" ? 55 : 57} MB${/huggingface/.test(weightsBase) ? " after conversion; 110 MB to download" : ""}).`);
    if (kind === "classifier" && target.distinct > 10) lines.push(`With ${target.distinct} classes the model answers in ${est.levels} steps, which multiplies time and storage by about ${est.levels}.`);
    if (table.rows > SOFT_LIMITS.rows || table.features > SOFT_LIMITS.columns) {
      const over = [table.rows > SOFT_LIMITS.rows ? `${table.rows.toLocaleString()} rows (comfortable up to ${SOFT_LIMITS.rows})` : null, table.features > SOFT_LIMITS.columns ? `${table.features} columns to learn from (comfortable up to ${SOFT_LIMITS.columns})` : null].filter(Boolean).join(" and ");
      problems.push(`This is a large table for a browser: ${over}. You can continue; expect the time and memory shown above, and keep this tab open and in front while it works.`);
    }
    if (kind === "classifier" && target.distinct < 2) problems.push("The column to predict has only one value, so there is nothing to learn.");
    if (kind === "classifier" && target.distinct > table.rows / 2 && target.distinct > 10) problems.push(`The column to predict has ${target.distinct} different values in ${table.rows} rows. If it is a number, choose "A number" below; if it is an identifier, pick another column.`);
    if (target.missing > 0) problems.push(`The column to predict has ${target.missing} empty cell${target.missing === 1 ? "" : "s"}; remove those rows first.`);
    if (est.memoryBytes > deviceMemoryGB * 1e9 * 0.75 && !est.refuse) problems.push(`That is a lot of memory for this device${navigator.deviceMemory ? ` (the browser reports about ${deviceMemoryGB} GB)` : ""}. Fewer members or fewer rows will need less.`);
    if (state.quotaLeft != null && est.storageBytes > state.quotaLeft) problems.push(`The browser has only about ${formatBytes(state.quotaLeft)} of storage left for this site, less than the fitted model needs. Delete a stored model first, or choose fewer members.`);
    if (est.refuse) problems.unshift(est.refuse);
    $("estimate").replaceChildren(...lines.map((t) => el("p", {}, t)));
    $("warnings").replaceChildren(...problems.map((t) => el("p", {}, t)));
    $("warnings").hidden = problems.length === 0;
    $("warnings").dataset.tone = est.refuse ? "bad" : "warn";
    $("fit").disabled = state.busy || !!est.refuse || target.missing > 0 || (kind === "classifier" && target.distinct < 2) || !state.engine;
  }

  function setBusy(busy) {
    state.busy = busy;
    for (const b of document.querySelectorAll("[data-busy-off]")) b.disabled = busy;
    $("fit").disabled = busy;
    $("predict").disabled = busy || !state.selected;
    if (!busy && state.info) refreshEstimate();
  }

  async function fit() {
    const plan = currentPlan();
    if (plan.est.refuse) return;
    setBusy(true);
    const status = $("fit-status"), bar = $("fit-bar");
    say(status, "Getting the model ready.");
    bar.hidden = false;
    bar.removeAttribute("value");
    $("cancel").hidden = false;
    state.cancelled = false;
    let ticker;
    try {
      const weightsBlob = await getWeights(plan.kind);
      if (state.cancelled) throw new Error("stopped");
      say(status, inPage() ? `Fitting now. The page will not respond until it is done (${formatDuration(plan.est.seconds)}).` : "Reading the table.");
      if (inPage()) await new Promise((r) => setTimeout(r, 50)); // let that message appear first
      const table = await state.engine.call(await state.engine.first(), "table", { text: state.csv, target: plan.target.name, regression: plan.kind === "regressor", estimators: plan.members });
      const workers = Math.min(plan.workers, table.members);
      const useCard = plan.useCard && !!state.card;
      const sized = { ...plan.table, rows: table.rows, features: table.features, classes: table.classes, members: table.members };
      const cpuEst = estimate({ ...sized, workers, rate: state.rate, adjust: state.adjust });
      const cardEst = useCard ? estimateCard({ ...sized, rate: state.rate, ...cardOpts() }) : null;
      const est = cardEst || cpuEst;
      const where = () => (useCard && state.card ? "on the graphics card" : `on ${workers} processor thread${workers === 1 ? "" : "s"}`);
      const began = performance.now();
      let progress = { done: 0, members: table.members };
      const paint = () => {
        const secs = (performance.now() - began) / 1000;
        // members finished are exact; between them the bar moves by the clock against the estimate
        bar.max = table.members;
        bar.value = Math.max(progress.done, Math.min(table.members - 0.3, (table.members * secs) / Math.max(est.seconds, 0.001)));
        say(status, `Fitting: ${progress.done} of ${table.members} member${table.members === 1 ? "" : "s"} done, ${Math.round(secs)} s so far (estimate ${formatDuration(est.seconds)}), ${where()}.`);
      };
      ticker = setInterval(paint, 250);
      const { entry, stats } = await fitModel(state.engine, state.store, {
        kind: plan.kind, weightsBlob, table, kv: plan.table.kv, workers, card: useCard, onCardTrouble: cardTrouble, name: $("model-name").value.trim() || state.csvName || "model", target: plan.target.name,
        expiresDays: Number($("expiry").value), onProgress: (p) => { progress = p; },
      });
      clearInterval(ticker);
      state.catalogue.models.unshift(entry);
      await writeCatalogue(state.store, state.catalogue);
      // remember how the estimate did here, so the next one is closer (kept between 1/4 and 4); the card and
      // the processor separately
      const mean = (v) => v.reduce((a, b) => a + b, 0) / v.length / 1000;
      const learn = (key, measured, estimated) => {
        if (estimated > 0 && measured > 0) {
          state[key] = Math.min(4, Math.max(0.25, Math.sqrt(state[key] * ((state[key] * measured) / estimated))));
          local.set(key, state[key]);
        }
      };
      if (stats.memberMs.length) learn("adjust", mean(stats.memberMs) / crowding(stats.workers), cpuEst.memberSeconds);
      if (stats.cardMemberMs.length && cardEst) learn("adjustCard", mean(stats.cardMemberMs), cardEst.memberSeconds);
      bar.max = 1; bar.value = 1;
      const how = stats.onCard === table.members ? " on the graphics card" : stats.onCard ? ` (${stats.onCard} member${stats.onCard === 1 ? "" : "s"} on the graphics card, the rest on the processor because ${state.cardWhy})` : useCard ? ` on the processor, because ${state.cardWhy}` : " on the processor";
      say(status, `Done in ${Math.max(1, Math.round(stats.ms / 1000))} s${how}. "${entry.name}" is stored in this browser (${formatBytes(entry.bytes)}); each worker used up to ${formatBytes(stats.mem)} of memory.${table.members < plan.members ? ` This table has only ${table.members} distinct arrangement${table.members === 1 ? "" : "s"} of its columns, so ${table.members} member${table.members === 1 ? " was" : "s were"} fitted.` : ""}`, "ok");
      selectModel(entry.id);
      askPersist();
    } catch (e) {
      clearInterval(ticker);
      bar.hidden = true;
      if (state.cancelled) say(status, "Stopped. Nothing was stored.", "warn");
      else say(status, `Fitting failed: ${e.message || e}.${e.fatal ? " Try fewer members, fewer rows or fewer columns." : ""}`, "bad");
      if (!state.engine.workers.length && !state.cancelled) startEngine().then(startCard, engineFailed);
    } finally {
      $("cancel").hidden = true;
      scheduleTrim();
      setBusy(false);
      showModels();
    }
  }

  async function cancel() {
    state.cancelled = true;
    state.engine.stop(); // ends every running fit at once (the card's too); a fresh engine starts below
    try {
      await startEngine();
      startCard();
    } catch (e) {
      engineFailed(e);
    }
  }

  let trimTimer;
  function scheduleTrim() {
    clearTimeout(trimTimer);
    // extra workers each hold a copy of the weights; let them go after a quiet minute
    trimTimer = setTimeout(() => { if (!state.busy && state.engine) state.engine.trim(1); }, 60000);
  }

  // ---- stored models ------------------------------------------------------------------------------------

  const day = (t) => new Date(t).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });

  function showModels() {
    const list = $("models");
    const models = state.catalogue.models;
    $("no-models").hidden = models.length > 0;
    list.replaceChildren(...models.map((m) => {
      const daysLeft = m.expires ? Math.max(0, Math.ceil((m.expires - Date.now()) / 864e5)) : 0;
      const expiry = el("select", { id: `expiry-${m.id}`, "aria-label": `When to delete ${m.name}`, "data-busy-off": true, onchange: async (ev) => {
        const d = Number(ev.target.value);
        if (d < 0) return;
        m.expires = d > 0 ? Date.now() + d * 864e5 : 0;
        await writeCatalogue(state.store, state.catalogue);
        showModels();
      } },
        el("option", { value: "-1", selected: true }, m.expires ? `deleted in ${daysLeft} day${daysLeft === 1 ? "" : "s"} (${day(m.expires)})` : "kept until you delete it"),
        el("option", { value: "7" }, "delete 7 days from now"),
        el("option", { value: "30" }, "delete 30 days from now"),
        el("option", { value: "90" }, "delete 90 days from now"),
        el("option", { value: "0" }, "keep until I delete it"),
      );
      const confirmRow = el("span", { class: "confirm", hidden: true },
        "Delete for good? ",
        el("button", { type: "button", class: "danger", onclick: async () => {
          await deleteModel(state.store, state.catalogue, m.id);
          if (state.selected === m.id) selectModel(null);
          showModels();
          showStorage();
        } }, "Yes, delete"),
        " ",
        el("button", { type: "button", onclick: () => { confirmRow.hidden = true; } }, "Keep"),
      );
      return el("li", { class: state.selected === m.id ? "model chosen" : "model" },
        el("div", { class: "model-head" },
          el("label", { class: "pick" },
            el("input", { type: "radio", name: "model", id: `pick-${m.id}`, value: m.id, checked: state.selected === m.id, onchange: () => selectModel(m.id) }),
            el("span", { class: "model-name" }, m.name),
          ),
          el("span", { class: "model-size" }, formatBytes(m.bytes)),
        ),
        el("p", { class: "model-facts" }, `Predicts "${m.target}" (${m.kind === "regressor" ? "a number" : `${m.classes} classes`}) from ${m.features} column${m.features === 1 ? "" : "s"}; learned from ${m.rows.toLocaleString()} rows with ${m.members} member${m.members === 1 ? "" : "s"}; made ${day(m.created)}.`),
        el("div", { class: "model-actions" }, expiry, el("button", { type: "button", "data-busy-off": true, onclick: () => { confirmRow.hidden = false; } }, "Delete"), confirmRow),
      );
    }));
    if (state.busy) for (const b of list.querySelectorAll("[data-busy-off]")) b.disabled = true;
    showPredictTarget();
  }

  function selectModel(id) {
    state.selected = id;
    state.lastResult = null;
    say($("predict-status"), "");
    $("results").hidden = true;
    showModels();
  }

  const chosenModel = () => state.catalogue.models.find((m) => m.id === state.selected) || null;

  function showPredictTarget() {
    const m = chosenModel();
    $("predict-with").textContent = m ? `Using "${m.name}".` : "Choose a fitted model above, or fit one first.";
    const cols = m ? m.schema.split("\n").filter((l) => /^(num|cat)\t/.test(l)).map((l) => l.split("\t")[1]) : [];
    $("predict-columns").textContent = m ? `The rows need a header line and these columns, in any order: ${cols.join(", ")}. Extra columns are ignored.` : "";
    $("new-rows").placeholder = m ? `${cols.join(",")}\n(one row per line)` : "Paste rows here, header line first";
    $("predict").disabled = state.busy || !m;
  }

  // ---- predicting ---------------------------------------------------------------------------------------

  async function predict() {
    const m = chosenModel();
    const text = $("new-rows").value;
    const status = $("predict-status");
    if (!m) return;
    if (!text.trim()) return say(status, "Paste some rows, or choose a file, first.", "warn");
    setBusy(true);
    $("results").hidden = true;
    try {
      say(status, "Getting the model ready.");
      const weightsBlob = await getWeights(m.kind);
      const { x } = await state.engine.call(await state.engine.first(), "features", { schema: m.schema, text });
      const rows = m.features ? x.length / m.features : 0;
      if (!rows) throw new Error("there are no rows under the header line");
      const workers = inPage() ? 1 : planWorkers({ members: m.members, cores, deviceMemoryGB, perWorkerBytes: 176e6 + 2 * m.bytes / m.members });
      const began = performance.now();
      say(status, `Predicting ${rows.toLocaleString()} row${rows === 1 ? "" : "s"}.`);
      // the card if there is a real one (or the visitor chose it): prediction is where it gains most
      const useCard = cardForPrediction({ choice: $("device").value, card: state.card, trainRows: m.rows, features: m.features, rows });
      const where = useCard ? "on the graphics card" : "on the processor";
      const result = await predictModel(state.engine, state.store, { model: m, weightsBlob, x, rows, workers, card: useCard, onCardTrouble: cardTrouble, onProgress: (p) => say(status, `Predicting ${rows.toLocaleString()} row${rows === 1 ? "" : "s"} ${where}: ${p.done} of ${p.members} members have answered.`) });
      state.lastResult = { model: m, result, rows, csv: resultsCsv(m, result, rows) };
      showResults();
      const st = result.stats;
      const how = !useCard ? " on the processor"
        : st.onCard < m.members ? ` (${st.onCard} member${st.onCard === 1 ? "" : "s"} on the graphics card, the rest on the processor because ${state.cardWhy})`
        : st.cardLayers === 0 ? ` on the processor: the graphics card could not take this model${st.cardProblem ? ` (${st.cardProblem})` : " (it is larger than the graphics memory set aside for it)"}`
        : st.cardProblem ? ` on the graphics card, with part of it on the processor (${st.cardProblem})` : " on the graphics card";
      const whyNot = !useCard && state.card ? (state.card.software && $("device").value !== "card" ? " (the graphics card here is a software stand-in)" : $("device").value === "processor" ? " (as chosen)" : " (for a model this small the processor answers sooner)") : !useCard ? ` (${state.cardWhy})` : "";
      say(status, `Predicted ${rows.toLocaleString()} row${rows === 1 ? "" : "s"} in ${((performance.now() - began) / 1000).toFixed(1)} s${how}${whyNot}.`, "ok");
    } catch (e) {
      say(status, `Predicting failed: ${e.message || e}.`, "bad");
      if (!state.engine.workers.length) startEngine().then(startCard, engineFailed);
    } finally {
      scheduleTrim();
      setBusy(false);
    }
  }

  function showResults() {
    const { model, result, rows } = state.lastResult;
    const shown = Math.min(rows, 200);
    const fmt = (v) => (Math.abs(v) >= 1e6 || (v !== 0 && Math.abs(v) < 1e-3) ? v.toExponential(3) : Number(v.toPrecision(5)).toString());
    let head, body = [];
    if (model.kind === "regressor") {
      head = ["Row", "Prediction (mean)", "Median", "Low (5%)", "High (95%)"];
      for (let r = 0; r < shown; r++) body.push([String(r + 1), ...Array.from(result.values.subarray(4 * r, 4 * r + 4), fmt)]);
    } else {
      const c = model.classes, wide = c <= 8;
      head = ["Row", "Prediction", ...(wide ? model.labels.map((l) => `Chance of ${l}`) : ["Chance", "Runner-up", "Chance"])];
      for (let r = 0; r < shown; r++) {
        const p = Array.from(result.proba.subarray(r * c, (r + 1) * c));
        const order = p.map((_, i) => i).sort((a, b) => p[b] - p[a]);
        const pct = (v) => `${(100 * v).toFixed(1)}%`;
        body.push([String(r + 1), model.labels[order[0]], ...(wide ? p.map(pct) : [pct(p[order[0]]), model.labels[order[1]], pct(p[order[1]])])]);
      }
    }
    $("results-head").replaceChildren(el("tr", {}, head.map((t) => el("th", { scope: "col" }, t))));
    $("results-body").replaceChildren(...body.map((cells) => el("tr", {}, cells.map((t, i) => el("td", { class: i === 1 && model.kind !== "regressor" ? "" : "num" }, t)))));
    $("results-note").textContent = `${rows > shown ? `Showing the first ${shown} of ${rows.toLocaleString()} rows; copying gives all of them. ` : ""}${model.kind === "regressor" ? "Low and high mark the range the model expects 9 times out of 10." : ""}`;
    $("results-csv").value = state.lastResult.csv;
    $("results-csv-wrap").hidden = true;
    $("download").hidden = framed;
    $("results").hidden = false;
  }

  async function copyCsv() {
    const note = $("copy-note");
    try {
      await navigator.clipboard.writeText(state.lastResult.csv);
      say(note, "Copied. Paste it into a spreadsheet or a text file.", "ok");
    } catch {
      // the clipboard was refused: show the text selected so it can be copied by hand
      $("results-csv-wrap").hidden = false;
      $("results-csv").focus();
      $("results-csv").select();
      say(note, "The browser would not copy for the page. The text is selected below: press Ctrl+C (Cmd+C on a Mac).", "warn");
    }
  }

  function download() {
    const a = el("a", { href: URL.createObjectURL(new Blob([state.lastResult.csv], { type: "text/csv" })), download: "predictions.csv" });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  }

  // ---- wiring -------------------------------------------------------------------------------------------

  const readFile = async (file, into) => {
    if (!file) return;
    if (file.size > 300e6) return say($("table-error"), "That file is larger than 300 MB, too large to read in a browser tab.", "bad");
    into(await file.text(), file.name);
  };
  $("file").addEventListener("change", (ev) => readFile(ev.target.files[0], useCsv));
  $("sample").addEventListener("click", () => useCsv(sampleCsv(), "seedlings.csv", true));
  const drop = $("drop");
  for (const type of ["dragenter", "dragover"]) drop.addEventListener(type, (ev) => { ev.preventDefault(); drop.classList.add("over"); });
  for (const type of ["dragleave", "drop"]) drop.addEventListener(type, (ev) => { ev.preventDefault(); drop.classList.remove("over"); });
  drop.addEventListener("drop", (ev) => readFile(ev.dataTransfer.files[0], useCsv));
  $("target").addEventListener("change", targetChanged);
  for (const id of ["kind-classifier", "kind-regressor", "precision", "device"]) $(id).addEventListener("change", () => { state.membersTouched = false; refreshEstimate(); });
  $("members").addEventListener("change", () => { state.membersTouched = true; refreshEstimate(); });
  $("fit").addEventListener("click", fit);
  $("cancel").addEventListener("click", cancel);
  $("predict").addEventListener("click", predict);
  $("copy").addEventListener("click", copyCsv);
  $("download").addEventListener("click", download);
  $("rows-file").addEventListener("change", (ev) => readFile(ev.target.files[0], (text) => { $("new-rows").value = text; }));
  const rowsDrop = $("new-rows");
  rowsDrop.addEventListener("dragover", (ev) => ev.preventDefault());
  rowsDrop.addEventListener("drop", (ev) => { if (ev.dataTransfer.files.length) { ev.preventDefault(); readFile(ev.dataTransfer.files[0], (text) => { rowsDrop.value = text; }); } });
  for (const kind of ["classifier", "regressor"]) {
    $(`weights-${kind}`).querySelector(".get").addEventListener("click", () => getWeights(kind).then(() => state.info && refreshEstimate(), () => {}));
    $(`weights-${kind}`).querySelector(".drop").addEventListener("click", () => dropWeights(kind));
  }

  (async () => {
    state.store = await openStore();
    state.catalogue = await readCatalogue(state.store);
    const removed = await sweep(state.store, state.catalogue).catch(() => []);
    if (removed.length) say($("expired-note"), `Deleted because ${removed.length === 1 ? "its" : "their"} time was up: ${removed.join(", ")}.`);
    try { state.persisted = navigator.storage && navigator.storage.persisted ? await navigator.storage.persisted() : undefined; } catch { /* not available */ }
    showWeights();
    showModels();
    showStorage();
    if (state.catalogue.models.length) selectModel(state.catalogue.models[0].id);
    try {
      await startEngine();
      for (const b of document.querySelectorAll("[data-needs-engine]")) b.disabled = false;
      startCard();
    } catch (e) {
      engineFailed(e);
    }
  })();
}
