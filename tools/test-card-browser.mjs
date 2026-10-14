// The graphics card in a real Chromium against the processor workers, using the page's own engine code
// (site/app.js and site/worker.js): fit and predict the same tables both ways, compare the answers, report times.
//   NODE_PATH=SOMEWHERE/node_modules node tools/test-card-browser.mjs [--headless] [--browser /usr/bin/chromium]
//       [--tables 150x4,500x100] [--members 4] [--ui]
// Without --headless the browser opens a (small, off-screen) window in the desktop session: headless Chromium
// offers only a software stand-in for a graphics card ("swiftshader"), a window gets the real one.
// --ui also drives the page itself on its example table: fit and predict on the card, then on the processor.
import http from "node:http";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const puppeteer = require("puppeteer-core");
const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "site");
const arg = (name, d) => { const i = process.argv.indexOf(name); return i > 0 ? process.argv[i + 1] : d; };
const headless = process.argv.includes("--headless");
const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".wasm": "application/wasm", ".json": "application/json", ".bin": "application/octet-stream" };
const server = http.createServer((req, res) => {
  const url = new URL(req.url, "http://x");
  const file = path.join(root, url.pathname === "/" ? "index.html" : decodeURIComponent(url.pathname));
  if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); return res.end("not found"); }
  res.writeHead(200, { "content-type": types[path.extname(file)] || "application/octet-stream" });
  fs.createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${server.address().port}`;
const profile = fs.mkdtempSync(path.join(process.env.SCRATCH || os.tmpdir(), "browsertabicl-card-"));
const args = ["--no-first-run", "--enable-unsafe-webgpu", "--enable-features=Vulkan"];
if (process.argv.includes("--no-webgpu")) args.splice(1, 1, "--disable-webgpu", "--disable-features=WebGPU");
if (!headless) args.push("--ozone-platform-hint=auto", "--window-position=4000,4000", "--window-size=900,700");
const browser = await puppeteer.launch({ executablePath: arg("--browser", "/usr/bin/chromium"), headless, userDataDir: profile, args, protocolTimeout: 600000 });
let failures = 0;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failures++; };

try {
  const page = await browser.newPage();
  page.on("pageerror", (e) => console.log("page error:", String(e)));
  page.on("console", (m) => { if (m.type() === "error") console.log("console:", m.text()); });
  await page.goto(base + "/index.html");
  const tables = arg("--tables", "150x4,500x100") === "none" ? [] : arg("--tables", "150x4,500x100").split(",").map((t) => t.split("x").map(Number));
  const members = Number(arg("--members", "4"));
  const report = !tables.length ? { runs: [] } : await page.evaluate(async (tables, members) => {
    const app = await import("./app.js");
    const eng = new app.Engine(() => new Worker(new URL("worker.js", location.href), { type: "module" }));
    const first = await eng.spawn(400);
    const card = await eng.openCard({ workMB: 768, modelsMB: 1024, calibrateMs: 300 });
    const out = { cpuRate: first.hello.rate, card: card.info, runs: [] };
    const weights = {};
    for (const kind of ["classifier", "regressor"]) weights[kind] = (await app.fetchWeights(app.weightsSource("weights/", kind, location.href))).blob;
    const store = app.memoryStore();
    let seed = 7;
    const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
    for (const [rows, features] of tables) {
      for (const kind of ["classifier", "regressor"]) {
        // a table with some signal in it: the target depends on the first few columns
        const head = [...Array.from({ length: features }, (_, j) => `c${j}`), "y"].join(",");
        const lines = [head];
        for (let r = 0; r < rows + 40; r++) {
          const v = Array.from({ length: features }, () => rnd() * 4 - 2);
          const score = v.slice(0, 3).reduce((a, b) => a + b, 0) + (rnd() - 0.5);
          lines.push([...v.map((x) => x.toFixed(3)), kind === "classifier" ? ["low", "mid", "high"][score < -0.7 ? 0 : score < 0.7 ? 1 : 2] : (10 * score).toFixed(3)].join(","));
        }
        const train = lines.slice(0, rows + 1).join("\n"), test = [head, ...lines.slice(rows + 1)].join("\n");
        const table = await eng.call(first, "table", { text: train, target: "y", regression: kind === "regressor", estimators: members });
        const { x } = await eng.call(first, "features", { schema: table.schema, text: test });
        const nt = x.length / table.features;
        const run = { kind, rows, features, members: table.members };
        const fitted = {};
        for (const where of ["processor", "card"]) {
          const t = performance.now();
          const { entry, stats } = await app.fitModel(eng, store, { kind, weightsBlob: weights[kind], table, workers: Math.min(members, Math.max(1, (navigator.hardwareConcurrency || 4) - 1)), card: where === "card", name: "t", target: "y" });
          fitted[where] = entry;
          run[`fit_${where}_s`] = (performance.now() - t) / 1000;
          run[`fit_${where}_member_ms`] = (where === "card" ? stats.cardMemberMs : stats.memberMs).map(Math.round);
          if (where === "card") { run.fit_on_card = stats.onCard; run.card_trouble = stats.cardTrouble; }
        }
        const answers = {};
        for (const [model, where] of [["processor", "processor"], ["processor", "card"], ["card", "processor"]]) {
          // twice: the first prediction on the card also puts the member on it
          for (const pass of [1, 2]) {
            const t = performance.now();
            const r = await app.predictModel(eng, store, { model: fitted[model], weightsBlob: weights[kind], x, rows: nt, workers: Math.min(members, 7), card: where === "card" });
            run[`predict_${model}fit_on_${where}_${pass}_s`] = (performance.now() - t) / 1000;
            answers[`${model}-${where}`] = Array.from(r.proba || r.values);
            if (where === "card") { run.predict_card_layers = r.stats.cardLayers; run.predict_card_problem = r.stats.cardProblem; run.predict_card_trouble = r.stats.cardTrouble; }
          }
        }
        const worst = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0);
        run.diff_predicted_on_card = worst(answers["processor-processor"], answers["processor-card"]);
        run.diff_fitted_on_card = worst(answers["processor-processor"], answers["card-processor"]);
        run.scale = kind === "classifier" ? 1 : Math.max(...answers["processor-processor"].map(Math.abs));
        out.runs.push(run);
      }
    }
    eng.stop();
    return out;
  }, tables, members);
  if (tables.length) console.log(`processor: ${(report.cpuRate / 1e9).toFixed(1)} billion multiply-adds a second per thread; graphics card: ${report.card.name}${report.card.software ? " (software stand-in)" : ""}, ${(report.card.rate / 1e9).toFixed(0)} billion a second, opened in ${Math.round(report.card.openMs)} ms`);
  for (const r of report.runs) {
    console.log(JSON.stringify(r));
    const tol = r.kind === "classifier" ? 1e-3 : 1e-3 * r.scale;
    check(r.fit_on_card === r.members && !r.card_trouble, `${r.kind} ${r.rows} x ${r.features}: all ${r.members} members fitted on the card`);
    check(r.predict_card_layers > 0 && !r.predict_card_trouble, `${r.kind} ${r.rows} x ${r.features}: prediction ran on the card (${r.predict_card_layers} layers)`);
    check(r.diff_predicted_on_card < tol, `${r.kind} ${r.rows} x ${r.features}: predicted on the card = on the processor (worst difference ${r.diff_predicted_on_card.toExponential(2)})`);
    check(r.diff_fitted_on_card < tol, `${r.kind} ${r.rows} x ${r.features}: fitted on the card = fitted on the processor (worst difference ${r.diff_fitted_on_card.toExponential(2)})`);
  }

  if (process.argv.includes("--ui")) {
    // the page itself, on its example table
    const ui = await browser.newPage();
    ui.on("pageerror", (e) => console.log("page error:", String(e)));
    await ui.goto(base + "/index.html" + (arg("--query", "") ? "?" + arg("--query", "") : ""));
    await ui.waitForFunction(() => /Engine running/.test(document.getElementById("hw-engine").textContent), { timeout: 60000 });
    await ui.waitForFunction(() => !/getting it ready|Looking/.test(document.getElementById("hw-gpu").textContent), { timeout: 60000 });
    console.log("     " + (await ui.$eval("#hw-gpu", (n) => n.textContent)));
    await ui.click("#sample");
    await ui.waitForFunction(() => !document.getElementById("table-details").hidden);
    const results = {};
    for (const device of ["card", "processor"]) {
      await ui.select("#device", device);
      console.log(`     [${device}] ` + (await ui.$eval("#estimate", (n) => n.textContent)));
      await ui.click("#fit");
      await ui.waitForFunction(() => /^(Done|Fitting failed)/.test(document.getElementById("fit-status").textContent), { timeout: 300000 });
      const fitText = await ui.$eval("#fit-status", (n) => n.textContent);
      console.log(`     [${device}] ` + fitText);
      const expect = arg("--expect-card", "yes") === "yes" || device === "processor";
      check(device === "card" && expect ? /on the graphics card/.test(fitText) : /on the processor/.test(fitText), `page: fitted ${device === "card" && expect ? "on the graphics card" : "on the processor"}, and says so`);
      const rows = "light_hours,water_ml,soil,temperature_c\n8,100,loam,22\n5,30,sand,14\n12,150,clay,27\n";
      await ui.$eval("#new-rows", (n, v) => { n.value = v; }, rows);
      for (const pass of [1, 2]) {
        await ui.$eval("#predict-status", (n) => { n.textContent = ""; });
        await ui.click("#predict");
        await ui.waitForFunction(() => /^(Predicted|Predicting failed)/.test(document.getElementById("predict-status").textContent), { timeout: 120000 });
        console.log(`     [${device}] prediction ${pass}: ` + (await ui.$eval("#predict-status", (n) => n.textContent)));
      }
      results[device] = await ui.$eval("#results-csv", (n) => n.value);
    }
    check(results.card && results.processor && results.card.split("\n")[1].split(",")[0] === results.processor.split("\n")[1].split(",")[0], "page: the card-fitted and processor-fitted models give the same answer on the first row");
    await ui.close();
  }
} finally {
  await browser.close();
  server.close();
  fs.rmSync(profile, { recursive: true, force: true });
}
console.log(failures ? `${failures} check(s) failed` : "all checks passed");
process.exit(failures ? 1 : 0);
