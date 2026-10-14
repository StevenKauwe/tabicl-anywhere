// Drive the page in a real (headless) Chromium: start, example table, fit, predict, reload, expiry controls.
//   npm install --prefix SOMEWHERE puppeteer-core
//   NODE_PATH=SOMEWHERE/node_modules node tools/test-browser.mjs [--shots DIR] [--browser /usr/bin/chromium]
// Not part of the default tests: it needs a Chromium-family browser and puppeteer-core (neither is a dependency).
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
const shots = arg("--shots", null);
const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".wasm": "application/wasm", ".json": "application/json", ".bin": "application/octet-stream", ".txt": "text/plain" };
const requests = [];
const server = http.createServer((req, res) => {
  const url = new URL(req.url, "http://x");
  requests.push(url.pathname);
  if (url.pathname === "/framed.html") {
    // the page inside a sandboxed frame with no origin of its own: a stand-in for a restrictive host
    res.writeHead(200, { "content-type": types[".html"] });
    return res.end(`<!DOCTYPE html><title>frame</title><iframe sandbox="allow-scripts" src="/index.html" style="width:100%;height:90vh"></iframe>`);
  }
  const file = path.join(root, url.pathname === "/" ? "index.html" : decodeURIComponent(url.pathname));
  if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); return res.end("not found"); }
  res.writeHead(200, { "content-type": types[path.extname(file)] || "application/octet-stream", "content-length": fs.statSync(file).size, ...(url.searchParams.has("cors") || req.headers.origin === "null" ? { "access-control-allow-origin": "*" } : {}) });
  fs.createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${server.address().port}`;
const profile = fs.mkdtempSync(path.join(process.env.SCRATCH || os.tmpdir(), "tabicl-anywhere-profile-"));
const browser = await puppeteer.launch({ executablePath: arg("--browser", "/usr/bin/chromium"), headless: true, userDataDir: profile, args: ["--no-first-run", "--enable-unsafe-webgpu"] });
let failures = 0;
const check = (ok, what) => { console.log(`${ok ? "ok  " : "FAIL"} ${what}`); if (!ok) failures++; };
const text = (page, sel) => page.$eval(sel, (n) => n.textContent.trim());
const shot = async (page, name) => { if (shots) { fs.mkdirSync(shots, { recursive: true }); await page.screenshot({ path: path.join(shots, name + ".png"), fullPage: true }); } };

try {
  const page = await browser.newPage();
  const problems = [];
  page.on("console", (m) => { if (m.type() === "error") problems.push(m.text()); });
  page.on("pageerror", (e) => problems.push(String(e)));
  await page.setViewport({ width: 1100, height: 900 });
  await page.goto(base + "/index.html");
  await page.waitForFunction(() => /Engine running/.test(document.getElementById("hw-engine").textContent), { timeout: 40000 });
  console.log("     " + (await text(page, "#hw-engine")));
  await page.waitForFunction(() => !/getting it ready|Looking/.test(document.getElementById("hw-gpu").textContent), { timeout: 40000 });
  console.log("     " + (await text(page, "#hw-gpu")));
  // headless Chromium offers only a software stand-in for a card, which the page reports and passes over
  check(/Graphics card/.test(await text(page, "#hw-gpu")) && /processor/.test(await text(page, "#hw-gpu")), "the page says what graphics hardware it found and that the processor is used");
  await page.waitForFunction(() => !document.getElementById("storage-line").hidden);
  console.log("     " + (await text(page, "#storage-line")));
  await shot(page, "1-start-desktop-light");

  await page.click("#sample");
  await page.waitForFunction(() => !document.getElementById("table-details").hidden);
  check((await page.$$eval("#columns-body tr", (r) => r.length)) === 5, "example table: 5 columns listed");
  check((await page.$eval("#target", (s) => s.selectedOptions[0].textContent)) === "outcome" && (await page.$eval("#kind-classifier", (r) => r.checked)), "the last column is preselected as the one to predict, as a category");
  console.log("     " + (await text(page, "#estimate")));
  await page.select("#target", "0");
  check(await page.$eval("#kind-regressor", (r) => r.checked), "choosing a numeric column with many values switches to predicting a number");
  await page.select("#target", "2");
  check(await page.$eval("#kind-regressor", (r) => r.disabled), "a text column cannot be predicted as a number");
  await page.select("#target", "4");
  await shot(page, "2-table-desktop-light");

  const t0 = Date.now();
  await page.click("#fit");
  await page.waitForFunction(() => /^(Done|Fitting failed|Stopped)/.test(document.getElementById("fit-status").textContent), { timeout: 300000 });
  const status = await text(page, "#fit-status");
  console.log(`     ${status} [wall clock ${((Date.now() - t0) / 1000).toFixed(1)} s including the weights download from this machine]`);
  check(/^Done/.test(status), "fit completes");
  check(requests.filter((p) => /classifier-v2-half\.\d\d\.bin$/.test(p)).length === 4 && !requests.some((p) => /regressor/.test(p)), "only the classifier's four weight parts were fetched");
  check((await page.$$eval("#models li", (l) => l.length)) === 1, "the fitted model is listed");
  await page.$eval("#new-rows", (t) => { t.value = "light_hours,water_ml,soil,temperature_c\n12,110,loam,22\n4.5,20,clay,13\n9,,sand,25\n"; });
  await page.click("#predict");
  await page.waitForFunction(() => /^(Predicted|Predicting failed)/.test(document.getElementById("predict-status").textContent), { timeout: 120000 });
  console.log("     " + (await text(page, "#predict-status")));
  const rows = await page.$$eval("#results-body tr", (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent)));
  console.log("     " + JSON.stringify(rows));
  check(rows.length === 3 && rows[0][1] === "thrived" && rows[1][1] === "died", "three predictions shown; the well-kept seedling thrives, the neglected one dies");
  const csv = await page.$eval("#results-csv", (t) => t.value);
  check(/^prediction,p\(died\),p\(survived\),p\(thrived\)\n/.test(csv) && csv.trim().split("\n").length === 4, "results are available as CSV with the command line's columns");
  await browser.defaultBrowserContext().overridePermissions(base, ["clipboard-read", "clipboard-write"]);
  await page.bringToFront();
  await page.click("#copy");
  await page.waitForFunction(() => !document.getElementById("copy-note").hidden);
  const clip = await page.evaluate(() => navigator.clipboard.readText()).catch(() => null);
  const why = await page.evaluate(() => navigator.clipboard.writeText("x").then(() => "", (e) => e.message));
  const selected = await page.evaluate(() => { const t = document.getElementById("results-csv"); return !t.closest("[hidden]") && t.selectionEnd - t.selectionStart === t.value.length; });
  if (clip === csv) check(true, `"Copy as CSV" puts the CSV on the clipboard (${await text(page, "#copy-note")})`);
  else check(selected, `this browser refused the clipboard (${why}); the page then shows the CSV selected for copying by hand`);
  check(!(await page.$eval("#download", (b) => b.hidden)), "outside a frame the download button is offered as an extra");
  await shot(page, "3-results-desktop-light");

  // "Fill in example rows": fresh invented seedlings without the column to predict, their true answers shown
  const fillIn = async (n) => {
    await page.$eval("#example-count", (b, v) => { b.value = v; }, String(n));
    await page.click("#add-examples");
    return page.$eval("#new-rows", (t) => t.value.trim().split("\n"));
  };
  const predictRows = async () => {
    await page.click("#predict");
    await page.waitForFunction(() => /^(Predicted|Predicting failed)/.test(document.getElementById("predict-status").textContent), { timeout: 120000 });
    return { head: await page.$$eval("#results-head th", (t) => t.map((n) => n.textContent)), body: await page.$$eval("#results-body tr", (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent))) };
  };
  let filled = await fillIn(7);
  check(filled.length === 8 && filled[0] === "light_hours,water_ml,soil,temperature_c" && filled.slice(1).every((l) => l.split(",").length === 4), `"Fill in" 7 example rows: 7 rows under a header without "outcome" (${await text(page, "#examples-note")})`);
  let got = await predictRows();
  console.log("     " + JSON.stringify(got.body.slice(0, 3)));
  check(got.body.length === 7 && got.head.at(-1) === "Actual" && got.body.every((r) => /^(thrived|survived|died)$/.test(r.at(-1))), "predicting the 7 example rows gives 7 results, each with its true outcome beside it");
  check((await page.$eval("#results-csv", (t) => t.value)).trim().split("\n")[0] === "prediction,p(died),p(survived),p(thrived)", "the CSV of the results keeps the command line's columns (no Actual)");
  if (shots) await (await page.$("#h-predict")).evaluate((h) => h.closest("section").id = "predict-section");
  if (shots) await (await page.$("#predict-section")).screenshot({ path: path.join(shots, "3b-example-rows.png") });
  await page.$eval("#new-rows", (t) => { t.value = t.value.replace(/\n$/, "\n12,110,loam,22\n"); });
  got = await predictRows();
  check(got.body.length === 8 && got.head.at(-1) !== "Actual", "after the rows are edited, no Actual column is shown");

  // a number to predict: downloads the second model
  await page.select("#target", "0");
  await page.select("#members", "2");
  await page.click("#fit");
  await page.waitForFunction(() => /^(Done|Fitting failed)/.test(document.getElementById("fit-status").textContent) && !document.getElementById("fit").disabled, { timeout: 300000 });
  check(/^Done/.test(await text(page, "#fit-status")), "a number-predicting model fits too (second model downloaded)");
  filled = await fillIn(7);
  check(filled.length === 8 && filled[0] === "water_ml,soil,temperature_c,outcome" && filled.slice(1).every((l) => l.split(",").length === 4), "with the first column to predict, the example rows leave out light_hours instead");
  got = await predictRows();
  check(got.body.length === 7 && got.head.at(-1) === "Actual" && got.body.every((r) => r.length === 6 && Number.isFinite(+r[5])), "predicting those gives 7 number predictions with the true light hours beside them");
  await page.$eval("#new-rows", (t) => { t.value = "water_ml,soil,temperature_c,outcome\n110,loam,22,thrived\n20,clay,13,died\n"; });
  await page.click("#predict");
  await page.waitForFunction(() => /^(Predicted|Predicting failed)/.test(document.getElementById("predict-status").textContent), { timeout: 120000 });
  const reg = await page.$$eval("#results-body tr", (trs) => trs.map((tr) => [...tr.children].map((td) => td.textContent)));
  console.log("     " + JSON.stringify(reg));
  check(reg.length === 2 && reg.every((r) => r.length === 5 && +r[3] <= +r[1] && +r[1] <= +r[4]), "number predictions come with a low-high range around them");

  // come back later: everything is still there, nothing is downloaded again
  const before = requests.length;
  await page.reload();
  await page.waitForFunction(() => /Engine running/.test(document.getElementById("hw-engine").textContent), { timeout: 40000 });
  check((await page.$$eval("#models li", (l) => l.length)) === 2, "after reloading, both fitted models are still listed");
  check((await page.$$eval(".weights li[data-have=yes]", (l) => l.length)) === 2, "after reloading, both downloaded models are found in the browser");
  await page.$eval("#new-rows", (t) => { t.value = "water_ml,soil,temperature_c,outcome\n110,loam,22,thrived\n"; });
  await page.click("#predict");
  await page.waitForFunction(() => /^(Predicted|Predicting failed)/.test(document.getElementById("predict-status").textContent), { timeout: 120000 });
  check(/^Predicted/.test(await text(page, "#predict-status")) && !requests.slice(before).some((p) => /\.bin$/.test(p)), "predicting after a reload works without fetching the weights again");
  console.log("     " + (await text(page, "#storage-line")));
  // expiry and delete
  await page.select("#models li select", "7");
  await page.waitForFunction(() => /deleted in 7 days/.test(document.querySelector("#models li select").selectedOptions[0].textContent));
  check(true, "a model's expiry can be changed");
  await page.evaluate(() => [...document.querySelectorAll("#models li .model-actions > button")][0].click());
  await page.evaluate(() => document.querySelector("#models li .confirm .danger").click());
  await page.waitForFunction(() => document.querySelectorAll("#models li").length === 1);
  check(true, "a model can be deleted, with an in-page confirmation");

  // layout: no sideways scrolling at phone width, in both themes
  for (const [w, scheme] of [[390, "light"], [390, "dark"], [1100, "dark"]]) {
    await page.setViewport({ width: w, height: 800 });
    await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: scheme }]);
    await page.click("#sample");
    await new Promise((r) => setTimeout(r, 400));
    const over = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    check(over <= 0, `no sideways page scroll at ${w}px, ${scheme}`);
    await shot(page, `4-${w}-${scheme}`);
  }
  check(problems.length === 0, `no errors in the browser console${problems.length ? ": " + problems.join(" | ") : ""}`);

  // a host that isolates the page (sandboxed frame without an origin): must either work or say why not
  const framed = await browser.newPage();
  await framed.goto(base + "/framed.html");
  const frame = await (await framed.waitForSelector("iframe")).contentFrame();
  await frame.waitForFunction(() => /Engine running|did not start/.test(document.getElementById("hw-engine").textContent), { timeout: 100000 });
  const engineLine = await frame.$eval("#hw-engine", (n) => n.textContent);
  const shown = await frame.$eval("#engine-error", (n) => (n.hidden ? "" : n.textContent.trim()));
  console.log(`     sandboxed frame: ${engineLine}${shown ? " | " + shown : ""}`);
  console.log("     " + (await frame.$eval("#storage-line", (n) => n.textContent)));
  check(/Engine running/.test(engineLine) || shown.length > 0, "in a sandboxed frame the page either runs or explains that the engine could not start (no hang)");
  if (/Engine running/.test(engineLine)) {
    await frame.click("#sample");
    await frame.waitForFunction(() => !document.getElementById("table-details").hidden);
    await frame.select("#members", "1");
    await frame.click("#fit");
    await frame.waitForFunction(() => /^(Done|Fitting failed)/.test(document.getElementById("fit-status").textContent), { timeout: 300000 });
    console.log("     sandboxed frame: " + (await frame.$eval("#fit-status", (n) => n.textContent)));
    check(/^Done/.test(await frame.$eval("#fit-status", (n) => n.textContent)), "fitting works in the sandboxed frame");
    check(await frame.$eval("#download", (b) => b.hidden), "inside a frame the download button stays hidden (copying is the way out)");
  }
  await shot(framed, "5-sandboxed-frame");
} finally {
  await browser.close();
  server.close();
  fs.rmSync(profile, { recursive: true, force: true });
}
console.log(failures ? `\n${failures} check(s) FAILED` : "\nall browser checks passed");
process.exit(failures ? 1 : 0);
