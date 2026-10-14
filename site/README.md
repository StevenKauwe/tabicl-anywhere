# The BrowserTabICL page

A static page that fits a TabICL v2 model on a visitor's CSV table and predicts with it, entirely in their
browser. No server, no framework, no build step for the page itself.

## What is in this folder

| file | size | what it is |
|---|---|---|
| `page.html` | 19 KB | the page, without `<html>`/`<head>`/`<body>` (hosts such as Claude artifacts add their own) |
| `index.html` | 20 KB | `page.html` wrapped as a complete document. Generated: edit `page.html`, then run `tools/build-site.sh` |
| `app.js` | 69 KB | the page's logic (one module) |
| `worker.js` | 10 KB | one engine on a background thread; the page starts several, and one more for the graphics card |
| `pkg/browsertabicl_wasm.js` | 75 KB | glue for the engine (generated) |
| `pkg/browsertabicl_wasm_bg.wasm` | 2.36 MB (0.55 MB compressed in transit) | the engine, graphics-card code included (generated) |
| `weights/classifier-v2-half.json` + `.00.bin` ... `.03.bin` | 55.3 MB in 4 parts | the classification model, 16-bit, each part at most 14 MB |
| `weights/regressor-v2-half.json` + `.00.bin` ... `.03.bin` | 57.3 MB in 4 parts | the number-prediction model, likewise |
| `weights/LICENSE-TabICL.txt` | 2 KB | the licence the weights are redistributed under (also shown in the page footer) |
| `.nojekyll` | 0 | tells GitHub Pages to serve the folder as it is |

A visitor downloads about 2.5 MB to see the page, then 55.3 MB the first time they fit a category model, and
57.3 MB more only if they ever predict a number. Both are kept in their browser afterwards.

## Rebuilding

```
tools/build-site.sh            # engine -> pkg/, weights -> weights/ (from fixtures/*.tirs), index.html
tools/build-site.sh path/to/tabicl-classifier-v2-20260212.ckpt path/to/tabicl-regressor-v2-20260212.ckpt
```

Needs `rustup target add wasm32-unknown-unknown` and `wasm-bindgen` (the command-line tool, same version as in
`Cargo.lock`) on the PATH. The weights step is just:

```
browsertabicl convert IN.ckpt-or-.tirs OUT.tirs --half
browsertabicl split OUT.tirs site/weights/classifier-v2-half      # parts of at most 14 MB, plus the .json list
```

## Where the weights come from: one setting

In `page.html`, on the `<main id="app" data-weights="...">` line:

* `weights/` (default): the half-size copy in this folder. Works everywhere, including hosts that only let a page
  fetch its own files.
* `https://huggingface.co/jingang/TabICL/resolve/main/`: the authors' published checkpoints (110 MB each, twice
  the download). The page converts them to half size in the browser before storing them. Hugging Face allows
  this from any site. Then `weights/*.bin` need not be published at all. Not possible in a Claude artifact.
* any other folder URL holding the same `*.json` and `*.bin` files (it must allow cross-site requests).

## Publishing on GitHub Pages

`site/` is where the page is built and tried out; what is published is a copy of exactly the deployable files
in `docs/`, which GitHub Pages serves from the main branch.

1. `tools/deploy-docs.sh` rebuilds the page (it runs `tools/build-site.sh`) and copies the files into `docs/`
   (`tools/deploy-docs.sh --no-build` only copies what is already in `site/`). The largest file is 14 MB;
   GitHub's limit is 100 MB per file.
2. Commit `docs/`. Repository Settings -> Pages -> Source: "Deploy from a branch", main, `/docs`.
3. Open `https://USER.github.io/REPO/`.

Nothing needs special response headers. Pages serves `.wasm` with the right type.

## Publishing as a Claude artifact

Publish `page.html` as the page, and these files alongside it at the same relative paths:

```
app.js
worker.js
pkg/browsertabicl_wasm.js
pkg/browsertabicl_wasm_bg.wasm
weights/classifier-v2-half.json
weights/classifier-v2-half.00.bin   weights/classifier-v2-half.01.bin
weights/classifier-v2-half.02.bin   weights/classifier-v2-half.03.bin
weights/regressor-v2-half.json
weights/regressor-v2-half.00.bin    weights/regressor-v2-half.01.bin
weights/regressor-v2-half.02.bin    weights/regressor-v2-half.03.bin
```

Do not publish `index.html` there. Every file is under the 15 MB limit; the `.bin` parts should be given the
type `application/octet-stream`. Leave `data-weights="weights/"`. One publish may carry at most 64 MB, so send
the two models' parts in two goes (the regressor's parts can be left out entirely if only categories are needed:
the page then reports a failed download when someone chooses to predict a number).

## The graphics card

Where the browser offers WebGPU, the page also starts a worker that holds the graphics card, and fits and
predicts there with the same shaders as the native build (`src/gpu.rs`). Ensemble members go one after another
on the one card; the processor workers are untouched and take over whenever the card cannot.

* At start the page asks for the card, measures it (about a third of a second of the card's attention) and
  says what it found: a real card ("Graphics card: nvidia ampere, ready"), only a software stand-in (headless
  Chromium's "swiftshader": correct but slower than the processor, so not used unless chosen), or none.
* Before fitting, the estimate is worked out for both and says which will be used and why ("because it should
  be quicker than the processor"; "with more than 10 classes ..."; "you chose ..."), with the other one's time
  beside it. The card's estimate is `estimateCard` in `app.js`, fitted to the measurements below; like the
  processor's, it corrects itself from the visitor's own fits. "More options" has "Where to work": whichever
  should be quicker (the default), the graphics card, or the processor.
* Prediction uses the card for models of about 10,000 training cells or more, or 1,000 new rows or more (below
  that the processor answers sooner: the card costs about a tenth of a second per member to start). A fitted
  member stays on the card (up to 1 GB of graphics memory) for the next prediction.
* If the card fails during a job (out of memory, the device lost), what is left of the job goes to the
  processor workers, the page says so beside the result and in the hardware line, and the card is not used
  again until the page is reloaded. If the card's working memory is too small for part of a job, that part is
  done on the processor inside the card worker, and a fit the card could do no part of counts as a failure.
* Graphics memory, at most: 768 MB of working memory, plus 1 GB for fitted members kept between predictions,
  plus about 81 MB of weights.

For trying the fallbacks: `index.html?card-fail` makes the card's first job fail as a lost card would;
`index.html?card-work-mb=1` cuts the card's working memory to 1 MB once it has been measured.

Measured in Chromium 152 under the desktop session (a window, not headless) on an RTX 3090, adapter reported as
"nvidia ampere", processor i7-11700K (16.1 billion multiply-adds a second per thread in WebAssembly), card
measured at 38-41 billion:

| table | members | fit: processor (7 workers) | fit: card | predict 40 rows: processor | predict: card (first / again) |
|---|---|---|---|---|---|
| example (150 x 4, classes) | 8 | 3 s | 1 s | 0.1 s (3 rows) | 0.5 / 0.3 s (3 rows) |
| 500 x 100, classes | 4 | 9.4 s | 2.6 s | 0.56 s | 0.75 / 0.33 s |
| 500 x 100, number | 4 | 7.6 s | 2.4 s | 0.57 s | 0.78 / 0.33 s |
| 1000 x 20, classes | 4 | 6.9 s | 1.9 s | 0.35 s | 0.74 / 0.25 s |
| 2000 x 10, classes | 4 | 13.7 s | 1.9 s | 0.49 s | 1.25 / 0.31 s |
| 50 x 3, classes | 4 | 0.34 s | 0.70 s | 0.10 s | 0.48 / 0.32 s |

Card against processor (`tools/test-card-browser.mjs`): probabilities agree to 5e-6 whether the card fitted or
predicted, regression values to 1.5e-4 on a spread of about 45. The same checks pass on the software stand-in
(headless), where a card fit of the example table takes 9 s against the estimate's 10 s.

## How the page starts the engine

It tries, in order: a background worker from `worker.js`; a worker from a one-line blob that loads `worker.js`;
and finally the engine inside the page itself (works without workers, but uses one processor thread and the page
cannot respond while it is busy; the page says so). If none starts within 30 seconds it shows a message instead of
waiting. Storage likewise: the browser's private file system, else IndexedDB, else memory for the session (the
page says which).

## Tests

```
node tools/test-wasm.cjs       # engine against the reference; one-member path = whole-ensemble path; half weights
node tools/test-site.mjs       # the page's own logic under Node: estimates, weight parts, CSV path, worker pool
node tools/test-browser.mjs    # the real page in headless Chromium (needs puppeteer-core; see the file's header)
node tools/test-card-browser.mjs --ui   # card against processor in a Chromium window (the real card), fits and
                                        # predictions compared, times printed; --headless for the software stand-in
```
