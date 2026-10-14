# The BrowserTabICL page

A static page that fits a TabICL v2 model on a visitor's CSV table and predicts with it, entirely in their
browser. No server, no framework, no build step for the page itself.

## What is in this folder

| file | size | what it is |
|---|---|---|
| `page.html` | 19 KB | the page, without `<html>`/`<head>`/`<body>` (hosts such as Claude artifacts add their own) |
| `index.html` | 19 KB | `page.html` wrapped as a complete document. Generated: edit `page.html`, then run `tools/build-site.sh` |
| `app.js` | 56 KB | the page's logic (one module) |
| `worker.js` | 6 KB | one engine on a background thread; the page starts several |
| `pkg/browsertabicl_wasm.js` | 26 KB | glue for the engine (generated) |
| `pkg/browsertabicl_wasm_bg.wasm` | 1.56 MB (0.4 MB compressed in transit) | the engine (generated) |
| `weights/classifier-v2-half.json` + `.00.bin` ... `.03.bin` | 55.3 MB in 4 parts | the classification model, 16-bit, each part at most 14 MB |
| `weights/regressor-v2-half.json` + `.00.bin` ... `.03.bin` | 57.3 MB in 4 parts | the number-prediction model, likewise |
| `weights/LICENSE-TabICL.txt` | 2 KB | the licence the weights are redistributed under (also shown in the page footer) |
| `.nojekyll` | 0 | tells GitHub Pages to serve the folder as it is |

A visitor downloads about 1.7 MB to see the page, then 55.3 MB the first time they fit a category model, and
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

1. Commit this whole `site/` folder (the largest file is 14 MB; GitHub's limit is 100 MB per file). Note that the
   repository's `.gitignore` must not exclude `site/weights/*.bin` or `site/pkg/`.
2. Repository Settings -> Pages -> Source: "GitHub Actions", with the stock "Static HTML" workflow and its
   upload `path:` set to `site`. (Or "Deploy from a branch" if the contents of `site/` are placed in `/docs` or
   at the root of a branch.)
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
```
