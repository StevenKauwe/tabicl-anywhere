// Runs site/worker.js inside a Node worker thread, standing in for a browser's Web Worker (tools/test-site.mjs).
import { parentPort, workerData } from "node:worker_threads";
import { readFileSync } from "node:fs";
globalThis.__BROWSERTABICL_DEBUG = !!process.env.DEBUG_ENGINE;
globalThis.__BROWSERTABICL_WASM = readFileSync(workerData.wasm);
// in a Web Worker `self` is the global object itself; the engine looks things up on it (the clock, for one)
globalThis.self = globalThis;
globalThis.postMessage = (message, transfer) => parentPort.postMessage(message, transfer);
await import(workerData.script);
parentPort.on("message", (data) => globalThis.self.onmessage({ data }));
