#!/usr/bin/env python3
"""Drive the page in a real WebKit (WebKitGTK, the engine of Safari and of every browser on iPhone and iPad).

    python3 tools/test-webkit.py [--desktop]

Needs WebKitGTK 4.1 with its Python bindings (Arch: webkit2gtk-4.1, python-gobject) and a display; the page
is drawn off screen. Build first (tools/build-site.sh). It serves site/ on a local port, opens the page with an
iPhone's user agent (so the page plans for a phone: --desktop leaves the agent alone), and checks that WebKit
refuses relaxed SIMD and gets the plain-SIMD engine, then fits the example table and predicts two rows.
WebKitGTK is not iOS: it is the same JavaScript and WebAssembly engine, not the same memory limits.
"""
import functools
import http.server
import json
import os
import sys
import threading
import time

# drawn off screen and in software: no GL context is needed (or available, under some Wayland setups)
os.environ.setdefault("WEBKIT_DISABLE_COMPOSITING_MODE", "1")
os.environ.setdefault("WEBKIT_DISABLE_DMABUF_RENDERER", "1")
os.environ.setdefault("GDK_BACKEND", "x11" if os.environ.get("DISPLAY") else "wayland")

import gi  # noqa: E402

gi.require_version("Gtk", "3.0")
gi.require_version("WebKit2", "4.1")
from gi.repository import GLib, Gtk, WebKit2  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "site")
IPHONE = "Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/141.0.7390.96 Mobile/15E148 Safari/604.1"
requests = []


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm", ".js": "text/javascript", ".bin": "application/octet-stream"}

    def log_message(self, *args):
        requests.append(self.path.split("?")[0])


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=ROOT))
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{server.server_address[1]}"

failures = 0


def check(ok, what):
    global failures
    print(("ok   " if ok else "FAIL ") + what, flush=True)
    failures += 0 if ok else 1


view = WebKit2.WebView.new_with_context(WebKit2.WebContext.new_ephemeral())
settings = view.get_settings()
settings.set_enable_write_console_messages_to_stdout(True)
settings.set_hardware_acceleration_policy(WebKit2.HardwareAccelerationPolicy.NEVER)
if "--desktop" not in sys.argv:
    settings.set_user_agent(IPHONE)
window = Gtk.OffscreenWindow()
window.set_default_size(390, 844)
window.add(view)
window.show_all()


def pump(seconds):
    end = time.time() + seconds
    ctx = GLib.MainContext.default()
    while time.time() < end:
        ctx.iteration(False)
        time.sleep(0.005)


def js(script, timeout=30):
    """Evaluate the expression `script` in the page (it may use await) and return its value, through JSON."""
    box = {}
    body = f"return JSON.stringify(await (async () => {{ return {script}; }})());"

    def done(v, res):
        try:
            box["v"] = v.call_async_javascript_function_finish(res).to_string()
        except Exception as e:  # noqa: BLE001
            box["e"] = str(e)

    view.call_async_javascript_function(body, -1, None, None, None, None, done)
    end = time.time() + timeout
    while "v" not in box and "e" not in box and time.time() < end:
        pump(0.05)
    if "e" in box:
        raise RuntimeError(box["e"])
    if "v" not in box:
        raise TimeoutError(script)
    return json.loads(box["v"]) if box["v"] not in (None, "undefined") else None


def wait_for(expr, timeout):
    end = time.time() + timeout
    while time.time() < end:
        if js(expr):
            return True
        pump(0.25)
    return False


loaded = {}
view.connect("load-changed", lambda _v, ev: ev == WebKit2.LoadEvent.FINISHED and loaded.update(done=True))
view.load_uri(base + "/index.html")
end = time.time() + 60
while not loaded.get("done") and time.time() < end:
    pump(0.05)
print(f"     WebKitGTK {WebKit2.get_major_version()}.{WebKit2.get_minor_version()}.{WebKit2.get_micro_version()}, user agent: {js('navigator.userAgent', 60)}")

# what this WebKit's WebAssembly accepts: the page's own probes, and each engine file as a whole
probes = js("""await (async () => {
  const w = await import("./worker.js");
  const out = { usable: w.usableBuilds(), files: {} };
  for (const [name, file] of Object.entries(w.BUILDS)) {
    const bytes = new Uint8Array(await (await fetch("pkg/" + file)).arrayBuffer());
    out.files[name] = WebAssembly.validate(bytes);
  }
  return out;
})()""", 120)
print(f"     probes: {probes}")
check(probes["usable"][0] == "simd" and "relaxed" not in probes["usable"], "WebKit refuses the relaxed-SIMD probe and accepts plain SIMD, so it is given the simd build")
check(probes["files"] == {"relaxed": False, "simd": True, "scalar": True}, "WebKit refuses the relaxed engine file and accepts the simd and scalar files as wholes")

requests.clear()
view.reload()
started = wait_for('/Engine running|did not start/.test(document.getElementById("hw-engine").textContent)', 90)
line = js('document.getElementById("hw-engine").textContent')
print(f"     {line}")
err = js('document.getElementById("engine-error").hidden ? "" : document.getElementById("engine-error").textContent.trim()')
if err:
    print(f"     {err}")
check(started and js('document.getElementById("hw-engine").dataset.build') == "simd", "the engine starts, with the simd build")
wasm = sorted({r for r in requests if r.endswith(".wasm")})
check(wasm == ["/pkg/tabicl_anywhere_wasm_bg.wasm"], f"only that engine file is downloaded ({', '.join(wasm)})")

js('document.getElementById("sample").click()')
wait_for('!document.getElementById("table-details").hidden', 30)
est = js('document.getElementById("estimate").textContent')
print("     " + est.split(". ")[0])
if "--desktop" not in sys.argv:
    check("on 2 processor threads" in est or "on 1 processor thread" in est, "with a phone's user agent the example table is planned on at most 2 workers")
js('document.getElementById("fit").click()')
done = wait_for('/^(Done|Fitting failed|Stopped)/.test(document.getElementById("fit-status").textContent)', 600)
status = js('document.getElementById("fit-status").textContent')
print(f"     {status}")
check(done and status.startswith("Done"), "fitting the example table works")
js('(() => { document.getElementById("new-rows").value = "light_hours,water_ml,soil,temperature_c\\n12,110,loam,22\\n4.5,20,clay,13\\n"; document.getElementById("predict").click(); })()')
wait_for('/^(Predicted|Predicting failed)/.test(document.getElementById("predict-status").textContent)', 300)
print("     " + js('document.getElementById("predict-status").textContent'))
rows = js('[...document.querySelectorAll("#results-body tr")].map((tr) => [...tr.children].map((td) => td.textContent))')
print(f"     {rows}")
check(rows is not None and len(rows) == 2 and rows[0][1] == "thrived" and rows[1][1] == "died", "predicting works: the well-kept seedling thrives, the neglected one dies")

server.shutdown()
print(f"\n{failures} check(s) FAILED" if failures else "\nall WebKit checks passed")
sys.exit(1 if failures else 0)
