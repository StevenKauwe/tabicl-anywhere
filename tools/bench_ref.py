"""Time the reference on a table saved by `tabicl synth`: fit with its key-value cache, then predict."""
import sys, time, struct, resource, json, numpy as np, torch
from tabicl import TabICLClassifier

def read(path):
    f = open(path, "rb"); f.read(8); n = struct.unpack("<Q", f.read(8))[0]
    out = {}
    for line in f.read(n).decode().splitlines():
        p = line.split(); f.seek(int(p[2]))
        out[p[0]] = np.frombuffer(f.read(int(p[3])), dtype=p[1]).reshape([int(v) for v in p[4:]])
    return out

ckpt, data, threads = sys.argv[1], read(sys.argv[2]), int(sys.argv[3])
device = sys.argv[4] if len(sys.argv) > 4 else "cpu"
torch.set_num_threads(threads)
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
res = {"threads": threads, "device": device, "import_rss_mb": rss()}
clf = TabICLClassifier(n_estimators=1, kv_cache=True, model_path=ckpt, allow_auto_download=False, device=device)
t = time.time(); clf.fit(data["x_train"].astype(np.float64), data["y_train"]); res["fit_s"] = time.time() - t; res["fit_rss_mb"] = rss()
xt = data["x_test"].astype(np.float64)
xt = np.concatenate([xt] * 32)[:2048]
for rows in (1, 64, 256, 1024, 2048):
    times = []
    for _ in range(5 if rows == 1 else 3):
        t = time.time(); p = clf.predict_proba(xt[:rows]); times.append(time.time() - t)
    res[f"predict_{rows}_ms"] = min(times) * 1e3
if device != "cpu":
    res["vram_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
res["peak_rss_mb"] = rss()
print(json.dumps(res))
