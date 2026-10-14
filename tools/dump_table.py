"""Run the reference's table-level classifier (preprocessing + ensemble + cache) and save what it did."""
import sys, numpy as np, torch
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
import bundle
from tabicl import TabICLClassifier

ckpt, out = sys.argv[1], sys.argv[2]
n, m, h, c, seed, n_est = (int(a) for a in sys.argv[3:9])
cache = len(sys.argv) < 10
rng = np.random.default_rng(seed)
X = rng.standard_normal((n + m, h))
X[:, 1] = np.exp(2 * X[:, 1])                 # heavy right tail
X[:, 2] = np.round(X[:, 2] * 2)               # few distinct values
X[:, 3] = 7.0                                 # constant: must be dropped
X[:, 4] = -np.abs(X[:, 4]) ** 3               # all negative, skewed
X[:, 5] = X[:, 5] * 1e6 + 3e7                 # large scale and offset
w = rng.standard_normal((h, c))
y = (np.nan_to_num(X / (np.abs(X).max(0) + 1e-9)) @ w + 0.3 * rng.standard_normal((n + m, c))).argmax(-1)
if c > 10:  # many classes: equal-count bins of one score, so every class really occurs
    score = np.nan_to_num(X[:, 0]) + 0.5 * np.nan_to_num(X[:, 2]) + 0.1 * rng.standard_normal(n + m)
    y = np.searchsorted(np.quantile(score, np.linspace(0, 1, c + 1)[1:-1]), score)
X[rng.random(X.shape) < 0.03] = np.nan        # missing cells
X[:, 3] = 7.0
torch.set_num_threads(8)
clf = TabICLClassifier(n_estimators=n_est, kv_cache=cache, model_path=ckpt, allow_auto_download=False, device="cpu", random_state=42)
clf.fit(X[:n], y[:n])
proba = clf.predict_proba(X[n:])
eg = clf.ensemble_generator_
t = {"x_train": X[:n], "y_train": y[:n].astype(np.int32), "x_test": X[n:], "y_test": y[n:].astype(np.int32), "proba": proba.astype(np.float64)}
i = 0
for method, cfgs in eg.ensemble_configs_.items():
    pp = eg.preprocessors_[method]
    t[f"train_{method}"] = pp.X_transformed_.astype(np.float64)
    t[f"test_{method}"] = pp.transform(eg.unique_filter_.transform(clf.X_encoder_.transform(X[n:]))).astype(np.float64)
    if method == "power":
        t["lambdas"] = pp.normalizer_.lambdas_.astype(np.float64)
    for fs, cs in cfgs:
        t[f"member{i}_method"] = np.frombuffer(method.encode(), dtype=np.uint8)
        t[f"member{i}_features"] = np.array(fs, dtype=np.int32)
        t[f"member{i}_classes"] = np.array(cs, dtype=np.int32)
        i += 1
bundle.write(out, t)
print(out, "members", i, "methods", list(eg.ensemble_configs_), "kept features", int(eg.unique_filter_.features_to_keep_.sum()), "accuracy", float((proba.argmax(1) == y[n:]).mean()))
