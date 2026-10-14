"""Run the reference's table-level regressor and save inputs and outputs."""
import sys, numpy as np, torch
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
import bundle
from tabicl import TabICLRegressor

ckpt, out = sys.argv[1], sys.argv[2]
n, m, h, seed, n_est = (int(a) for a in sys.argv[3:8])
rng = np.random.default_rng(seed)
X = rng.standard_normal((n + m, h))
X[:, 1] = np.exp(X[:, 1])
y = 3.0 * X[:, 0] - 2.0 * np.log(X[:, 1]) + np.sin(3 * X[:, 2]) + 0.3 * rng.standard_normal(n + m) + 50.0
X[rng.random(X.shape) < 0.02] = np.nan
torch.set_num_threads(8)
reg = TabICLRegressor(n_estimators=n_est, kv_cache=True, model_path=ckpt, allow_auto_download=False, device="cpu", random_state=42)
reg.fit(X[:n], y[:n])
alphas = [0.0005, 0.05, 0.25, 0.5, 0.9, 0.9995]
res = reg.predict(X[n:], output_type=["mean", "variance", "median", "quantiles"], alphas=alphas)
t = {"x_train": X[:n], "y_train": y[:n], "x_test": X[n:], "y_test": y[n:], "alphas": np.array(alphas)}
for k, v in res.items():
    t[k] = v.astype(np.float64)
bundle.write(out, t)
print(out, {k: v.shape for k, v in res.items()}, "rmse", float(np.sqrt(np.mean((res["mean"] - y[n:]) ** 2))))
