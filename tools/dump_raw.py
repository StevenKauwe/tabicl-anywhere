"""Run the reference model directly (no table preprocessing) and save every stage's numbers."""
import sys, numpy as np, torch
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
import bundle
from tabicl._model.tabicl import TabICL
from tabicl._model.kv_cache import TabICLCache
from tabicl._model.inference_config import InferenceConfig

ckpt, out = sys.argv[1], sys.argv[2]
n, m, h, c, seed = (int(a) for a in sys.argv[3:8])
ck = torch.load(ckpt, map_location="cpu", weights_only=True)
model = TabICL(**ck["config"]); model.load_state_dict(ck["state_dict"]); model.eval()
torch.set_num_threads(8)
# what the reference's own table wrapper uses on a CPU: plain 32-bit floats throughout
cfg = InferenceConfig()
dev = torch.device("cpu")
cfg.update_from_dict({k: {"device": dev, "use_amp": False, "use_fa3": False} for k in ("COL_CONFIG", "ROW_CONFIG", "ICL_CONFIG")})
g = torch.Generator().manual_seed(seed)
X = torch.randn(1, n + m, h, generator=g)
w = torch.randn(h, c, generator=g)
y = (X[0] @ w + 0.5 * torch.randn(n + m, c, generator=g)).argmax(-1)
y_train = y[:n].float().unsqueeze(0)
t = {"x_train": X[0, :n], "x_test": X[0, n:], "y_train": y[:n].to(torch.int32), "y_test": y[n:].to(torch.int32)}
with torch.no_grad():
    t["logits_plain"] = model(X.clone(), y_train.clone(), return_logits=True, inference_config=cfg)[0]
    cache = TabICLCache(train_shape=X[:, :n].shape, num_classes=c)
    emb = model.col_embedder.forward_with_cache(X[:, :n].clone(), col_cache=cache.col_cache, y_train=y_train.clone(), use_cache=False, store_cache=True, mgr_config=cfg.COL_CONFIG)
    t["train_emb"] = emb[0].clone()
    rep = model.row_interactor(emb, mgr_config=cfg.ROW_CONFIG)
    t["train_rep"] = rep[0].clone()
    model.icl_predictor.forward_with_cache(rep, icl_cache=cache.icl_cache, y_train=y_train.clone(), use_cache=False, store_cache=True, mgr_config=cfg.ICL_CONFIG)
    for b, e in cache.col_cache.kv.items():
        t[f"col_k{b}"], t[f"col_v{b}"] = e.key[0], e.value[0]
    for l, e in cache.icl_cache.kv.items():
        t[f"icl_k{l}"], t[f"icl_v{l}"] = e.key[0], e.value[0]
    emb = model.col_embedder.forward_with_cache(X[:, n:].clone(), col_cache=cache.col_cache, use_cache=True, store_cache=False, mgr_config=cfg.COL_CONFIG)
    t["test_emb"] = emb[0].clone()
    rep = model.row_interactor(emb, mgr_config=cfg.ROW_CONFIG)
    t["test_rep"] = rep[0].clone()
    t["logits"] = model.icl_predictor.forward_with_cache(rep, icl_cache=cache.icl_cache, num_classes=c, use_cache=True, store_cache=False, mgr_config=cfg.ICL_CONFIG)[0]
bundle.write(out, {k: v.numpy() for k, v in t.items()})
print(out, {k: tuple(v.shape) for k, v in t.items() if not k[:4] in ("col_", "icl_")}, "max |cached - plain| logits", float((t["logits"] - t["logits_plain"]).abs().max()))
