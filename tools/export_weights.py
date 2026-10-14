"""Turn a published TabICL checkpoint into a tabicl-rs weight bundle (same numbers, our layout)."""
import sys, json, numpy as np, torch
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
import bundle
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
t = {k: v.numpy() for k, v in ck["state_dict"].items()}
t["config.json"] = np.frombuffer(json.dumps(ck["config"]).encode(), dtype=np.uint8)
bundle.write(sys.argv[2], t)
print(sys.argv[2], len(t), "tensors")
