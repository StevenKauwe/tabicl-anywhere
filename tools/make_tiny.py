"""Make a tiny, randomly initialised TabICL v2 checkpoint (same architecture, much smaller widths).

    ref/venv/bin/python -I tools/make_tiny.py OUT.ckpt classifier|regressor [seed]

The weights mean nothing; the point is a checkpoint small enough to commit (tens of KB) that still runs every
stage of the network through the reference, so CI can compare against it without the 110 MB published weights.
Saved with torch.save, so it is a real PyTorch zip checkpoint and also exercises the Rust checkpoint reader.
"""
import sys, torch
from tabicl._model.tabicl import TabICL

out, kind = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
torch.manual_seed(seed)
config = dict(
    max_classes=10 if kind == "classifier" else 0,
    num_quantiles=99,
    embed_dim=8,
    col_num_blocks=2,
    col_nhead=2,
    col_num_inds=8,
    col_affine=False,
    col_feature_group="same",
    col_feature_group_size=3,
    col_target_aware=True,
    col_ssmax="qassmax-mlp-elementwise",
    row_num_blocks=2,
    row_nhead=2,
    row_num_cls=2,
    row_rope_base=100000,
    row_rope_interleaved=False,
    icl_num_blocks=2,
    icl_nhead=2,
    icl_ssmax="qassmax-mlp-elementwise",
    ff_factor=2,
    dropout=0.0,
    activation="gelu",
    norm_first=True,
    bias_free_ln=False,
    recompute=False,
)
model = TabICL(**config)
with torch.no_grad():
    # PyTorch's default initialisation leaves layer norms at exactly 1 and 0 and many biases tiny; move every
    # parameter off those special values so a mistake in any of them shows up in the comparison
    for name, p in model.named_parameters():
        if "rope" in name:
            continue
        p.add_(0.1 * torch.randn_like(p))
torch.save({"config": config, "state_dict": model.state_dict()}, out)
print(out, sum(p.numel() for p in model.parameters()), "parameters")
