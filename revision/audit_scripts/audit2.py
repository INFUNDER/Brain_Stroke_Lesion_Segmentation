#!/usr/bin/env python3
"""Read-only audit: true parameter counts from checkpoints and model definitions; CV logs."""
import os, re, glob, torch, json

ROOT = "/home/ronit.28010/Brain_Stroke"

def count_ckpt(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    n = 0
    dtypes = set()
    for k, v in sd.items():
        if k == "n_averaged" or not hasattr(v, "numel"):
            continue
        n += v.numel()
        dtypes.add(str(v.dtype))
    return n, dtypes

for p in [f"{ROOT}/mamba_results/checkpoints/best_fold1.pth", f"{ROOT}/checkpoints/best_metric_model.pth"]:
    n, dt = count_ckpt(p)
    print(f"{os.path.basename(p)}: {n/1e6:.3f}M params in state_dict, dtypes {dt}, size {os.path.getsize(p)/1e6:.1f} MB")

from monai.networks.nets import UNETR
m = UNETR(in_channels=3, out_channels=2, img_size=(96, 96, 96), feature_size=16, hidden_size=768, mlp_dim=3072,
          num_heads=12, proj_type="perceptron", norm_name="instance", res_block=True)
print("UNETR constructed params: %.3fM" % (sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6))
