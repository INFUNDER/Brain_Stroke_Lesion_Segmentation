#!/usr/bin/env python3
"""Read-only: raw mask value types for blank vs intact labels; MONAI empty-mask behaviour; old-run lesion-positive test Dice."""
import json, collections, sys, os
import numpy as np, nibabel as nib, torch
from monai.metrics import DiceMetric

ROOT = "/home/ronit.28010/Brain_Stroke"
RAW = f"{ROOT}/isless2022dataset/ISLES-2022_notformatted"
S = sys.argv[1]
rows = json.load(open(f"{S}/geometry.json"))

def sig(c):
    n = nib.load(f"{RAW}/derivatives/{c}/ses-0001/{c}_ses-0001_msk.nii.gz")
    a = np.asanyarray(n.dataobj)
    u = np.unique(a)
    return str(n.get_data_dtype()), tuple(sorted(float(x) for x in u[:4])), (None if n.dataobj.slope is None else float(n.dataobj.slope))

groups = dict(blank=[r for r in rows if r["raw_fg"] > 0 and r["fmt_fg"] == 0], ok=[r for r in rows if r["raw_fg"] > 0 and r["fmt_fg"] > 0])
for g, rs in groups.items():
    print(g, collections.Counter(sig(r["case"]) for r in rs).most_common(8))

# blank rate per acquisition geometry
geo = collections.defaultdict(lambda: [0, 0])
for r in rows:
    if r["raw_fg"] == 0:
        continue
    key = (tuple(r["dwi"][0]), tuple(r["dwi"][1]))
    geo[key][0] += 1
    geo[key][1] += int(r["fmt_fg"] == 0)
print("geometry -> (n lesion cases, n blanked):")
for k, v in sorted(geo.items(), key=lambda kv: -kv[1][0])[:8]:
    print("  ", k, v)
aniso = [r for r in rows if r["raw_fg"] > 0 and r["dwi"][1][2] >= 4.0]
iso = [r for r in rows if r["raw_fg"] > 0 and r["dwi"][1][2] < 4.0]
print("thick-slice (>=4mm): n", len(aniso), "blanked", sum(r["fmt_fg"] == 0 for r in aniso))
print("thin-slice (<4mm): n", len(iso), "blanked", sum(r["fmt_fg"] == 0 for r in iso))

# MONAI behaviour
p_empty = torch.zeros(1, 2, 8, 8, 8); p_empty[:, 0] = 1
y_empty = p_empty.clone()
p_pos = torch.zeros(1, 2, 8, 8, 8); p_pos[:, 0] = 1; p_pos[:, 1, 2:4, 2:4, 2:4] = 1; p_pos[:, 0, 2:4, 2:4, 2:4] = 0
m = DiceMetric(include_background=False, reduction="mean")
m(p_empty, y_empty); print("MONAI single empty case, aggregate() =", m.aggregate().item(), " raw buffer =", m.get_buffer().flatten().tolist()); m.reset()
m(p_pos, p_pos); m(p_empty, y_empty); print("MONAI dataset of [perfect lesion case, empty case], aggregate() =", m.aggregate().item(), "(0.5 would mean empties counted as 0; 1.0 means excluded)"); m.reset()
m2 = DiceMetric(include_background=False, reduction="mean", ignore_empty=False)
m2(p_empty, y_empty); print("MONAI ignore_empty=False on correct empty =", m2.aggregate().item()); m2.reset()

# March run per-case
d = json.load(open(f"{ROOT}/mamba_results/results/final_test_FP32_20260323_152217.json"))
names = [os.path.basename(c["case"]).split(".nii")[0] for c in d["per_case"]]
blanks = {r["case"] for r in rows if r["fmt_fg"] == 0}
vals = [c["dice"] for c, n in zip(d["per_case"], names) if n not in blanks]
print(f"March run on the test set, excluding blank-label cases the same way CV did: mean {np.mean(vals):.4f} n={len(vals)} (paper reports 0.6418 with them counted as 0)")
