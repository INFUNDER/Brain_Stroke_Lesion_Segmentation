#!/usr/bin/env python3
"""Re-score the saved OmniMamba test-set probabilities against the original (raw) ISLES masks, on the same grid. Read-only."""
import os, json, sys
import numpy as np
import nibabel as nib
from scipy import stats
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd

ROOT = "/home/ronit.28010/Brain_Stroke"
RAW = f"{ROOT}/isless2022dataset/ISLES-2022_notformatted"
RES = f"{ROOT}/mamba_results"
S = sys.argv[1]

test = []
for line in open(f"{RES}/test_set_paths.txt"):
    p = line.strip().split("  |  ")
    if len(p) == 2:
        test.append(os.path.basename(p[1]).split(".nii")[0])

tf = Compose([
    LoadImaged(keys=["label"]),
    EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
    Spacingd(keys=["label"], pixdim=(1.5, 1.5, 2.0), mode="nearest"),
    Orientationd(keys=["label"], axcodes="RAS"),
])

un = json.load(open(f"{RES}/results/unetr_test_results.json"))["results"]
un = {r["case_name"]: r for r in un}
gt_blank = {}

def dsc(p, g):
    if g.sum() == 0:
        return np.nan
    if p.sum() == 0:
        return 0.0
    return 2 * (p & g).sum() / (p.sum() + g.sum())

rows = []
for c in sorted(test):
    prob = np.load(f"{RES}/non_leaked_probs/{c}_prob.npy")
    old = np.load(f"{RES}/non_leaked_probs/{c}_gt.npy") > 0
    d = tf({"label": f"{RAW}/derivatives/{c}/ses-0001/{c}_ses-0001_msk.nii.gz"})
    true = np.asarray(d["label"][0]) > 0
    if true.shape != prob.shape:
        print("SHAPE MISMATCH", c, true.shape, prob.shape)
        continue
    p = prob >= 0.5
    rows.append(dict(case=c, true_fg=int(true.sum()), old_fg=int(old.sum()), pred_fg=int(p.sum()), dice_true=dsc(p, true), dice_old=dsc(p, old),
                     same=bool((true == old).all())))
print("cases with blank old label but real lesion:")
for r in rows:
    if r["old_fg"] == 0 and r["true_fg"] > 0:
        print(f"  {r['case']}: true lesion {r['true_fg']} vox ({r['true_fg']*0.0045:.2f} mL); model predicted {r['pred_fg']} vox; Dice vs TRUE mask = {r['dice_true']:.4f}")
print("truly empty:", [(r["case"], r["pred_fg"]) for r in rows if r["true_fg"] == 0])
print("cases where old != true on the rest:", [r["case"] for r in rows if r["old_fg"] > 0 and not r["same"]])
v = np.array([r["dice_true"] for r in rows if r["true_fg"] > 0])
print(f"OmniMamba 5-fold ensemble, P=0.5, against TRUE masks: mean Dice {np.mean(v):.4f} (n={len(v)} lesion-positive, median {np.median(v):.4f})")
un_vals = np.array([un[r["case"]]["corrected_dice"] for r in rows if r["old_fg"] > 0 and r["true_fg"] > 0])
print(f"For reference old blank-label reported lesion-positive subset mean UNETR {un_vals.mean():.4f}")
json.dump(rows, open(f"{S}/rescored_true_labels.json", "w"), indent=0, default=float)
