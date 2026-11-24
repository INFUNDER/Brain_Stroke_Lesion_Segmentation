#!/usr/bin/env python3
"""Read-only: are the three channels in ISLES2022_Formatted/imagesTr really DWI, ADC and spatially aligned FLAIR?"""
import json, sys
import numpy as np, nibabel as nib
from scipy.ndimage import affine_transform

ROOT = "/home/ronit.28010/Brain_Stroke"
RAW = f"{ROOT}/isless2022dataset/ISLES-2022_notformatted"
S = sys.argv[1]
rows = json.load(open(f"{S}/geometry.json"))
thick = [r["case"] for r in rows if r["dwi"][1][2] >= 4.0]
thin = [r["case"] for r in rows if r["dwi"][1][2] < 4.0][::9]
cases = thick + thin
res = []
for c in cases:
    ses = "ses-0001"
    d = nib.load(f"{RAW}/{c}/{ses}/dwi/{c}_{ses}_dwi.nii.gz"); a = nib.load(f"{RAW}/{c}/{ses}/dwi/{c}_{ses}_adc.nii.gz"); fl = nib.load(f"{RAW}/{c}/{ses}/anat/{c}_{ses}_FLAIR.nii.gz")
    f = nib.load(f"{ROOT}/ISLES2022_Formatted/imagesTr/{c}.nii.gz")
    img = np.asanyarray(f.dataobj).astype(np.float32)
    dw = np.asanyarray(d.dataobj).astype(np.float32); ad = np.asanyarray(a.dataobj).astype(np.float32); fv = np.asanyarray(fl.dataobj).astype(np.float32)
    mat = np.linalg.inv(fl.affine) @ d.affine
    fl_in_dwi = affine_transform(fv, mat[:3, :3], mat[:3, 3], output_shape=dw.shape, order=1)
    brain = dw > np.percentile(dw[dw > 0], 10) if (dw > 0).any() else dw > 0
    def r(x, y):
        return float(np.corrcoef(x[brain], y[brain])[0, 1])
    res.append(dict(case=c, thick=c in thick, c0_dwi=r(img[..., 0], dw), c1_adc=r(img[..., 1], ad), c2_flair_hdr=r(img[..., 2], fl_in_dwi), c0_c2=r(img[..., 0], img[..., 2]),
                    c2_zero=bool((img[..., 2] == 0).all())))
for name, g in [("thick", [x for x in res if x["thick"]]), ("thin", [x for x in res if not x["thick"]])]:
    print(name, "n", len(g), {k: (round(float(np.min([x[k] for x in g])), 3), round(float(np.median([x[k] for x in g])), 3)) for k in ("c0_dwi", "c1_adc", "c2_flair_hdr")}, "(min, median)  all-zero FLAIR:", sum(x["c2_zero"] for x in g))
bad = sorted(res, key=lambda x: x["c2_flair_hdr"])[:5]
print("lowest FLAIR-vs-header agreement:", [(b["case"], round(b["c2_flair_hdr"], 3)) for b in bad])
json.dump(res, open(f"{S}/channels.json", "w"), indent=0)
