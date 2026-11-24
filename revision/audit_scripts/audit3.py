#!/usr/bin/env python3
"""Root-cause the blank labels and re-score saved test predictions against the true masks. Read-only."""
import os, glob, json, sys, collections
import numpy as np
import nibabel as nib

ROOT = "/home/ronit.28010/Brain_Stroke"
RAW = f"{ROOT}/isless2022dataset/ISLES-2022_notformatted"
FMT = f"{ROOT}/ISLES2022_Formatted"
S = sys.argv[1]

def geom(n):
    return tuple(int(x) for x in n.shape[:3]), tuple(round(float(z), 3) for z in n.header.get_zooms()[:3])

rows = []
for i in range(1, 251):
    c = f"sub-strokecase{i:04d}"
    ses = "ses-0001"
    p = dict(
        dwi=f"{RAW}/{c}/{ses}/dwi/{c}_{ses}_dwi.nii.gz",
        adc=f"{RAW}/{c}/{ses}/dwi/{c}_{ses}_adc.nii.gz",
        flair=f"{RAW}/{c}/{ses}/anat/{c}_{ses}_FLAIR.nii.gz",
        msk=f"{RAW}/derivatives/{c}/{ses}/{c}_{ses}_msk.nii.gz",
    )
    if not all(os.path.exists(v) for v in p.values()):
        print("missing", c, [k for k, v in p.items() if not os.path.exists(v)])
        continue
    n = {k: nib.load(v) for k, v in p.items()}
    fi = nib.load(f"{FMT}/imagesTr/{c}.nii.gz")
    fl = nib.load(f"{FMT}/labelsTr/{c}.nii.gz")
    raw_mask = np.asanyarray(n["msk"].dataobj) > 0
    fmt_mask = np.asanyarray(fl.dataobj) > 0
    js = json.load(open(p["dwi"].replace(".nii.gz", ".json"))) if os.path.exists(p["dwi"].replace(".nii.gz", ".json")) else {}
    rows.append(dict(
        case=c,
        raw_fg=int(raw_mask.sum()), fmt_fg=int(fmt_mask.sum()),
        dwi=geom(n["dwi"]), adc=geom(n["adc"]), flair=geom(n["flair"]), msk=geom(n["msk"]),
        fmt_img=geom(fi), fmt_lbl=geom(fl),
        msk_eq_dwi_affine=bool(np.allclose(n["msk"].affine, n["dwi"].affine, atol=1e-2)),
        fmt_eq_dwi_affine=bool(np.allclose(fi.affine, n["dwi"].affine, atol=1e-2)),
        same_shape_msk_fmtlbl=(n["msk"].shape[:3] == fl.shape[:3]),
        vendor=js.get("Manufacturer"), model=js.get("ManufacturersModelName"), field=js.get("MagneticFieldStrength"), inst=js.get("InstitutionName"),
    ))
json.dump(rows, open(f"{S}/geometry.json", "w"), indent=0)
print("rows", len(rows))
blank = [r for r in rows if r["raw_fg"] > 0 and r["fmt_fg"] == 0]
truly_empty = [r for r in rows if r["raw_fg"] == 0]
ok = [r for r in rows if r["raw_fg"] > 0 and r["fmt_fg"] > 0]
print("wrongly blank:", len(blank), " truly empty raw:", [r["case"] for r in truly_empty], " ok:", len(ok))
print("identical fg among ok:", sum(r["raw_fg"] == r["fmt_fg"] for r in ok))
for name, grp in [("BLANK", blank), ("OK", ok)]:
    print(f"--- {name}")
    print("  msk shape==formatted label shape:", collections.Counter(r["same_shape_msk_fmtlbl"] for r in grp))
    print("  msk affine==dwi affine:", collections.Counter(r["msk_eq_dwi_affine"] for r in grp))
    print("  formatted image affine==dwi affine:", collections.Counter(r["fmt_eq_dwi_affine"] for r in grp))
    print("  dwi shape/spacing top:", collections.Counter((r["dwi"]) for r in grp).most_common(5))
    print("  msk shape/spacing top:", collections.Counter((r["msk"]) for r in grp).most_common(5))
    print("  fmt lbl shape/spacing top:", collections.Counter((r["fmt_lbl"]) for r in grp).most_common(5))
    print("  vendor/field:", collections.Counter((r["vendor"], r["field"]) for r in grp).most_common(8))
    print("  model:", collections.Counter((r["model"]) for r in grp).most_common(8))
print("sequential index of blank:", [int(r["case"][-4:]) for r in blank])

# --- all-cohort scanner table
print("=== full cohort scanners")
print(collections.Counter((r["vendor"], r["model"], r["field"]) for r in rows).most_common(20))
print("flair shape/spacing top:", collections.Counter((r["flair"]) for r in rows).most_common(6))
print("institution:", collections.Counter(r["inst"] for r in rows).most_common(6))
