#!/usr/bin/env python3
"""Build the per-case table (lesion burden, acquisition geometry, demographics).

The table drives fold stratification and every stratified analysis. It is built
only from the original ISLES files and the repaired labels, never from any
model output.

Protocol groups are defined from acquisition geometry alone, because scanner
vendor tags are missing for roughly 100 subjects:
  A  thin-slice  : DWI slice thickness < 3 mm
  B1 thick-slice : DWI slice thickness >= 4 mm, in-plane spacing >= 1.5 mm
  B2 thick-slice : DWI slice thickness >= 4 mm, in-plane spacing <  1.5 mm
"""
import csv
import json

import cc3d
import nibabel as nib
import numpy as np

from strokebench import paths


def protocol_group(spacing) -> str:
    in_plane, thickness = float(spacing[0]), float(spacing[2])
    if thickness < 3.0:
        return "A"
    if thickness >= 4.0:
        return "B1" if in_plane >= 1.5 else "B2"
    raise ValueError(f"unexpected slice thickness {thickness}")


def read_participants() -> dict:
    out = {}
    with open(paths.RAW_ISLES / "participants.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            age = row.get("age", "n/a")
            age = int(age.rstrip("Y")) if age not in ("n/a", "", None) else ""
            sex = row.get("sex", "n/a")
            out[row["participant_id"]] = {"sex": "" if sex == "n/a" else sex, "age": age}
    return out


def build() -> list:
    people = read_participants()
    rows = []
    for i in range(1, paths.N_CASES + 1):
        case = paths.case_id(i)
        p = paths.raw_paths(case)
        dwi = nib.load(p["dwi"])
        label_img = nib.load(p["label"])
        label = np.asanyarray(label_img.dataobj) > 0
        assert label.shape == dwi.shape[:3], case
        assert np.allclose(label_img.affine, dwi.affine, atol=1e-3), case
        spacing = [float(z) for z in dwi.header.get_zooms()[:3]]
        voxel_ml = float(np.prod(spacing)) / 1000.0
        n_lesions = int(cc3d.connected_components(label.astype(np.uint8), connectivity=26, return_N=True)[1])
        meta_path = str(p["dwi"]).replace(".nii.gz", ".json")
        try:
            meta = json.load(open(meta_path))
        except FileNotFoundError:
            meta = {}
        rows.append({
            "case": case,
            "protocol": protocol_group(spacing),
            "thick_slice": int(spacing[2] >= 4.0),
            "shape": "x".join(str(s) for s in dwi.shape[:3]),
            "spacing_x": round(spacing[0], 4), "spacing_y": round(spacing[1], 4), "spacing_z": round(spacing[2], 4),
            "voxel_ml": round(voxel_ml, 8),
            "lesion_voxels": int(label.sum()),
            "lesion_ml": round(float(label.sum()) * voxel_ml, 4),
            "n_lesions": n_lesions,
            "vendor": meta.get("Manufacturer", ""),
            "field_T": meta.get("MagneticFieldStrength", ""),
            "sex": people.get(case, {}).get("sex", ""),
            "age": people.get(case, {}).get("age", ""),
        })
    return rows


def main() -> None:
    rows = build()
    paths.DATA.mkdir(parents=True, exist_ok=True)
    with open(paths.CASES_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    pos = [r for r in rows if r["lesion_voxels"] > 0]
    vol = np.array([r["lesion_ml"] for r in pos])
    print(f"cases {len(rows)} | lesion-positive {len(pos)} | empty {[r['case'] for r in rows if r['lesion_voxels'] == 0]}")
    for g in ("A", "B1", "B2"):
        sub = [r for r in rows if r["protocol"] == g]
        v = np.array([r["lesion_ml"] for r in sub if r["lesion_voxels"] > 0])
        print(f"  protocol {g}: n={len(sub)} median lesion {np.median(v):.2f} mL, median lesion count {np.median([r['n_lesions'] for r in sub]):.0f}")
    print(f"lesion volume mL: median {np.median(vol):.2f}, IQR {np.percentile(vol, 25):.2f}-{np.percentile(vol, 75):.2f}, range {vol.min():.3f}-{vol.max():.1f}")
    print(f"wrote {paths.CASES_CSV}")


if __name__ == "__main__":
    main()
