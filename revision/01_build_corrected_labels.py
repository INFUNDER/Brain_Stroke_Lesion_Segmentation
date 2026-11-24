#!/usr/bin/env python3
"""Rebuild ISLES-2022 labels directly from the original derivative masks.

Why this exists: 47 of the 250 original masks are stored as int16 with an
scl_slope of 1.5259e-05, so their lesion value reads back as 0.9999999998.
Casting that to an integer gives 0, which blanks the whole lesion. The previous
ISLES2022_Formatted/labelsTr has exactly those 47 cases blank. Thresholding the
original mask at 0.5 is safe for every storage type in the dataset.

Nothing in ISLES2022_Formatted is modified. Output goes to ISLES2022_Corrected/.
"""
import argparse
import json
import os

import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "isless2022dataset", "ISLES-2022_notformatted")
OLD = os.path.join(ROOT, "ISLES2022_Formatted")
NEW = os.path.join(ROOT, "ISLES2022_Corrected")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=250)
    args = ap.parse_args()

    os.makedirs(os.path.join(NEW, "labelsTr"), exist_ok=True)
    images = os.path.join(NEW, "imagesTr")
    if not os.path.lexists(images):
        os.symlink(os.path.join(OLD, "imagesTr"), images)

    summary = []
    for i in range(1, args.n_cases + 1):
        case = f"sub-strokecase{i:04d}"
        raw_path = os.path.join(RAW, "derivatives", case, "ses-0001", f"{case}_ses-0001_msk.nii.gz")
        raw = nib.load(raw_path)
        raw_data = np.asanyarray(raw.dataobj)
        fixed = (raw_data > 0.5).astype(np.uint8)

        out_path = os.path.join(NEW, "labelsTr", f"{case}.nii.gz")
        header = raw.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(fixed, raw.affine, header), out_path)

        # Read the written file back and check it against the source mask.
        written = nib.load(out_path)
        back = np.asanyarray(written.dataobj)
        assert back.shape == raw_data.shape, case
        assert np.allclose(written.affine, raw.affine), case
        assert set(np.unique(back).tolist()) <= {0, 1}, case
        assert int(back.sum()) == int((raw_data > 0).sum()), f"{case}: voxel count changed"

        old = np.asanyarray(nib.load(os.path.join(OLD, "labelsTr", f"{case}.nii.gz")).dataobj) > 0
        state = "empty_in_source" if back.sum() == 0 else ("repaired" if old.sum() == 0 else "unchanged")
        if state == "unchanged":
            assert np.array_equal(old, back > 0), f"{case}: previously intact label differs"
        summary.append(dict(case=case, voxels=int(back.sum()), state=state))

    with open(os.path.join(NEW, "label_repair_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    counts = {s: sum(r["state"] == s for r in summary) for s in ("unchanged", "repaired", "empty_in_source")}
    print(counts)
    print("empty in source:", [r["case"] for r in summary if r["state"] == "empty_in_source"])


if __name__ == "__main__":
    main()
