#!/usr/bin/env python3
"""Write the nnU-Net raw datasets from the original ISLES files and the repaired labels.

Dataset522  DWI + ADC            (primary analysis; both come from one acquisition)
Dataset523  DWI + ADC + FLAIR    (modality comparison only; FLAIR is resampled onto the
                                  DWI grid from header geometry, it is NOT registered)

Images are written in native DWI space. Every case is checked for matching
shape and affine between channels and label before it is written.
"""
import argparse
import json
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform

from strokebench import paths


def same_grid(a, b, atol=1e-3) -> bool:
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=atol)


def resample_to(src, ref, order=1) -> np.ndarray:
    """Resample `src` onto the voxel grid of `ref` using header geometry only."""
    mat = np.linalg.inv(src.affine) @ ref.affine
    data = np.asanyarray(src.dataobj).astype(np.float32)
    return affine_transform(data, mat[:3, :3], mat[:3, 3], output_shape=ref.shape[:3], order=order, mode="constant", cval=0.0)


def write_like(data: np.ndarray, ref, path, dtype) -> None:
    header = nib.Nifti1Header()
    header.set_data_dtype(dtype)
    img = nib.Nifti1Image(np.asarray(data, dtype=dtype), ref.affine, header)
    img.set_qform(ref.affine, code=1)
    img.set_sform(ref.affine, code=1)
    nib.save(img, path)


def build(key: str, limit: int) -> None:
    ds_id, ds_name = paths.DATASETS[key]
    channels = ["DWI", "ADC"] + (["FLAIR"] if key == "dwi_adc_flair" else [])
    out = paths.NNUNET_RAW / ds_name
    (out / "imagesTr").mkdir(parents=True, exist_ok=True)
    (out / "labelsTr").mkdir(parents=True, exist_ok=True)

    n_resampled_adc = 0
    for i in range(1, limit + 1):
        case = paths.case_id(i)
        p = paths.raw_paths(case)
        dwi, adc, label = nib.load(p["dwi"]), nib.load(p["adc"]), nib.load(p["label"])
        assert len(dwi.shape) == 3, f"{case}: DWI is not 3-D"
        assert same_grid(dwi, label), f"{case}: label grid differs from DWI"

        dwi_data = np.asanyarray(dwi.dataobj).astype(np.float32)
        if same_grid(dwi, adc):
            adc_data = np.asanyarray(adc.dataobj).astype(np.float32)
        else:
            adc_data = resample_to(adc, dwi)
            n_resampled_adc += 1
        lab = (np.asanyarray(label.dataobj) > 0).astype(np.uint8)
        assert np.isfinite(dwi_data).all() and np.isfinite(adc_data).all(), case

        write_like(dwi_data, dwi, out / "imagesTr" / f"{case}_0000.nii.gz", np.float32)
        write_like(adc_data, dwi, out / "imagesTr" / f"{case}_0001.nii.gz", np.float32)
        if "FLAIR" in channels:
            write_like(resample_to(nib.load(p["flair"]), dwi), dwi, out / "imagesTr" / f"{case}_0002.nii.gz", np.float32)
        write_like(lab, dwi, out / "labelsTr" / f"{case}.nii.gz", np.uint8)

    dataset_json = {
        "name": ds_name,
        "description": "ISLES 2022 training release, labels rebuilt from the original masks (threshold 0.5)",
        "reference": "Hernandez Petzsche et al., Sci Data 9, 762 (2022)",
        "licence": "CC BY 4.0",
        "channel_names": {str(k): c for k, c in enumerate(channels)},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": limit,
        "file_ending": ".nii.gz",
    }
    with open(out / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=1)
    print(f"{ds_name}: wrote {limit} cases, channels {channels}, ADC resampled for {n_resampled_adc} cases")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(paths.DATASETS), required=True)
    ap.add_argument("--limit", type=int, default=paths.N_CASES)
    args = ap.parse_args()
    os.makedirs(paths.NNUNET_RAW, exist_ok=True)
    build(args.dataset, args.limit)


if __name__ == "__main__":
    main()
