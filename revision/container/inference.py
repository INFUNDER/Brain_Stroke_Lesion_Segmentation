#!/usr/bin/env python3
"""ISLES 2022 hidden-test algorithm (PROTOCOL.md, amendment 2026-11-26, second entry, E5a).

Submission 1: nnU-Net, 5-fold ensemble of the frozen cross-validation models, DWI + ADC,
nnU-Net's default inference (sliding window, step 0.5, Gaussian weighting, mirroring).

Grand-challenge interface (VERIFY against the challenge's algorithm page before submitting):
  input   /input/images/dwi-brain-mri/<uuid>.mha    /input/images/adc-brain-mri/<uuid>.mha
          (/input/images/flair-brain-mri/ is ignored)
  output  /output/images/stroke-lesion-segmentation/<uuid>.mha, uint8, 0/1, on the DWI grid
Paths can be overridden with the environment variables below, which also serve the local test.
"""
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

INPUT = Path(os.environ.get("ISLES_INPUT", "/input"))
OUTPUT = Path(os.environ.get("ISLES_OUTPUT", "/output"))
MODEL = Path(os.environ.get("ISLES_MODEL", "/opt/algorithm/model"))   # nnU-Net trained-model folder
FOLDS = tuple(int(f) for f in os.environ.get("ISLES_FOLDS", "0,1,2,3,4").split(","))
MIRRORING = os.environ.get("ISLES_MIRRORING", "1") == "1"
DWI_DIR, ADC_DIR = "images/dwi-brain-mri", "images/adc-brain-mri"
OUT_DIR = "images/stroke-lesion-segmentation"


def single_image(folder: Path) -> Path:
    files = sorted(p for p in folder.iterdir() if p.suffix in (".mha", ".gz", ".nii"))
    if len(files) != 1:
        raise RuntimeError(f"expected one image in {folder}, found {[p.name for p in files]}")
    return files[0]


def on_grid(moving: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """ADC on the DWI grid (nnU-Net needs identical geometry for all channels)."""
    same = (moving.GetSize() == reference.GetSize()
            and np.allclose(moving.GetSpacing(), reference.GetSpacing(), atol=1e-3)
            and np.allclose(moving.GetOrigin(), reference.GetOrigin(), atol=1e-2)
            and np.allclose(moving.GetDirection(), reference.GetDirection(), atol=1e-3))
    if same:
        moving.CopyInformation(reference)
        return moving
    return sitk.Resample(moving, reference, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)


def main() -> None:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    dwi_path, adc_path = single_image(INPUT / DWI_DIR), single_image(INPUT / ADC_DIR)
    dwi = sitk.Cast(sitk.ReadImage(str(dwi_path)), sitk.sitkFloat32)
    adc = on_grid(sitk.Cast(sitk.ReadImage(str(adc_path)), sitk.sitkFloat32), dwi)
    work = Path(tempfile.mkdtemp())
    try:
        sitk.WriteImage(dwi, str(work / "case_0000.nii.gz"))
        sitk.WriteImage(adc, str(work / "case_0001.nii.gz"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=MIRRORING,
                                    perform_everything_on_device=device.type == "cuda", device=device,
                                    verbose=False, allow_tqdm=False)
        predictor.initialize_from_trained_model_folder(str(MODEL), use_folds=FOLDS, checkpoint_name="checkpoint_final.pth")
        predictor.predict_from_files([[str(work / "case_0000.nii.gz"), str(work / "case_0001.nii.gz")]],
                                     [str(work / "pred")], save_probabilities=False, overwrite=True,
                                     num_processes_preprocessing=1, num_processes_segmentation_export=1)
        seg = sitk.ReadImage(str(work / "pred.nii.gz"))
        arr = (sitk.GetArrayFromImage(seg) > 0).astype(np.uint8)
        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(dwi)
        (OUTPUT / OUT_DIR).mkdir(parents=True, exist_ok=True)
        name = dwi_path.name.replace(".nii.gz", ".mha").replace(".nii", ".mha")
        sitk.WriteImage(out, str(OUTPUT / OUT_DIR / name), useCompression=True)
        print(f"wrote {OUTPUT / OUT_DIR / name}: {int(arr.sum())} lesion voxels")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
