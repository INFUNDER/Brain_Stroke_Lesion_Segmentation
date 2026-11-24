"""Single place for every path used in the revision experiments."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVISION = ROOT / "revision"

# Inputs (never modified).
RAW_ISLES = ROOT / "isless2022dataset" / "ISLES-2022_notformatted"
FORMATTED = ROOT / "ISLES2022_Formatted"      # old 3-channel images; labels there are corrupted, never use them
CORRECTED = ROOT / "ISLES2022_Corrected"      # labels rebuilt by 01_build_corrected_labels.py

# Version-controlled study metadata.
DATA = REVISION / "data"
CASES_CSV = DATA / "cases.csv"
SPLITS_JSON = DATA / "splits_final.json"

# nnU-Net working directories (large, git-ignored).
NNUNET = ROOT / "nnunet"
NNUNET_RAW = NNUNET / "raw"
NNUNET_PREPROCESSED = NNUNET / "preprocessed"
NNUNET_RESULTS = NNUNET / "results"

# Primary analysis uses DWI+ADC (both native to one acquisition, no registration needed).
# The FLAIR dataset exists only for the pre-specified modality comparison.
DATASETS = {
    "dwi_adc": (522, "Dataset522_ISLES22DwiAdc"),
    "dwi_adc_flair": (523, "Dataset523_ISLES22DwiAdcFlair"),
}

N_CASES = 250


def case_id(i: int) -> str:
    return f"sub-strokecase{i:04d}"


def raw_paths(case: str) -> dict:
    ses = "ses-0001"
    return {
        "dwi": RAW_ISLES / case / ses / "dwi" / f"{case}_{ses}_dwi.nii.gz",
        "adc": RAW_ISLES / case / ses / "dwi" / f"{case}_{ses}_adc.nii.gz",
        "flair": RAW_ISLES / case / ses / "anat" / f"{case}_{ses}_FLAIR.nii.gz",
        "mask": RAW_ISLES / "derivatives" / case / ses / f"{case}_{ses}_msk.nii.gz",
        "label": CORRECTED / "labelsTr" / f"{case}.nii.gz",
    }


def set_nnunet_env() -> None:
    """Must run before `import nnunetv2` (nnU-Net reads these at import time)."""
    os.environ["nnUNet_raw"] = str(NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    os.environ["nnUNet_results"] = str(NNUNET_RESULTS)
    # torch.compile is switched off for every model so that timings are comparable
    # and the custom state-space operators are not traced.
    os.environ.setdefault("nnUNet_compile", "f")
