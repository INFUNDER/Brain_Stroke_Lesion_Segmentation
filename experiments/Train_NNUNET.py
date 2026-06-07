#!/usr/bin/env python3
"""
STEP 1: nnU-Net Data Conversion + Training
==========================================
nnU-Net expects a specific folder structure.
This script:
  1. Converts ISLES2022 data to nnU-Net format
  2. Runs nnU-Net preprocessing
  3. Trains all 5 folds
  4. Runs prediction on test set

Run this FIRST before ensemble script.
"""

import os, sys, shutil, json, glob
import numpy as np
import subprocess

# =============================================================================
# PATHS — adjust if needed
# =============================================================================
ISLES_DIR   = "/home/ronit.28010/Brain_Stroke/ISLES2022_Formatted"
NNUNET_BASE = "/home/ronit.28010/Brain_Stroke/nnunet_workspace"
SEED        = 42

# nnU-Net environment variables (must be set before training)
os.environ["nnUNet_raw"]          = os.path.join(NNUNET_BASE, "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = os.path.join(NNUNET_BASE, "nnUNet_preprocessed")
os.environ["nnUNet_results"]      = os.path.join(NNUNET_BASE, "nnUNet_results")

DATASET_ID   = 101          # arbitrary ID, must be 3 digits
DATASET_NAME = f"Dataset{DATASET_ID:03d}_ISLES2022"
RAW_DIR      = os.path.join(os.environ["nnUNet_raw"], DATASET_NAME)

for d in [NNUNET_BASE,
          os.environ["nnUNet_raw"],
          os.environ["nnUNet_preprocessed"],
          os.environ["nnUNet_results"]]:
    os.makedirs(d, exist_ok=True)

print("=" * 60)
print("  STEP 1: nnU-Net Setup + Training")
print("=" * 60)
print(f"ISLES data   : {ISLES_DIR}")
print(f"nnU-Net base : {NNUNET_BASE}")
print(f"Dataset ID   : {DATASET_ID}")

# =============================================================================
# LOAD THE SAME TEST SPLIT used by MambaUNet
# So both models evaluate on identical test cases
# =============================================================================
MAMBA_RESULTS = os.path.join(
    os.environ.get("PBS_O_WORKDIR", "."), "mamba_results_v5"
)
TEST_PATHS_FILE = os.path.join(MAMBA_RESULTS, "test_set_paths.txt")

if not os.path.exists(TEST_PATHS_FILE):
    print(f"ERROR: {TEST_PATHS_FILE} not found!")
    print("Run MambaUNet pipeline first so the test split is fixed.")
    sys.exit(1)

test_image_paths = set()
with open(TEST_PATHS_FILE) as f:
    for line in f:
        parts = line.strip().split("  |  ")
        if len(parts) == 2:
            test_image_paths.add(parts[0].strip())

print(f"Test cases (from MambaUNet split): {len(test_image_paths)}")

# =============================================================================
# CONVERT ISLES → nnU-Net FORMAT
#
# nnU-Net expects:
#   imagesTr/  ISLES2022_0000_0000.nii.gz (modality 0 = DWI)
#              ISLES2022_0000_0001.nii.gz (modality 1 = ADC)
#              ISLES2022_0000_0002.nii.gz (modality 2 = FLAIR)
#   labelsTr/  ISLES2022_0000.nii.gz
#   imagesTs/  (test set — no labels)
# =============================================================================
tr_img_dir = os.path.join(RAW_DIR, "imagesTr")
tr_lbl_dir = os.path.join(RAW_DIR, "labelsTr")
ts_img_dir = os.path.join(RAW_DIR, "imagesTs")
for d in [tr_img_dir, tr_lbl_dir, ts_img_dir]:
    os.makedirs(d, exist_ok=True)

all_images = sorted(glob.glob(os.path.join(ISLES_DIR, "imagesTr", "*.nii.gz")))
all_labels = sorted(glob.glob(os.path.join(ISLES_DIR, "labelsTr", "*.nii.gz")))
assert len(all_images) == len(all_labels)

print(f"\nConverting {len(all_images)} cases to nnU-Net format...")
print("(ISLES stores 3 modalities as channels in one file)")
print("(nnU-Net needs them as separate files _0000, _0001, _0002)")

import nibabel as nib

train_cases, test_cases = [], []
case_idx = 0

for img_path, lbl_path in zip(all_images, all_labels):
    case_name = os.path.basename(img_path).replace(".nii.gz", "")
    nn_case   = f"ISLES2022_{case_idx:04d}"
    is_test   = img_path in test_image_paths

    # Load multi-channel image (H, W, D, C) where C=3 (DWI, ADC, FLAIR)
    img_nib = nib.load(img_path)
    img_data = img_nib.get_fdata()   # shape (H, W, D, 3) or (H, W, D)

    if img_data.ndim == 4:
        # 3 modalities stored as 4th dim
        for mod_idx in range(img_data.shape[-1]):
            mod_data = img_data[..., mod_idx]
            mod_nib  = nib.Nifti1Image(mod_data, img_nib.affine, img_nib.header)
            out_name = f"{nn_case}_{mod_idx:04d}.nii.gz"
            out_dir  = ts_img_dir if is_test else tr_img_dir
            nib.save(mod_nib, os.path.join(out_dir, out_name))
    else:
        # Single modality — save as _0000
        mod_nib  = nib.Nifti1Image(img_data, img_nib.affine, img_nib.header)
        out_name = f"{nn_case}_0000.nii.gz"
        out_dir  = ts_img_dir if is_test else tr_img_dir
        nib.save(mod_nib, os.path.join(out_dir, out_name))

    if not is_test:
        # Copy label
        lbl_nib = nib.load(lbl_path)
        nib.save(lbl_nib, os.path.join(tr_lbl_dir, f"{nn_case}.nii.gz"))
        train_cases.append(nn_case)
    else:
        test_cases.append(nn_case)

    case_idx += 1

print(f"  Train cases: {len(train_cases)}")
print(f"  Test cases : {len(test_cases)}")

# =============================================================================
# CREATE dataset.json — nnU-Net metadata file
# =============================================================================
n_modalities = nib.load(all_images[0]).get_fdata().ndim
if nib.load(all_images[0]).get_fdata().ndim == 4:
    n_mod = nib.load(all_images[0]).get_fdata().shape[-1]
else:
    n_mod = 1

channel_names = {str(i): f"modality_{i}" for i in range(n_mod)}
# ISLES modality order: 0=DWI, 1=ADC, 2=FLAIR
if n_mod == 3:
    channel_names = {"0": "DWI", "1": "ADC", "2": "FLAIR"}
elif n_mod == 2:
    channel_names = {"0": "DWI", "1": "ADC"}

dataset_json = {
    "channel_names": channel_names,
    "labels": {"background": 0, "stroke_lesion": 1},
    "numTraining": len(train_cases),
    "file_ending": ".nii.gz",
    "overwrite_image_reader_writer": "SimpleITKIO"
}

with open(os.path.join(RAW_DIR, "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=2)

print(f"\ndataset.json written: {n_mod} modalities, {len(train_cases)} training cases")

# =============================================================================
# SAVE TEST CASE NAMES for ensemble script
# =============================================================================
test_names_file = os.path.join(NNUNET_BASE, "test_case_names.txt")
with open(test_names_file, "w") as f:
    for name in test_cases:
        f.write(name + "\n")
print(f"Test case names saved: {test_names_file}")

# =============================================================================
# RUN nnU-Net COMMANDS
# =============================================================================
print("\n" + "=" * 60)
print("  Running nnU-Net pipeline...")
print("=" * 60)

def run_cmd(cmd, desc):
    print(f"\n>>> {desc}")
    print(f"    {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: command failed with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"    Done.")

# Step A: Plan and preprocess
run_cmd(
    f"nnUNetv2_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity",
    "Planning and preprocessing"
)

# Step B: Train all 5 folds
for fold in range(5):
    run_cmd(
        f"nnUNetv2_train {DATASET_ID} 3d_fullres {fold} "
        f"--npz",   # save softmax probabilities (needed for ensemble)
        f"Training fold {fold+1}/5"
    )

# Step C: Predict on test set (saves softmax .npz files for ensemble)
PRED_DIR = os.path.join(NNUNET_BASE, "test_predictions")
os.makedirs(PRED_DIR, exist_ok=True)

run_cmd(
    f"nnUNetv2_predict "
    f"-i {ts_img_dir} "
    f"-o {PRED_DIR} "
    f"-d {DATASET_ID} "
    f"-c 3d_fullres "
    f"-f all "           # ensemble of all 5 folds
    f"--save_probabilities",  # save .npz softmax for our ensemble
    "Predicting test set (saving softmax probabilities)"
)

print("\n" + "=" * 60)
print("  nnU-Net DONE")
print(f"  Predictions saved to: {PRED_DIR}")
print(f"  Now run: python step3_ensemble_eval.py")
print("=" * 60)