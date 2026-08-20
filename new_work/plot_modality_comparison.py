#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import LoadImaged

WORKSPACE_DIR = "/home/ronit.28010/Brain_Stroke"
DATA_DIR = os.path.join(WORKSPACE_DIR, "ISLES2022_Formatted")
CACHE_DIR = os.path.join(WORKSPACE_DIR, "mamba_results/non_leaked_probs")
OUTPUT_PATH = os.path.join(WORKSPACE_DIR, "figures", "modality_prediction_comparison.png")

CASE_NAME = "sub-strokecase0047"
image_file = os.path.join(DATA_DIR, "imagesTr", f"{CASE_NAME}.nii.gz")
label_file = os.path.join(DATA_DIR, "labelsTr", f"{CASE_NAME}.nii.gz")

print(f"📖 Loading data for Case: {CASE_NAME}")
raw_data = LoadImaged(keys=["image", "label"])({"image": image_file, "label": label_file})
raw_img = raw_data["image"]  # (H, W, D, 3)
gt_np = np.load(os.path.join(CACHE_DIR, f"{CASE_NAME}_gt.npy"))
mamba_prob = np.load(os.path.join(CACHE_DIR, f"{CASE_NAME}_prob.npy"))
mamba_pred = (mamba_prob >= 0.45).astype(np.uint8)

H, W, D = gt_np.shape
lesion_areas = [gt_np[:, :, z].sum() for z in range(D)]
z_idx = int(np.argmax(lesion_areas))
print(f"Slice index with maximum lesion: {z_idx}")

dwi_slice = raw_img[:, :, z_idx, 0]
adc_slice = raw_img[:, :, z_idx, 1]
flair_slice = raw_img[:, :, z_idx, 2]
mamba_slice = mamba_pred[:, :, z_idx]

def orient(img_slice):
    return np.rot90(img_slice)

dwi_plot = orient(dwi_slice)
adc_plot = orient(adc_slice)
flair_plot = orient(flair_slice)
mamba_plot = orient(mamba_slice)

# Create 1x4 subplot
fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))

# DWI
axes[0].imshow(dwi_plot, cmap="gray")
axes[0].set_title("DWI (b=1000)", fontsize=16, fontweight="bold")
axes[0].axis("off")

# ADC
axes[1].imshow(adc_plot, cmap="gray")
axes[1].set_title("ADC Map", fontsize=16, fontweight="bold")
axes[1].axis("off")

# FLAIR
axes[2].imshow(flair_plot, cmap="gray")
axes[2].set_title("FLAIR", fontsize=16, fontweight="bold")
axes[2].axis("off")

# OmniMamba-UNet prediction overlay on DWI
axes[3].imshow(dwi_plot, cmap="gray")
mamba_mask = np.ma.masked_where(mamba_plot == 0, mamba_plot)
axes[3].imshow(mamba_mask, cmap="winter", alpha=0.5)
axes[3].contour(mamba_plot, colors="cyan", levels=[0.5], linewidths=2.5)
axes[3].set_title("OmniMamba-UNet Prediction", fontsize=16, fontweight="bold")
axes[3].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"✅ Figure saved to: {OUTPUT_PATH}")
