#!/usr/bin/env python3
import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, NormalizeIntensityd, EnsureTyped
)
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR

# Set plotting style for academic publication
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

WORKSPACE_DIR = "/home/ronit.28010/Brain_Stroke"
DATA_DIR = os.path.join(WORKSPACE_DIR, "ISLES2022_Formatted")
CACHE_DIR = os.path.join(WORKSPACE_DIR, "mamba_results/non_leaked_probs")
CKPT_PATH = os.path.join(WORKSPACE_DIR, "checkpoints/best_metric_model.pth")
OUTPUT_PATH = os.path.join(WORKSPACE_DIR, "lesion_predictions_comparison.png")

# Select a representative case
# Case 0047 has a clean, medium-sized lesion (11,501 voxels)
CASE_NAME = "sub-strokecase0047"
image_file = os.path.join(DATA_DIR, "imagesTr", f"{CASE_NAME}.nii.gz")
label_file = os.path.join(DATA_DIR, "labelsTr", f"{CASE_NAME}.nii.gz")

print(f"📖 Processing Case: {CASE_NAME}")

# 1. Load and Preprocess Image
transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image"], channel_dim=-1),
    EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image", "label"]),
])

data = transforms({"image": image_file, "label": label_file})
img_tensor = data["image"]  # (3, H, W, D)
gt_tensor = data["label"]    # (1, H, W, D)

# Raw un-normalized images for background display (so they look like standard MRIs)
raw_data = LoadImaged(keys=["image", "label"])({"image": image_file, "label": label_file})
raw_img = raw_data["image"]  # (H, W, D, 3)

# 2. Run UNETR CPU Inference
print("🤖 Running UNETR Inference on CPU...")
device = torch.device("cpu")
unetr = UNETR(
    in_channels=3,
    out_channels=2,
    img_size=(96, 96, 96),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    proj_type="perceptron",
    norm_name="instance",
    res_block=True,
)
unetr.load_state_dict(torch.load(CKPT_PATH, map_location=device))
unetr.eval()

with torch.no_grad():
    # Add batch dim
    inputs = img_tensor.unsqueeze(0).to(device)
    outputs = sliding_window_inference(inputs, (96, 96, 96), sw_batch_size=1, predictor=unetr, overlap=0.5)
    outputs = torch.softmax(outputs, dim=1)
    unetr_prob = outputs[0, 1].numpy()  # (H, W, D)

# 3. Load Mamba-UNet Predictions from Cache
print("🐍 Loading Mamba-UNet Cached Predictions...")
mamba_prob = np.load(os.path.join(CACHE_DIR, f"{CASE_NAME}_prob.npy"))
gt_np = np.load(os.path.join(CACHE_DIR, f"{CASE_NAME}_gt.npy"))

# Apply thresholds
unetr_pred = (unetr_prob >= 0.5).astype(np.uint8)
mamba_pred = (mamba_prob >= 0.45).astype(np.uint8)

# Transpose raw image to RAS space coordinates for visualization
# Raw image shape is (H, W, D, 3) -> we want to slice it along Z-axis (D)
# MONAI spacing and orientation transforms might change the shape slightly,
# so we match the shape of the predictions.
# The prediction tensors are in shape (H, W, D).
H, W, D = gt_np.shape
print(f"Volume Shape in RAS: {H} x {W} x {D}")

# Find Z slice with maximum lesion area
lesion_areas = [gt_np[:, :, z].sum() for z in range(D)]
z_idx = int(np.argmax(lesion_areas))
print(f"🎯 Selected Slice Index (Max Lesion Area): {z_idx} (Area: {lesion_areas[z_idx]} voxels)")

# Extract slices
dwi_slice = raw_img[:, :, z_idx, 0]
adc_slice = raw_img[:, :, z_idx, 1]
flair_slice = raw_img[:, :, z_idx, 2]
gt_slice = gt_np[:, :, z_idx]
unetr_slice = unetr_pred[:, :, z_idx]
mamba_slice = mamba_pred[:, :, z_idx]

# Rotate slices for standard orientation (RAS coordinates)
def orient(img_slice):
    # Transpose and rotate to standard axial view (anterior up, patient right on left)
    return np.rot90(img_slice)

dwi_plot = orient(dwi_slice)
adc_plot = orient(adc_slice)
flair_plot = orient(flair_slice)
gt_plot = orient(gt_slice)
unetr_plot = orient(unetr_slice)
mamba_plot = orient(mamba_slice)

# 4. Generate the side-by-side figure
print("🎨 Plotting Figure...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Modality 1: DWI
axes[0, 0].imshow(dwi_plot, cmap="gray")
axes[0, 0].set_title("DWI Modality (b=1000)")
axes[0, 0].axis("off")

# Modality 2: ADC
axes[0, 1].imshow(adc_plot, cmap="gray")
axes[0, 1].set_title("ADC Map")
axes[0, 1].axis("off")

# Modality 3: FLAIR
axes[0, 2].imshow(flair_plot, cmap="gray")
axes[0, 2].set_title("FLAIR Modality")
axes[0, 2].axis("off")

# Overlay GT on DWI
axes[1, 0].imshow(dwi_plot, cmap="gray")
# Create a masked array for GT overlay
gt_mask = np.ma.masked_where(gt_plot == 0, gt_plot)
axes[1, 0].imshow(gt_mask, cmap="spring", alpha=0.5)
# Draw contour boundary
axes[1, 0].contour(gt_plot, colors="lime", levels=[0.5], linewidths=2.0)
axes[1, 0].set_title("Ground Truth Lesion")
axes[1, 0].axis("off")

# Overlay UNETR on DWI
axes[1, 1].imshow(dwi_plot, cmap="gray")
unetr_mask = np.ma.masked_where(unetr_plot == 0, unetr_plot)
axes[1, 1].imshow(unetr_mask, cmap="autumn", alpha=0.5)
axes[1, 1].contour(unetr_plot, colors="red", levels=[0.5], linewidths=2.0)
axes[1, 1].set_title("UNETR Prediction")
axes[1, 1].axis("off")

# Overlay Mamba-UNet on DWI
axes[1, 2].imshow(dwi_plot, cmap="gray")
mamba_mask = np.ma.masked_where(mamba_plot == 0, mamba_plot)
axes[1, 2].imshow(mamba_mask, cmap="winter", alpha=0.5)
axes[1, 2].contour(mamba_plot, colors="cyan", levels=[0.5], linewidths=2.0)
axes[1, 2].set_title("Mamba-UNet Prediction")
axes[1, 2].axis("off")

# Add layout spacing and main title
plt.suptitle(f"Qualitative Segmentation Comparison on ISLES 2022 ({CASE_NAME})", y=0.98)
plt.tight_layout()

# Save at 300 DPI for publication
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"✅ Figure successfully saved to: {OUTPUT_PATH}")
