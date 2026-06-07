#!/usr/bin/env python3
import os
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

UNETR_GRID_PATH = os.path.join(WORKSPACE_DIR, "figures", "unetr_grid_predictions.png")
MAMBA_GRID_PATH = os.path.join(WORKSPACE_DIR, "figures", "mamba_grid_predictions.png")

# 3 Subjects representing different stroke lesion shapes & sizes
SUBJECTS = ["sub-strokecase0047", "sub-strokecase0246", "sub-strokecase0222"]

device = torch.device("cpu")
print("🤖 Loading UNETR Model...")
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

# Spacing/Orientation Transforms
transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image"], channel_dim=-1),
    EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image", "label"]),
])

raw_loader = LoadImaged(keys=["image", "label"])

# Helper function to rotate/orient slices for standard axial view
def orient(img_slice):
    return np.rot90(img_slice)

# Store loaded and computed data for all 3 subjects
data_store = []

print("📂 Loading data and running UNETR CPU inference...")
for subj in SUBJECTS:
    image_file = os.path.join(DATA_DIR, "imagesTr", f"{subj}.nii.gz")
    label_file = os.path.join(DATA_DIR, "labelsTr", f"{subj}.nii.gz")
    
    # Process
    data_dict = transforms({"image": image_file, "label": label_file})
    img_tensor = data_dict["image"]
    
    # Load Mamba-UNet probability and GT
    mamba_prob = np.load(os.path.join(CACHE_DIR, f"{subj}_prob.npy"))
    gt_np = np.load(os.path.join(CACHE_DIR, f"{subj}_gt.npy"))
    
    # Run UNETR CPU inference
    with torch.no_grad():
        inputs = img_tensor.unsqueeze(0).to(device)
        outputs = sliding_window_inference(inputs, (96, 96, 96), sw_batch_size=1, predictor=unetr, overlap=0.5)
        outputs = torch.softmax(outputs, dim=1)
        unetr_prob = outputs[0, 1].numpy()  # (H, W, D)
        
    unetr_pred = (unetr_prob >= 0.5).astype(np.uint8)
    mamba_pred = (mamba_prob >= 0.45).astype(np.uint8)
    
    # Find Z slice with max lesion cross-section
    D = gt_np.shape[2]
    lesion_areas = [gt_np[:, :, z].sum() for z in range(D)]
    z_idx = int(np.argmax(lesion_areas))
    
    # Extract slices directly from transformed image tensor
    dwi_slice = img_tensor[0, :, :, z_idx].numpy()
    gt_slice = gt_np[:, :, z_idx]
    unetr_slice = unetr_pred[:, :, z_idx]
    mamba_slice = mamba_pred[:, :, z_idx]
    
    # Save oriented slices
    data_store.append({
        "subject": subj,
        "dwi": orient(dwi_slice),
        "gt": orient(gt_slice),
        "unetr": orient(unetr_slice),
        "mamba": orient(mamba_slice)
    })
    print(f"  Processed {subj} | Selected Slice: {z_idx}")

# =============================================================================
# FIGURE 1: UNETR 3x3 GRID
# =============================================================================
print("🎨 Drawing UNETR 3x3 Grid...")
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for row_idx, data in enumerate(data_store):
    dwi = data["dwi"]
    gt = data["gt"]
    pred = data["unetr"]
    
    # Col 1: Original DWI
    axes[row_idx, 0].imshow(dwi, cmap="gray")
    if row_idx == 0:
        axes[row_idx, 0].set_title("Original (DWI)")
    axes[row_idx, 0].set_ylabel(data["subject"], labelpad=15, rotation=90, weight="bold")
    axes[row_idx, 0].set_xticks([])
    axes[row_idx, 0].set_yticks([])
    
    # Col 2: DWI + Ground Truth Overlay
    axes[row_idx, 1].imshow(dwi, cmap="gray")
    gt_mask = np.ma.masked_where(gt == 0, gt)
    axes[row_idx, 1].imshow(gt_mask, cmap="spring", alpha=0.5)
    axes[row_idx, 1].contour(gt, colors="lime", levels=[0.5], linewidths=2.0)
    if row_idx == 0:
        axes[row_idx, 1].set_title("Ground Truth")
    axes[row_idx, 1].axis("off")
    
    # Col 3: DWI + UNETR Prediction Overlay
    axes[row_idx, 2].imshow(dwi, cmap="gray")
    pred_mask = np.ma.masked_where(pred == 0, pred)
    axes[row_idx, 2].imshow(pred_mask, cmap="autumn", alpha=0.5)
    axes[row_idx, 2].contour(pred, colors="red", levels=[0.5], linewidths=2.0)
    if row_idx == 0:
        axes[row_idx, 2].set_title("UNETR Prediction")
    axes[row_idx, 2].axis("off")

plt.suptitle("Qualitative Visualizations - UNETR (Transformer) Grid Results", y=0.98)
plt.tight_layout()
plt.savefig(UNETR_GRID_PATH, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ UNETR 3x3 Grid saved to: {UNETR_GRID_PATH}")

# =============================================================================
# FIGURE 2: MAMBA-UNET 3x3 GRID
# =============================================================================
print("🎨 Drawing Mamba-UNet 3x3 Grid...")
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for row_idx, data in enumerate(data_store):
    dwi = data["dwi"]
    gt = data["gt"]
    pred = data["mamba"]
    
    # Col 1: Original DWI
    axes[row_idx, 0].imshow(dwi, cmap="gray")
    if row_idx == 0:
        axes[row_idx, 0].set_title("Original (DWI)")
    axes[row_idx, 0].set_ylabel(data["subject"], labelpad=15, rotation=90, weight="bold")
    axes[row_idx, 0].set_xticks([])
    axes[row_idx, 0].set_yticks([])
    
    # Col 2: DWI + Ground Truth Overlay
    axes[row_idx, 1].imshow(dwi, cmap="gray")
    gt_mask = np.ma.masked_where(gt == 0, gt)
    axes[row_idx, 1].imshow(gt_mask, cmap="spring", alpha=0.5)
    axes[row_idx, 1].contour(gt, colors="lime", levels=[0.5], linewidths=2.0)
    if row_idx == 0:
        axes[row_idx, 1].set_title("Ground Truth")
    axes[row_idx, 1].axis("off")
    
    # Col 3: DWI + Mamba-UNet Prediction Overlay
    axes[row_idx, 2].imshow(dwi, cmap="gray")
    pred_mask = np.ma.masked_where(pred == 0, pred)
    axes[row_idx, 2].imshow(pred_mask, cmap="winter", alpha=0.5)
    axes[row_idx, 2].contour(pred, colors="cyan", levels=[0.5], linewidths=2.0)
    if row_idx == 0:
        axes[row_idx, 2].set_title("Mamba-UNet Prediction")
    axes[row_idx, 2].axis("off")

plt.suptitle("Qualitative Visualizations - Mamba-UNet (Selective SSM) Grid Results", y=0.98)
plt.tight_layout()
plt.savefig(MAMBA_GRID_PATH, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Mamba-UNet 3x3 Grid saved to: {MAMBA_GRID_PATH}")
