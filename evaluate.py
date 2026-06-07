import torch
import numpy as np
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric, ConfusionMatrixMetric
from monai.data import decollate_batch
from monai.networks.nets import UNETR

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, NormalizeIntensityd, EnsureTyped
)
from monai.data import CacheDataset, DataLoader

DATA_DIR = "ISLES2022_Formatted"
ROI_SIZE = (96, 96, 96)

val_images = sorted(glob.glob(f"{DATA_DIR}/imagesTr/*.nii.gz"))
val_labels = sorted(glob.glob(f"{DATA_DIR}/labelsTr/*.nii.gz"))

val_files = [{"image": img, "label": lbl} for img, lbl in zip(val_images, val_labels)]

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image"], channel_dim=-1),
    EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0),
             mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image", "label"]),
])

val_ds = CacheDataset(val_files, val_transforms, cache_rate=1.0, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

# --- 1. SETUP MODEL (Architecture must match Training) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROI_SIZE = (96, 96, 96)

model = UNETR(
    in_channels=3,
    out_channels=2,
    img_size=ROI_SIZE,
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    proj_type="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)

# Load Weights
print("🔄 Loading Best Model Weights...")
model.load_state_dict(
    torch.load("best_metric_model.pth", map_location=torch.device("cpu"))
)

model.eval()
print("✅ Model Loaded Successfully!")

# --- 2. DEFINE METRICS ---
# Dice: Overlap accuracy
dice_metric = DiceMetric(include_background=False, reduction="none")
# HD95: Boundary error (95th percentile)
hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")
# Confusion Matrix for Precision/Recall
conf_metric = ConfusionMatrixMetric(include_background=False, metric_name=["precision", "recall"], reduction="none")

# Store results for every single patient
patient_results = []

print("📊 Starting Detailed Evaluation on Validation Set...")

# --- 3. EVALUATION LOOP ---
with torch.no_grad():
    for i, batch in enumerate(val_loader):
        val_inputs, val_labels = batch["image"].to(device), batch["label"].to(device)
        
        # Inference (Sliding Window)
        val_outputs = sliding_window_inference(val_inputs, ROI_SIZE, 4, model)
        
        # Convert to Discrete (0 or 1)
        val_outputs = [i.argmax(dim=0, keepdim=True) for i in decollate_batch(val_outputs)]
        val_labels = [i for i in decollate_batch(val_labels)]
        
        # --- Compute Metrics ---
        
        # 1. Dice
        dice_metric(y_pred=val_outputs, y=val_labels)
        dice_score = dice_metric.aggregate().item()
        
        # 2. Hausdorff Distance (Handle cases where prediction is empty)
        try:
            hd95_metric(y_pred=val_outputs, y=val_labels)
            hd95_score = hd95_metric.aggregate().item()
        except:
            # Agar model ne kuch predict hi nahi kiya, ya label empty hai
            hd95_score = np.nan 

        # 3. Precision & Recall
        conf_metric(y_pred=val_outputs, y=val_labels)
        conf_matrix = conf_metric.aggregate() # Returns [Precision, Recall]
        precision_score = conf_matrix[0].item()
        recall_score = conf_matrix[1].item()
        
        # Store in list
        patient_results.append({
            "Patient_ID": i + 1,
            "Dice": dice_score,
            "Hausdorff95": hd95_score,
            "Precision": precision_score,
            "Recall": recall_score
        })
        
        # Reset metrics for next batch
        dice_metric.reset()
        hd95_metric.reset()
        conf_metric.reset()
        
        if (i+1) % 5 == 0:
            print(f"   Processed {i+1} patients...")

# --- 4. VISUALIZATION & REPORT ---
df = pd.DataFrame(patient_results)

# Clean NaNs (Just in case HD95 failed for some)
df_clean = df.dropna()

# A. Print Summary Table
print("\n" + "="*40)
print(" 📝 FINAL MODEL PERFORMANCE REPORT ")
print("="*40)
summary = df_clean[["Dice", "Hausdorff95", "Precision", "Recall"]].describe().loc[['mean', 'std', 'min', 'max']]
print(summary)
print("="*40)

# B. Generate Box Plots (Standard for Research Papers)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Accuracy Metrics (Higher is Better)
sns.boxplot(data=df_clean[["Dice", "Precision", "Recall"]], ax=axes[0], palette="viridis")
axes[0].set_title("Segmentation Accuracy (Higher is Better ⬆️)")
axes[0].set_ylabel("Score (0.0 - 1.0)")
axes[0].grid(True, alpha=0.3)

# Plot 2: Error Metrics (Lower is Better)
sns.boxplot(data=df_clean[["Hausdorff95"]], ax=axes[1], color="salmon")
axes[1].set_title("Boundary Error (Lower is Better ⬇️)")
axes[1].set_ylabel("Distance (Pixels)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nFinal Mean Dice Score: {df_clean['Dice'].mean():.4f}")