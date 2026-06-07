# =========================
# IMPORTS
# =========================
import os
import sys
import glob
import time
import random
import torch
import torch.nn as nn
import monai

from mamba_ssm import Mamba

from monai.utils import set_determinism
import warnings
warnings.filterwarnings("ignore")

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, NormalizeIntensityd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, EnsureTyped, SpatialPadd,
    RandShiftIntensityd, RandScaleIntensityd,
    RandAffined, RandGaussianNoised, RandGaussianSmoothd,
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.networks.blocks import UnetBasicBlock, UnetUpBlock

# =========================
# CONFIG & PATHS
# =========================
DATA_DIR = "/home/ronit.28010/Brain_Stroke/ISLES2022_Formatted"

# FIX 1: Isotropic 1mm spacing — critical for small embolic lesion detection
VOXEL_SPACING = (1.0, 1.0, 1.0)

ROI_SIZE    = (128, 128, 128)
BATCH_SIZE  = 2
MAX_EPOCHS  = 500
VAL_INTERVAL = 2
NUM_WORKERS  = 4

# Seed fixed for reproducibility
set_determinism(seed=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"MONAI Version: {monai.__version__}", flush=True)
print(f"Using device: {device}", flush=True)

if not os.path.exists(DATA_DIR):
    print(f"FATAL: '{DATA_DIR}' not found.")
    sys.exit()

# =========================
# DATA LOADING
# Your files are pre-merged 4D NIfTIs: sub-strokecaseXXXX.nii.gz
# Shape on disk: (H, W, D, 3) — DWI, ADC, FLAIR stacked on last axis.
# EnsureChannelFirstd(channel_dim=-1) unpacks to (3, H, W, D).
train_images = sorted(glob.glob(os.path.join(DATA_DIR, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(DATA_DIR, "labelsTr", "*.nii.gz")))

if len(train_images) == 0:
    print(f"FATAL: No images found in {DATA_DIR}/imagesTr/")
    sys.exit()

assert len(train_images) == len(train_labels), \
    f"Image/label count mismatch: {len(train_images)} images, {len(train_labels)} labels"

data = [{"image": i, "label": l} for i, l in zip(train_images, train_labels)]

# FIX 3: Random stratified split instead of sequential split
random.seed(42)
random.shuffle(data)
split = int(0.8 * len(data))
train_files, val_files = data[:split], data[split:]

print(f"Total: {len(data)} | Train: {len(train_files)} | Val: {len(val_files)}", flush=True)

# =========================
# TRANSFORMS
# =========================
# Image files are pre-merged 4D NIfTIs shaped (H, W, D, 3).
# channel_dim=-1 on "image" unpacks the 3 modalities → (3, H, W, D).
# Label files are 3D (H, W, D); channel_dim="no_channel" adds a dim → (1, H, W, D).

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image"], channel_dim=-1),            # (3, H, W, D)
    EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),  # (1, H, W, D)

    # FIX 1: Isotropic 1mm resampling — critical for small embolic lesion detection
    Spacingd(
        keys=["image", "label"],
        pixdim=VOXEL_SPACING,
        mode=("bilinear", "nearest"),
    ),
    Orientationd(keys=["image", "label"], axcodes="RAS"),

    # Per-modality normalization (nonzero brain mask only, per channel)
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

    # Pad volumes smaller than ROI
    SpatialPadd(keys=["image", "label"], spatial_size=ROI_SIZE),

    # FIX 4: pos=2, neg=1 — over-sample lesion crops for better small-lesion F1
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=ROI_SIZE,
        pos=2, neg=1,
        num_samples=2,
        image_key="image",
        image_threshold=0,
    ),

    # Augmentation
    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.2),
    RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.2),
    RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.2),
    RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
    RandAffined(
        keys=["image", "label"],
        prob=0.2,
        rotate_range=(0.1, 0.1, 0.1),
        scale_range=(0.1, 0.1, 0.1),
        mode=("bilinear", "nearest"),
        padding_mode="border",
    ),
    RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),
    RandGaussianSmoothd(keys=["image"], prob=0.15,
                        sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.15),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.15),

    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image"], channel_dim=-1),
    EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
    Spacingd(
        keys=["image", "label"],
        pixdim=VOXEL_SPACING,
        mode=("bilinear", "nearest"),
    ),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image", "label"]),
])

# =========================
# DATALOADERS
# =========================
train_ds = CacheDataset(train_files, train_transforms, cache_rate=1.0, num_workers=NUM_WORKERS)
val_ds   = CacheDataset(val_files,   val_transforms,   cache_rate=1.0, num_workers=NUM_WORKERS)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,   batch_size=1,          shuffle=False, num_workers=NUM_WORKERS)


# =========================
# MODEL: Tri-Planar OmniMamba-UNet  (v2 — fixed)
# =========================

class OmniMambaLayer(nn.Module):
    """
    Scans the 3D feature map along all 3 axes independently using separate
    Mamba SSMs, fuses with a local depthwise conv, and adds residual.

    FIX 5: Independent LayerNorm per scanning axis — each axis sees different
    sequence statistics (lengths differ), so sharing one norm was incorrect.
    """
    def __init__(self, dim, d_state=32, d_conv=4, expand=2):
        super().__init__()

        # FIX 5: One norm per axis instead of one shared norm
        self.norm_z = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)
        self.norm_x = nn.LayerNorm(dim)

        self.mamba_z = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_y = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_x = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

        # Local 3D depthwise conv to preserve precise stroke boundaries
        self.local_conv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)

        # Learnable per-axis fusion weights instead of simple sum
        self.fusion_weights = nn.Parameter(torch.ones(4) / 4.0)

    def forward(self, x):
        b, c, h, w, d = x.shape
        local_feat = self.local_conv(x)

        # Z-axis scan: sequence over (h*w*d)
        z_flat  = x.flatten(2).transpose(1, 2)             # (B, H*W*D, C)
        out_z   = self.mamba_z(self.norm_z(z_flat))
        out_z   = out_z.transpose(1, 2).view(b, c, h, w, d)

        # Y-axis scan: permute to (B, C, W, D, H), scan over (w*d*h)
        y_perm  = x.permute(0, 1, 3, 4, 2).contiguous()
        y_flat  = y_perm.flatten(2).transpose(1, 2)
        out_y   = self.mamba_y(self.norm_y(y_flat))
        out_y   = out_y.transpose(1, 2).view(b, c, w, d, h).permute(0, 1, 4, 2, 3)

        # X-axis scan: permute to (B, C, D, H, W), scan over (d*h*w)
        x_perm  = x.permute(0, 1, 4, 2, 3).contiguous()
        x_flat  = x_perm.flatten(2).transpose(1, 2)
        out_x   = self.mamba_x(self.norm_x(x_flat))
        out_x   = out_x.transpose(1, 2).view(b, c, d, h, w).permute(0, 1, 3, 4, 2)

        # Learnable weighted fusion + residual
        w = torch.softmax(self.fusion_weights, dim=0)
        return w[0]*out_z + w[1]*out_y + w[2]*out_x + w[3]*local_feat + x


class MambaUNet(nn.Module):
    """
    OmniMamba-UNet v2:
      - OmniMamba skip connections at encoder levels 2, 3, 4
      - Double OmniMamba bottleneck
      - Deep supervision from decoder levels d3 and d4  (FIX 6)
      - in_channels=3 for DWI + ADC + FLAIR
    """
    def __init__(self, in_channels=3, out_channels=2,
                 features=(32, 64, 128, 256, 512)):
        super().__init__()

        # --- Encoder ---
        self.enc1 = UnetBasicBlock(3, in_channels,   features[0], 3, 1, norm_name="instance")
        self.enc2 = UnetBasicBlock(3, features[0],   features[1], 3, 2, norm_name="instance")
        self.enc3 = UnetBasicBlock(3, features[1],   features[2], 3, 2, norm_name="instance")
        self.enc4 = UnetBasicBlock(3, features[2],   features[3], 3, 2, norm_name="instance")

        # --- OmniMamba on skip connections ---
        self.skip4_mamba = OmniMambaLayer(dim=features[3])
        self.skip3_mamba = OmniMambaLayer(dim=features[2])
        self.skip2_mamba = OmniMambaLayer(dim=features[1])

        # --- Bottleneck ---
        self.bottleneck_conv = UnetBasicBlock(3, features[3], features[4], 3, 2, norm_name="instance")
        self.mamba_block1    = OmniMambaLayer(dim=features[4])
        self.mamba_block2    = OmniMambaLayer(dim=features[4])

        # --- Decoder ---
        self.dec4 = UnetUpBlock(3, features[4], features[3], 3, 1, upsample_kernel_size=2, norm_name="instance")
        self.dec3 = UnetUpBlock(3, features[3], features[2], 3, 1, upsample_kernel_size=2, norm_name="instance")
        self.dec2 = UnetUpBlock(3, features[2], features[1], 3, 1, upsample_kernel_size=2, norm_name="instance")
        self.dec1 = UnetUpBlock(3, features[1], features[0], 3, 1, upsample_kernel_size=2, norm_name="instance")

        # --- Output heads ---
        self.final      = nn.Conv3d(features[0], out_channels, kernel_size=1)

        # FIX 6: Deep supervision heads for auxiliary losses during training
        self.aux_head3  = nn.Conv3d(features[2], out_channels, kernel_size=1)
        self.aux_head4  = nn.Conv3d(features[3], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck_conv(e4)
        b = self.mamba_block1(b)
        b = self.mamba_block2(b)

        # OmniMamba-filtered skip connections
        e4_f = self.skip4_mamba(e4)
        e3_f = self.skip3_mamba(e3)
        e2_f = self.skip2_mamba(e2)

        # Decoder
        d4 = self.dec4(b,  e4_f)
        d3 = self.dec3(d4, e3_f)
        d2 = self.dec2(d3, e2_f)
        d1 = self.dec1(d2, e1)

        main_out = self.final(d1)

        if self.training:
            # Deep supervision: return auxiliary predictions at two scales
            aux3 = self.aux_head3(d3)
            aux4 = self.aux_head4(d4)
            return main_out, aux3, aux4

        return main_out


# =========================
# TEST-TIME AUGMENTATION (FIX 7)
# Flip along all 8 combinations of axes, average predictions.
# Adds ~+1-2% Dice at zero training cost.
# =========================
def tta_predict(model, inputs, roi_size, sw_batch_size=4):
    """8-flip TTA for sliding window inference."""
    flip_combos = [
        [],
        [2], [3], [4],
        [2, 3], [2, 4], [3, 4],
        [2, 3, 4],
    ]
    preds = []
    for axes in flip_combos:
        x = inputs
        for ax in axes:
            x = torch.flip(x, dims=[ax])
        pred = sliding_window_inference(x, roi_size, sw_batch_size, model)
        for ax in reversed(axes):
            pred = torch.flip(pred, dims=[ax])
        preds.append(torch.softmax(pred, dim=1))
    return torch.stack(preds).mean(0)


# =========================
# TRAINING SETUP
# =========================
model          = MambaUNet().to(device)
loss_function  = DiceFocalLoss(to_onehot_y=True, softmax=True,
                               squared_pred=True, batch=True)
optimizer      = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# FIX 8: Cosine annealing LR schedule — flat LR for 400 epochs kills performance
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MAX_EPOCHS, eta_min=1e-6
)

dice_metric = DiceMetric(include_background=False, reduction="mean")

best_metric    = -1
best_metric_ep = -1
start_time     = time.time()

# Deep supervision loss weights: main=1.0, aux3=0.4, aux4=0.2
DS_WEIGHTS = (1.0, 0.4, 0.2)
DS_TOTAL   = sum(DS_WEIGHTS)

print("Starting OmniMamba-UNet v2 Training...", flush=True)
print(f"Improvements: isotropic 1mm spacing | random split | per-axis norms | "
      f"deep supervision | cosine LR | pos=2/neg=1 crops | 8-flip TTA at val", flush=True)

# =========================
# TRAINING LOOP
# =========================
for epoch in range(MAX_EPOCHS):
    model.train()
    epoch_loss = 0.0

    for batch in train_loader:
        inputs = batch["image"].to(device)   # (B, 3, H, W, D)
        labels = batch["label"].to(device)   # (B, 1, H, W, D)

        optimizer.zero_grad()

        # FIX 6: Unpack deep supervision outputs
        main_out, aux3, aux4 = model(inputs)

        # aux3 is at 1/4 resolution, aux4 is at 1/8 resolution relative to input.
        # Downsample labels to match each aux head's spatial size before loss.
        labels_aux3 = torch.nn.functional.interpolate(
            labels.float(), size=aux3.shape[2:], mode="nearest").long()
        labels_aux4 = torch.nn.functional.interpolate(
            labels.float(), size=aux4.shape[2:], mode="nearest").long()

        # Compute weighted deep supervision loss
        loss = (
            DS_WEIGHTS[0] * loss_function(main_out, labels) +
            DS_WEIGHTS[1] * loss_function(aux3, labels_aux3) +
            DS_WEIGHTS[2] * loss_function(aux4, labels_aux4)
        ) / DS_TOTAL

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()  # FIX 8: update LR after each epoch
    epoch_loss /= len(train_loader)
    current_lr = scheduler.get_last_lr()[0]

    print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] Loss: {epoch_loss:.4f}  LR: {current_lr:.2e}", flush=True)

    # =========================
    # VALIDATION
    # =========================
    if (epoch + 1) % VAL_INTERVAL == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs = val_data["image"].to(device)
                val_labels = val_data["label"].to(device)

                # FIX 7: Use TTA during validation
                val_probs   = tta_predict(model, val_inputs, ROI_SIZE, sw_batch_size=4)
                val_outputs = [o.argmax(dim=0, keepdim=True)
                               for o in decollate_batch(val_probs)]
                val_labels  = decollate_batch(val_labels)

                dice_metric(val_outputs, val_labels)

        metric = dice_metric.aggregate().item()
        dice_metric.reset()

        if metric > best_metric:
            best_metric    = metric
            best_metric_ep = epoch + 1
            torch.save(model.state_dict(), "best_mamba_model_v2.pth")
            print(f"  *** New best Dice: {metric:.4f}  (epoch {best_metric_ep})", flush=True)
        else:
            print(f"  Val Dice: {metric:.4f}  [best: {best_metric:.4f} @ ep {best_metric_ep}]",
                  flush=True)

elapsed = (time.time() - start_time) / 60
print(f"\nTraining complete in {elapsed:.1f} min", flush=True)
print(f"Best Val Dice: {best_metric:.4f} at epoch {best_metric_ep}", flush=True)