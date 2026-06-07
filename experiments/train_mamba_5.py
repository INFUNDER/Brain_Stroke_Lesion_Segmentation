# =========================
# IMPORTS
# =========================
import os
import sys
import glob
import time
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
    RandFlipd, RandRotate90d, EnsureTyped, SpatialPadd
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.networks.blocks import UnetBasicBlock, UnetUpBlock

# =========================
# CONFIG & PATHS
# =========================
DATA_DIR = "/home/ronit.28010/Brain_Stroke/ISLES2022_Formatted" 

ROI_SIZE = (96, 96, 96)
BATCH_SIZE = 2
MAX_EPOCHS = 200
VAL_INTERVAL = 2
NUM_WORKERS = 4

# MATCHED WITH UNETR: Seed fixed for 100% fair comparison
set_determinism(seed=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"MONAI Version: {monai.__version__}", flush=True)
print(f"Using device: {device}", flush=True)

# Safety check for the dataset
if not os.path.exists(DATA_DIR):
    print(f"🚨 FATAL ERROR: Mujhe '{DATA_DIR}' naam ka folder nahi mila!")
    sys.exit()

# =========================
# DATASET
# =========================
train_images = sorted(glob.glob(os.path.join(DATA_DIR, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(DATA_DIR, "labelsTr", "*.nii.gz")))

if len(train_images) == 0:
    print(f"🚨 FATAL ERROR: Folder toh mil gaya, par usme images nahi hain!")
    sys.exit()

assert len(train_images) == len(train_labels), "Images and labels count mismatch"

data = [{"image": i, "label": l} for i, l in zip(train_images, train_labels)]
split = int(0.8 * len(data))
train_files, val_files = data[:split], data[split:]

print(f"🛑 Total Images: {len(data)} | Training: {len(train_files)} | Validation: {len(val_files)}", flush=True)

# =========================
# TRANSFORMS (MATCHED WITH UNETR)
# =========================
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image"], channel_dim=-1),
    EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0),
             mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    SpatialPadd(keys=["image", "label"], spatial_size=ROI_SIZE),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=ROI_SIZE,
        pos=1, neg=1, num_samples=2,
        image_key="image",       # Added to match UNETR
        image_threshold=0,       # Added to match UNETR
    ),
    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.1),
    RandRotate90d(keys=["image", "label"], prob=0.1, max_k=3), # max_k added to match UNETR
    EnsureTyped(keys=["image", "label"]),
])

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

# =========================
# DATALOADERS
# =========================
train_ds = CacheDataset(train_files, train_transforms, cache_rate=1.0, num_workers=NUM_WORKERS)
val_ds = CacheDataset(val_files, val_transforms, cache_rate=1.0, num_workers=NUM_WORKERS)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)



# =========================
# THE BEAST: Tri-Planar OmniMamba-UNet (For H100)
# =========================
class OmniMambaLayer(nn.Module):
    def __init__(self, dim, d_state=32, d_conv=4, expand=2): # High d_state for better memory tracking
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        
        # NOVELTY: 3 Separate Mamba Engines for 3D Space Scanning
        self.mamba_z = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand) # Depth-wise
        self.mamba_y = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand) # Height-wise
        self.mamba_x = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand) # Width-wise
        
        # Local 3D Conv to preserve exact stroke boundaries
        self.local_conv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        b, c, h, w, d = x.shape
        local_feat = self.local_conv(x)
        
        # 1. Z-Axis Scanning (Standard H, W, D)
        z_flat = x.flatten(2).transpose(1, 2)
        z_norm = self.norm(z_flat)
        out_z = self.mamba_z(z_norm).transpose(1, 2).view(b, c, h, w, d)
        
        # 2. Y-Axis Scanning (Permuted to W, D, H)
        y_perm = x.permute(0, 1, 3, 4, 2).contiguous()
        y_flat = y_perm.flatten(2).transpose(1, 2)
        y_norm = self.norm(y_flat)
        out_y = self.mamba_y(y_norm).transpose(1, 2).view(b, c, w, d, h).permute(0, 1, 4, 2, 3)
        
        # 3. X-Axis Scanning (Permuted to D, H, W)
        x_perm = x.permute(0, 1, 4, 2, 3).contiguous()
        x_flat = x_perm.flatten(2).transpose(1, 2)
        x_norm = self.norm(x_flat)
        out_x = self.mamba_x(x_norm).transpose(1, 2).view(b, c, d, h, w).permute(0, 1, 3, 4, 2)
        
        # Fusion of all 3 spatial planes + local features + residual
        return out_z + out_y + out_x + local_feat + x

class MambaUNet(nn.Module):
    # H100 POWER: Doubled the feature channels to match UNETR's capacity
    def __init__(self, in_channels=3, out_channels=2, features=(32, 64, 128, 256, 512)):
        super().__init__()

        # Encoder Layers
        self.enc1 = UnetBasicBlock(spatial_dims=3, in_channels=in_channels, out_channels=features[0], kernel_size=3, stride=1, norm_name="instance")
        self.enc2 = UnetBasicBlock(spatial_dims=3, in_channels=features[0], out_channels=features[1], kernel_size=3, stride=2, norm_name="instance")
        self.enc3 = UnetBasicBlock(spatial_dims=3, in_channels=features[1], out_channels=features[2], kernel_size=3, stride=2, norm_name="instance")
        self.enc4 = UnetBasicBlock(spatial_dims=3, in_channels=features[2], out_channels=features[3], kernel_size=3, stride=2, norm_name="instance")

        # NOVELTY: OmniMamba Skip Connections (Smart 3D Attention Filters)
        self.skip4_mamba = OmniMambaLayer(dim=features[3]) 
        self.skip3_mamba = OmniMambaLayer(dim=features[2])
        self.skip2_mamba = OmniMambaLayer(dim=features[1]) # Added deep skip filtering!

        # Bottleneck + Mega OmniMamba Layer
        self.bottleneck_conv = UnetBasicBlock(spatial_dims=3, in_channels=features[3], out_channels=features[4], kernel_size=3, stride=2, norm_name="instance")
        self.mamba_block1 = OmniMambaLayer(dim=features[4])
        self.mamba_block2 = OmniMambaLayer(dim=features[4]) # Double Bottleneck Power

        # Decoder Layers
        self.dec4 = UnetUpBlock(spatial_dims=3, in_channels=features[4], out_channels=features[3], kernel_size=3, stride=1, upsample_kernel_size=2, norm_name="instance")
        self.dec3 = UnetUpBlock(spatial_dims=3, in_channels=features[3], out_channels=features[2], kernel_size=3, stride=1, upsample_kernel_size=2, norm_name="instance")
        self.dec2 = UnetUpBlock(spatial_dims=3, in_channels=features[2], out_channels=features[1], kernel_size=3, stride=1, upsample_kernel_size=2, norm_name="instance")
        self.dec1 = UnetUpBlock(spatial_dims=3, in_channels=features[1], out_channels=features[0], kernel_size=3, stride=1, upsample_kernel_size=2, norm_name="instance")

        # Final Convolution
        self.final = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck (Double OmniMamba)
        b = self.bottleneck_conv(e4)
        b = self.mamba_block1(b)
        b = self.mamba_block2(b) 

        # OmniMamba Filtered Skip Connections
        e4_filtered = self.skip4_mamba(e4)
        e3_filtered = self.skip3_mamba(e3)
        e2_filtered = self.skip2_mamba(e2)

        # Decoder with Smart Skip Connections
        d4 = self.dec4(b, e4_filtered)       
        d3 = self.dec3(d4, e3_filtered)      
        d2 = self.dec2(d3, e2_filtered)               
        d1 = self.dec1(d2, e1)

        return self.final(d1)


# =========================
# TRAINING LOOP
# =========================
model = MambaUNet().to(device)
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
dice_metric = DiceMetric(include_background=False, reduction="mean")

best_metric = -1
start_time = time.time()

print("🐍 Starting Mamba Training...", flush=True)

for epoch in range(MAX_EPOCHS):
    model.train()
    epoch_loss = 0

    for batch in train_loader:
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] Loss: {epoch_loss:.4f}", flush=True)

    if (epoch + 1) % VAL_INTERVAL == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs = val_data["image"].to(device)
                val_labels = val_data["label"].to(device)
                
                val_outputs = sliding_window_inference(val_inputs, ROI_SIZE, 4, model)
                val_outputs = [o.argmax(dim=0, keepdim=True) for o in decollate_batch(val_outputs)]
                val_labels = decollate_batch(val_labels) 
                
                dice_metric(val_outputs, val_labels)

            metric = dice_metric.aggregate().item()
            dice_metric.reset()

            if metric > best_metric:
                best_metric = metric
                torch.save(model.state_dict(), "best_mamba_model.pth")
                print(f"⭐ New best Dice: {metric:.4f}", flush=True)
            else:
                print(f"Val Dice: {metric:.4f}", flush=True)

print(f"Training completed in {(time.time()-start_time)/60:.2f} minutes", flush=True)
