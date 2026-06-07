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
import torch.nn.functional as F
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
from monai.losses import DiceFocalLoss
# =========================
# CONFIG & PATHS
# =========================
DATA_DIR = "/home/ronit.28010/Brain_Stroke/ISLES2022_Formatted" 

ROI_SIZE = (128, 128, 128)
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
# 1. TRI-PLANAR OMNIMAMBA (The Global Visionary)
# =========================
class OmniMambaLayer(nn.Module):
    def __init__(self, dim, d_state=32, d_conv=4, expand=2): 
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        
        self.mamba_z = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand) 
        self.mamba_y = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand) 
        self.mamba_x = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand) 
        
        self.local_conv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        b, c, h, w, d = x.shape
        local_feat = self.local_conv(x)
        
        # Z, Y, X axis scans
        z_flat = x.flatten(2).transpose(1, 2)
        out_z = self.mamba_z(self.norm(z_flat)).transpose(1, 2).view(b, c, h, w, d)
        
        y_perm = x.permute(0, 1, 3, 4, 2).contiguous()
        y_flat = y_perm.flatten(2).transpose(1, 2)
        out_y = self.mamba_y(self.norm(y_flat)).transpose(1, 2).view(b, c, w, d, h).permute(0, 1, 4, 2, 3)
        
        x_perm = x.permute(0, 1, 4, 2, 3).contiguous()
        x_flat = x_perm.flatten(2).transpose(1, 2)
        out_x = self.mamba_x(self.norm(x_flat)).transpose(1, 2).view(b, c, d, h, w).permute(0, 1, 3, 4, 2)
        
        return out_z + out_y + out_x + local_feat + x

# =========================
# 2. NOVELTY: 3D ATTENTION GATES (The Sharp Shooter)
# =========================
class AttentionGate3D(nn.Module):
    def __init__(self, in_channels_skip, in_channels_gate, inter_channels):
        super().__init__()
        # W_g processes the gating signal (from Decoder)
        self.W_g = nn.Sequential(
            nn.Conv3d(in_channels_gate, inter_channels, kernel_size=1, stride=1),
            nn.InstanceNorm3d(inter_channels)
        )
        # W_x processes the skip connection (from Encoder), stride=2 aligns spatial dims
        self.W_x = nn.Sequential(
            nn.Conv3d(in_channels_skip, inter_channels, kernel_size=1, stride=2), 
            nn.InstanceNorm3d(inter_channels)
        )
        # psi generates the attention coefficients ($\alpha$)
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, stride=1),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g_conv = self.W_g(g)
        x_conv = self.W_x(x)
        
        # Align spatial dimensions dynamically
        if g_conv.shape[2:] != x_conv.shape[2:]:
            x_conv = F.interpolate(x_conv, size=g_conv.shape[2:], mode='trilinear', align_corners=False)
            
        out = self.relu(g_conv + x_conv)
        alpha = self.psi(out) # Attention Map (Values between 0 and 1)
        
        # Upsample attention map to original skip connection size
        alpha = F.interpolate(alpha, size=x.shape[2:], mode='trilinear', align_corners=False)
        
        # Multiply skip connection by attention map (suppresses background)
        return x * alpha

# =========================
# 3. ATTENTION-GUIDED OMNIMAMBA-UNET
# =========================
class MambaUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, features=(32, 64, 128, 256, 512)):
        super().__init__()

        # ENCODER
        self.enc1 = UnetBasicBlock(spatial_dims=3, in_channels=in_channels, out_channels=features[0], kernel_size=3, stride=1, norm_name="instance")
        self.enc2 = UnetBasicBlock(spatial_dims=3, in_channels=features[0], out_channels=features[1], kernel_size=3, stride=2, norm_name="instance")
        self.enc3 = UnetBasicBlock(spatial_dims=3, in_channels=features[1], out_channels=features[2], kernel_size=3, stride=2, norm_name="instance")
        self.enc4 = UnetBasicBlock(spatial_dims=3, in_channels=features[2], out_channels=features[3], kernel_size=3, stride=2, norm_name="instance")

        # BOTTLENECK (Double OmniMamba Power)
        self.bottleneck_conv = UnetBasicBlock(spatial_dims=3, in_channels=features[3], out_channels=features[4], kernel_size=3, stride=2, norm_name="instance")
        self.mamba_block1 = OmniMambaLayer(dim=features[4])
        self.mamba_block2 = OmniMambaLayer(dim=features[4])

        # ATTENTION GATES (Filters Skip Connections)
        self.ag4 = AttentionGate3D(in_channels_skip=features[3], in_channels_gate=features[4], inter_channels=features[2])
        self.ag3 = AttentionGate3D(in_channels_skip=features[2], in_channels_gate=features[3], inter_channels=features[1])
        self.ag2 = AttentionGate3D(in_channels_skip=features[1], in_channels_gate=features[2], inter_channels=features[0])

        # DECODER
        self.dec4 = UnetUpBlock(spatial_dims=3, in_channels=features[4], out_channels=features[3], kernel_size=3, stride=1, upsample_kernel_size=2, norm_name="instance")
        self.dec3 = UnetUpBlock(spatial_dims=3, in_channels=features[3], out_channels=features[2], kernel_size=3, stride=1, upsample_kernel_size=2, norm_name="instance")
        self.dec2 = UnetUpBlock(spatial_dims=3, in_channels=features[2], out_channels=features[1], kernel_size=3, stride=1, upsample_kernel_size=2, norm_name="instance")
        self.dec1 = UnetUpBlock(spatial_dims=3, in_channels=features[1], out_channels=features[0], kernel_size=3, stride=1, upsample_kernel_size=2, norm_name="instance")

        self.final = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # 1. Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # 2. Bottleneck
        b = self.bottleneck_conv(e4)
        b = self.mamba_block1(b)
        b = self.mamba_block2(b) 

        # 3. Attention + Decode
        e4_attn = self.ag4(e4, b)          # Gate: Bottleneck (b), Skip: e4
        d4 = self.dec4(b, e4_attn)       

        e3_attn = self.ag3(e3, d4)         # Gate: d4, Skip: e3
        d3 = self.dec3(d4, e3_attn)      

        e2_attn = self.ag2(e2, d3)         # Gate: d3, Skip: e2
        d2 = self.dec2(d3, e2_attn)               

        d1 = self.dec1(d2, e1)             # Highest res block usually doesn't need attention

        return self.final(d1)

# =========================
# TRAINING LOOP
# =========================
model = MambaUNet().to(device)
loss_function = DiceFocalLoss(to_onehot_y=True, softmax=True, squared_pred=True)
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
                
                val_outputs = sliding_window_inference(val_inputs, ROI_SIZE, 4, model, overlap=0.625)
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