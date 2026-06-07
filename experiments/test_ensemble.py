import os
import glob
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from monai.networks.blocks import UnetBasicBlock, UnetUpBlock
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, NormalizeIntensityd, EnsureTyped
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

# =========================
# 1. PASTE THE CLASSES DIRECTLY (No Imports from train_mamba)
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
        z_flat = x.flatten(2).transpose(1, 2)
        out_z = self.mamba_z(self.norm(z_flat)).transpose(1, 2).view(b, c, h, w, d)
        y_perm = x.permute(0, 1, 3, 4, 2).contiguous()
        y_flat = y_perm.flatten(2).transpose(1, 2)
        out_y = self.mamba_y(self.norm(y_flat)).transpose(1, 2).view(b, c, w, d, h).permute(0, 1, 4, 2, 3)
        x_perm = x.permute(0, 1, 4, 2, 3).contiguous()
        x_flat = x_perm.flatten(2).transpose(1, 2)
        out_x = self.mamba_x(self.norm(x_flat)).transpose(1, 2).view(b, c, d, h, w).permute(0, 1, 3, 4, 2)
        return out_z + out_y + out_x + local_feat + x

class MambaUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, features=(32, 64, 128, 256, 512)):
        super().__init__()
        self.enc1 = UnetBasicBlock(spatial_dims=3, in_channels=in_channels, out_channels=features[0], kernel_size=3, stride=1, norm_name="instance")
        self.enc2 = UnetBasicBlock(spatial_dims=3, in_channels=features[0], out_channels=features[1], kernel_size=3, stride=2, norm_name="instance")
        self.enc3 = UnetBasicBlock(spatial_dims=3, in_channels=features[1], out_channels=features[2], kernel_size=3, stride=2, norm_name="instance")
        self.enc4 = UnetBasicBlock(spatial_dims=3, in_channels=features[2], out_channels=features[3], kernel_size=3, stride=2, norm_name="instance")

        self.skip4_mamba = OmniMambaLayer(dim=features[3]) 
        self.skip3_mamba = OmniMambaLayer(dim=features[2])
        self.skip2_mamba = OmniMambaLayer(dim=features[1]) 

        self.bottleneck_conv = UnetBasicBlock(spatial_dims=3, in_channels=features[3], out_channels=features[4], kernel_size=3, stride=2, norm_name="instance")
        self.mamba_block1 = OmniMambaLayer(dim=features[4])
        self.mamba_block2 = OmniMambaLayer(dim=features[4]) 

        self.dec4 = UnetUpBlock(spatial_dims=3, in_channels=features[4], out_channels=features[3], kernel_size=3, stride=1, upsample_kernel_size=2, norm_name="instance")
        self.dec3 = UnetUpBlock(spatial_dims=3, in_channels=features[3], out_channels=features[2], kernel_size=3, stride=1, upsample_kernel_size=2, norm_name="instance")
        self.dec2 = UnetUpBlock(spatial_dims=3, in_channels=features[2], out_channels=features[1], kernel_size=3, stride=1, upsample_kernel_size=2, norm_name="instance")
        self.dec1 = UnetUpBlock(spatial_dims=3, in_channels=features[1], out_channels=features[0], kernel_size=3, stride=1, upsample_kernel_size=2, norm_name="instance")

        self.final = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck_conv(e4)
        b = self.mamba_block1(b)
        b = self.mamba_block2(b) 

        e4_filtered = self.skip4_mamba(e4)
        e3_filtered = self.skip3_mamba(e3)
        e2_filtered = self.skip2_mamba(e2)

        d4 = self.dec4(b, e4_filtered)       
        d3 = self.dec3(d4, e3_filtered)      
        d2 = self.dec2(d3, e2_filtered)               
        d1 = self.dec1(d2, e1)

        return self.final(d1)

# =========================
# CONFIG
# =========================
DATA_DIR = "/home/ronit.28010/Brain_Stroke/ISLES2022_Formatted"
ROI_SIZE = (128, 128, 128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD ALL 5 MODELS
# =========================
print("🧠 Loading 5-Fold Master Ensemble...")
models = []
for i in range(5):
    model = MambaUNet().to(device)
    weight_path = f"best_mamba_model_fold_{i}.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    models.append(model)
    print(f"✅ Model Fold {i} Loaded!")

# =========================
# TEST DATA PREP
# =========================
# Note: Ideal case mein aapke paas ek alag 'Test Set' hona chahiye. 
# Abhi test karne ke liye hum validation dataset wala hi path de rahe hain.
test_images = sorted(glob.glob(os.path.join(DATA_DIR, "imagesTr", "*.nii.gz")))[-20:] # Last 20 images for quick test
test_labels = sorted(glob.glob(os.path.join(DATA_DIR, "labelsTr", "*.nii.gz")))[-20:]

test_files = [{"image": i, "label": l} for i, l in zip(test_images, test_labels)]

test_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image"], channel_dim=-1),
    EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image", "label"]),
])

test_ds = Dataset(data=test_files, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

dice_metric = DiceMetric(include_background=False, reduction="mean")

# =========================
# ENSEMBLE INFERENCE LOOP
# =========================
print(f"🚀 Starting Ensemble Inference on {len(test_files)} images...")

with torch.no_grad():
    for i, batch_data in enumerate(test_loader):
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        
        ensemble_probs = 0.0
        
        # Har image ko 5 models se pass karein
        for model in models:
            # Overlap 0.625 zaroori hai SOTA smooth boundaries ke liye
            logits = sliding_window_inference(inputs, ROI_SIZE, 4, model, overlap=0.625)
            # Logits ko Probabilities (0 to 1) mein convert karein
            probs = torch.softmax(logits, dim=1) 
            ensemble_probs += probs
            
        # 5 Models ka average
        ensemble_probs = ensemble_probs / 5.0
        
        # Jis class ki probability sabse zyada hai, use final prediction maanein
        final_prediction = torch.argmax(ensemble_probs, dim=1, keepdim=True)
        
        # Metrics calculate karein
        val_labels = decollate_batch(labels)
        val_outputs = [final_prediction.squeeze(0)] # Decollate
        
        dice_metric(y_pred=val_outputs, y=val_labels)
        print(f"Scan {i+1}/{len(test_files)} processed.")

final_dice = dice_metric.aggregate().item()
print("="*40)
print(f"🏆 FINAL ENSEMBLE DICE SCORE: {final_dice:.4f}")
print("="*40)