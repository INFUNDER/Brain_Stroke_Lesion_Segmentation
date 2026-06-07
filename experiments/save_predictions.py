import os
import glob
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from monai.networks.blocks import UnetBasicBlock, UnetUpBlock
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, NormalizeIntensityd, EnsureTyped, SaveImage
)
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference
import monai

# =========================
# 1. CLASSES (No Imports)
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
# 2. CONFIG & LOAD MODELS
# =========================
DATA_DIR = "/home/ronit.28010/Brain_Stroke/ISLES2022_Formatted"
OUTPUT_DIR = "/home/ronit.28010/Brain_Stroke/ensemble_outputs" # Folder for NIfTI files
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROI_SIZE = (128, 128, 128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🧠 Loading 5-Fold Master Ensemble...")
models = []
for i in range(5):
    model = MambaUNet().to(device)
    weight_path = f"best_mamba_model_fold_{i}.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()
    models.append(model)
print("✅ All 5 Models Loaded Successfully!")

# =========================
# 3. TEST DATA PREP
# =========================
test_images = sorted(glob.glob(os.path.join(DATA_DIR, "imagesTr", "*.nii.gz")))[-20:] 
test_files = [{"image": i} for i in test_images] # No labels needed for saving predictions

test_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"], channel_dim=-1),
    Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
    Orientationd(keys=["image"], axcodes="RAS"),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image"]),
])

test_ds = Dataset(data=test_files, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

# MONAI's magical saver tool
saver = SaveImage(output_dir=OUTPUT_DIR, output_ext=".nii.gz", output_postfix="mamba_pred", separate_folder=False, resample=False)
# =========================
# 4. INFERENCE & SAVE LOOP (UPDATED)
# =========================
print(f"🚀 Generating 3D Predictions and saving to: {OUTPUT_DIR}")

with torch.no_grad():
    for i, batch_data in enumerate(test_loader):
        inputs = batch_data["image"].to(device)
        
        ensemble_probs = 0.0
        for model in models:
            logits = sliding_window_inference(inputs, ROI_SIZE, 4, model, overlap=0.625)
            ensemble_probs += torch.softmax(logits, dim=1)
            
        ensemble_probs = ensemble_probs / 5.0
        final_prediction = torch.argmax(ensemble_probs, dim=1, keepdim=True)
        
        # THE FIX: Safely un-batch the tensors so Affine matrix becomes 2D (4x4) again
        for val_pred, val_input in zip(final_prediction, inputs):
            if isinstance(val_input, monai.data.MetaTensor):
                # Attach the clean, unbatched metadata to our prediction
                pred_meta_tensor = monai.data.MetaTensor(val_pred, affine=val_input.affine, meta=val_input.meta)
            else:
                pred_meta_tensor = val_pred
                
            # Save it! (Saver extracts metadata automatically from MetaTensor)
            saver(pred_meta_tensor)
            
        print(f"✅ Saved Scan {i+1}/{len(test_files)}")

print("\n🎉 All 3D predictions successfully saved! Ready for ITK-SNAP visualization.")