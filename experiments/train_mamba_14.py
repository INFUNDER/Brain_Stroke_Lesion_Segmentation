# =============================================================================
# MAMBA-UNET: FULLY AUTOMATED END-TO-END PIPELINE (PRO VERSION)
# H100 80GB | 5-Fold CV | Gradient Accumulation | FP32 NaN-Fixed Eval
# =============================================================================

import os
import sys
import glob
import time
import json
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import monai

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datetime import datetime
from mamba_ssm import Mamba
from sklearn.model_selection import KFold

from monai.utils import set_determinism
import warnings
warnings.filterwarnings("ignore")

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, NormalizeIntensityd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, EnsureTyped, SpatialPadd,
    RandGaussianNoised, RandScaleIntensityd, RandShiftIntensityd,
    RandGaussianSmoothd, AsDiscrete
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.networks.blocks import UnetBasicBlock, UnetUpBlock
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel

# =============================================================================
# CONFIG — THE PERFORMANCE BOOSTERS
# =============================================================================
DATA_DIR      = "/home/ronit.28010/Brain_Stroke/ISLES2022_Formatted"
OUTPUT_DIR    = os.path.join(os.environ.get("PBS_O_WORKDIR", "."), "mamba_results")

ROI_SIZE           = (128, 128, 128) # 🔥 Upgraded from 96 for better context
BATCH_SIZE         = 1
ACCUMULATION_STEPS = 4               # 🔥 Smoothes training (Effective Batch = 4)
TTA_OVERLAP        = 0.625           # 🔥 Better edge detection during testing

MAX_EPOCHS    = 200
VAL_INTERVAL  = 2
NUM_WORKERS   = 4   
WARMUP_EPOCHS = 10
EMA_DECAY     = 0.999
N_FOLDS       = 5
TEST_FRACTION = 0.10       
BOUNDARY_W    = 0.2        
SEED          = 42

# =============================================================================
# DIRECTORY SETUP & LOGGING
# =============================================================================
LOG_DIR   = os.path.join(OUTPUT_DIR, "logs")
CKPT_DIR  = os.path.join(OUTPUT_DIR, "checkpoints")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")

for d in [OUTPUT_DIR, LOG_DIR, CKPT_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, f"run_{run_id}.log")),
    ]
)
log = logging.getLogger()

def banner(msg):
    log.info("=" * 65)
    log.info(f"  {msg}")
    log.info("=" * 65)

set_determinism(seed=SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# DATA SPLIT
# =============================================================================
train_images = sorted(glob.glob(os.path.join(DATA_DIR, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(DATA_DIR, "labelsTr", "*.nii.gz")))
all_data = [{"image": i, "label": l} for i, l in zip(train_images, train_labels)]

rng      = np.random.default_rng(seed=SEED)
perm     = rng.permutation(len(all_data))
n_test   = max(1, int(TEST_FRACTION * len(all_data)))
test_idx = perm[:n_test]
cv_idx   = perm[n_test:]

test_files = [all_data[i] for i in test_idx]
cv_data    = [all_data[i] for i in cv_idx]

test_path_file = os.path.join(OUTPUT_DIR, "test_set_paths.txt")
with open(test_path_file, "w") as f:
    for d in test_files:
        f.write(f"{d['image']}  |  {d['label']}\n")

# =============================================================================
# TRANSFORMS & LOSS
# =============================================================================
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image"], channel_dim=-1),
    EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    SpatialPadd(keys=["image", "label"], spatial_size=ROI_SIZE),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=ROI_SIZE, pos=1, neg=1, num_samples=2, image_key="image", image_threshold=0),
    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
    RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
    RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image"], channel_dim=-1),
    EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image", "label"]),
])

class BoundaryLoss(nn.Module):
    def forward(self, pred_softmax, target):
        # FIX: Explicitly disable autocast for this specific mathematical block
        with torch.amp.autocast('cuda', enabled=False):
            # Bring inputs safely to float32
            t = target.float()
            p = pred_softmax.float()
            
            eroded   = -F.max_pool3d(-t, kernel_size=3, stride=1, padding=1)
            boundary = (t - eroded).clamp(0, 1)
            pred_fg  = p[:, 1:2].clamp(1e-6, 1 - 1e-6)
            
            return F.binary_cross_entropy(pred_fg, boundary)
# =============================================================================
# MODEL — Tri-Planar OmniMamba-UNet
# =============================================================================
class OmniMambaLayer(nn.Module):
    def __init__(self, dim, d_state=32, d_conv=4, expand=2):
        super().__init__()
        self.norm       = nn.LayerNorm(dim)
        self.mamba_z    = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_y    = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_x    = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.local_conv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        b, c, h, w, d = x.shape
        local_feat = self.local_conv(x)
        z_flat = x.flatten(2).transpose(1, 2)
        out_z  = self.mamba_z(self.norm(z_flat)).transpose(1, 2).view(b, c, h, w, d)
        y_perm = x.permute(0, 1, 3, 4, 2).contiguous()
        y_flat = y_perm.flatten(2).transpose(1, 2)
        out_y  = self.mamba_y(self.norm(y_flat)).transpose(1, 2).view(b, c, w, d, h).permute(0, 1, 4, 2, 3)
        x_perm = x.permute(0, 1, 4, 2, 3).contiguous()
        x_flat = x_perm.flatten(2).transpose(1, 2)
        out_x  = self.mamba_x(self.norm(x_flat)).transpose(1, 2).view(b, c, d, h, w).permute(0, 1, 3, 4, 2)
        return out_z + out_y + out_x + local_feat + x

class MambaUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, features=(32, 64, 128, 256, 512)):
        super().__init__()
        self.enc1 = UnetBasicBlock(3, in_channels,  features[0], kernel_size=3, stride=1, norm_name="instance")
        self.enc2 = UnetBasicBlock(3, features[0],  features[1], kernel_size=3, stride=2, norm_name="instance")
        self.enc3 = UnetBasicBlock(3, features[1],  features[2], kernel_size=3, stride=2, norm_name="instance")
        self.enc4 = UnetBasicBlock(3, features[2],  features[3], kernel_size=3, stride=2, norm_name="instance")
        self.skip1_mamba = OmniMambaLayer(dim=features[0])
        self.skip2_mamba = OmniMambaLayer(dim=features[1])
        self.skip3_mamba = OmniMambaLayer(dim=features[2])
        self.skip4_mamba = OmniMambaLayer(dim=features[3])
        self.bottleneck_conv = UnetBasicBlock(3, features[3], features[4], kernel_size=3, stride=2, norm_name="instance")
        self.mamba_block1    = OmniMambaLayer(dim=features[4])
        self.mamba_block2    = OmniMambaLayer(dim=features[4])
        self.dec4 = UnetUpBlock(3, features[4], features[3], kernel_size=3, stride=1, upsample_kernel_size=2, norm_name="instance")
        self.dec3 = UnetUpBlock(3, features[3], features[2], kernel_size=3, stride=1, upsample_kernel_size=2, norm_name="instance")
        self.dec2 = UnetUpBlock(3, features[2], features[1], kernel_size=3, stride=1, upsample_kernel_size=2, norm_name="instance")
        self.dec1 = UnetUpBlock(3, features[1], features[0], kernel_size=3, stride=1, upsample_kernel_size=2, norm_name="instance")
        self.final = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.mamba_block2(self.mamba_block1(self.bottleneck_conv(e4)))
        d4 = self.dec4(b,  self.skip4_mamba(e4))
        d3 = self.dec3(d4, self.skip3_mamba(e3))
        d2 = self.dec2(d3, self.skip2_mamba(e2))
        d1 = self.dec1(d2, self.skip1_mamba(e1))
        return self.final(d1)

# =============================================================================
# FP32 EVALUATION HELPERS (NaN FIXED)
# =============================================================================
def load_model(ckpt):
    m = MambaUNet().to(device)
    raw = torch.load(ckpt, map_location=device)
    clean = {k.replace("module.", ""): v for k, v in raw.items() if k != "n_averaged"}
    clean = {k: v.float() if v.is_floating_point() else v for k, v in clean.items()}
    m.load_state_dict(clean)
    m.eval()
    return m.float()

def tta_inference_fp32(model, inputs):
    inputs = inputs.float()
    preds = []
    for axes in [[], [2], [3], [4], [2,3], [2,4], [3,4], [2,3,4]]:
        x = torch.flip(inputs.clone(), axes) if axes else inputs.clone()
        out = sliding_window_inference(x, ROI_SIZE, sw_batch_size=2, predictor=model, overlap=TTA_OVERLAP, mode="gaussian")
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        out = torch.softmax(out, dim=1)
        if axes:
            out = torch.flip(out, axes)
        preds.append(out.cpu())
    
    result = torch.stack(preds).mean(0)
    if result.isnan().any():
        result = torch.zeros_like(result)
        result[:, 0] = 1.0 
    return result

# =============================================================================
# TRAIN ONE FOLD
# =============================================================================
def train_fold(fold_idx, train_files, val_files):
    fold_start = time.time()
    banner(f"FOLD {fold_idx + 1} / {N_FOLDS}  |  train={len(train_files)}  val={len(val_files)}")

    train_ds = CacheDataset(train_files, train_transforms, cache_rate=1.0, num_workers=NUM_WORKERS)
    val_ds   = CacheDataset(val_files,   val_transforms,   cache_rate=1.0, num_workers=NUM_WORKERS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,          shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model     = MambaUNet().to(device)
    ema_model = AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(EMA_DECAY))

    dice_focal    = DiceFocalLoss(to_onehot_y=True, softmax=True, squared_pred=True)
    boundary_loss = BoundaryLoss()

    optimizer    = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler       = torch.amp.GradScaler('cuda') # Added AMP for fast/memory-efficient training
    
    warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler    = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS])

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    best_metric  = -1
    ckpt_path    = os.path.join(CKPT_DIR, f"best_fold{fold_idx+1}.pth")

    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            inputs, labels = batch["image"].to(device), batch["label"].to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                probs   = torch.softmax(outputs, dim=1)
                loss    = (dice_focal(outputs, labels) + BOUNDARY_W * boundary_loss(probs, labels)) / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                ema_model.update_parameters(model)
                optimizer.zero_grad()

            epoch_loss += loss.item() * ACCUMULATION_STEPS

        epoch_loss /= len(train_loader)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info(f"  [F{fold_idx+1}] Ep {epoch+1:>3}/{MAX_EPOCHS} | loss={epoch_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % VAL_INTERVAL == 0:
            ema_model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    vi, vl = val_data["image"].to(device), val_data["label"].to(device)
                    # Use standard TTA during validation to save time, switch to FP32 pure TTA for final test
                    vo = tta_inference_fp32(load_model(ema_model), vi) if False else sliding_window_inference(vi, ROI_SIZE, 2, ema_model, overlap=0.5) 
                    vo = torch.softmax(vo, dim=1)
                    vo = [o.argmax(dim=0, keepdim=True) for o in decollate_batch(vo)]
                    vl = decollate_batch(vl)
                    dice_metric(vo, vl)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()

                if metric > best_metric:
                    best_metric = metric
                    torch.save(ema_model.state_dict(), ckpt_path)
                    log.info(f"  [F{fold_idx+1}] ⭐ NEW BEST Val Dice: {metric:.4f}")

    log.info(f"  [F{fold_idx+1}] Done in {(time.time() - fold_start) / 60:.1f} min | Best: {best_metric:.4f}")
    return best_metric, ckpt_path

# =============================================================================
# FINAL TEST EVALUATION
# =============================================================================
def evaluate_test_set(ckpt_paths, test_files):
    banner("FINAL TEST SET EVALUATION (FP32 NaN-FIXED)")
    test_ds     = CacheDataset(test_files, val_transforms, cache_rate=1.0, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    post_pred  = AsDiscrete(argmax=True, to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd_metric   = HausdorffDistanceMetric(include_background=False, percentile=95)

    per_case_dice, per_case_hd = [], []

    with torch.no_grad():
        for idx, test_data in enumerate(test_loader):
            t_input = test_data["image"].float().to(device)
            t_label = test_data["label"]
            gt_fg = (t_label > 0).sum().item()

            all_preds = []
            for ckpt in ckpt_paths:
                m = load_model(ckpt)
                pred = tta_inference_fp32(m, t_input)
                all_preds.append(pred)
                del m
                torch.cuda.empty_cache()

            ensemble = torch.stack(all_preds).mean(0)
            
            pred_list  = [post_pred(p)  for p in decollate_batch(ensemble)]
            label_list = [post_label(l) for l in decollate_batch(t_label)]

            dice_metric(pred_list, label_list)
            case_dice = dice_metric.aggregate().item()
            dice_metric.reset()

            pred_fg = (ensemble.argmax(dim=1) > 0).sum().item()
            if gt_fg > 0 and pred_fg > 0:
                hd_metric(pred_list, label_list)
                case_hd = hd_metric.aggregate().item()
                hd_metric.reset()
            else:
                case_hd = float("nan")

            per_case_dice.append(case_dice)
            per_case_hd.append(case_hd)
            log.info(f"  Test {idx+1:>2}/{len(test_files)} | Dice={case_dice:.4f} | HD95={case_hd:.2f}mm")

    return float(np.mean(per_case_dice)), float(np.std(per_case_dice)), float(np.nanmean(per_case_hd)), float(np.nanstd(per_case_hd))

# =============================================================================
# MAIN
# =============================================================================
def main():
    total_start = time.time()
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_scores, ckpt_paths = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(cv_data)):
        fold_train = [cv_data[i] for i in train_idx]
        fold_val   = [cv_data[i] for i in val_idx]
        best_dice, ckpt = train_fold(fold_idx, fold_train, fold_val)
        fold_scores.append(best_dice)
        ckpt_paths.append(ckpt)

    cv_mean = float(np.mean(fold_scores))
    
    banner("CROSS-VALIDATION SUMMARY")
    for i, score in enumerate(fold_scores): log.info(f"  Fold {i+1}: {score:.4f}")
    log.info(f"  CV Mean: {cv_mean:.4f}")

    test_dice, test_dice_std, test_hd, test_hd_std = evaluate_test_set(ckpt_paths, test_files)

    banner("FINAL LEADERBOARD RESULTS")
    log.info(f"  Test Dice: {test_dice:.4f} ± {test_dice_std:.4f}")
    log.info(f"  Test HD95: {test_hd:.2f} mm")
    log.info(f"  Total runtime: {(time.time() - total_start) / 60:.1f} minutes")

if __name__ == "__main__":
    main()