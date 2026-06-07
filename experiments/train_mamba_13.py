# =============================================================================
# MAMBA-UNET: FULLY AUTOMATED END-TO-END PIPELINE
# H100 80GB | 5-Fold Cross-Validation | Final Test Evaluation
#
# USAGE:
#   qsub run_mamba.pbs
#   OR directly: python mamba_pipeline.py
#
# OUTPUT:
#   logs/                  — per-epoch logs for each fold
#   checkpoints/           — best model per fold
#   results/               — fold scores, final test score, summary report
#   test_set_paths.txt     — held-out test cases (never used during training)
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
    RandGaussianSmoothd,
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.networks.blocks import UnetBasicBlock, UnetUpBlock
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel

# =============================================================================
# CONFIG — edit these if needed
# =============================================================================
DATA_DIR      = "/home/ronit.28010/Brain_Stroke/ISLES2022_Formatted"
OUTPUT_DIR    = os.path.join(os.environ.get("PBS_O_WORKDIR", "."), "mamba_results")

ROI_SIZE      = (96, 96, 96)
BATCH_SIZE    = 1
MAX_EPOCHS    = 200
VAL_INTERVAL  = 2
NUM_WORKERS   = 4   # safe with 8 CPUs allocated
WARMUP_EPOCHS = 10
EMA_DECAY     = 0.999
N_FOLDS       = 5
TEST_FRACTION = 0.10       # 10% held-out test set (25 of 250 images)
BOUNDARY_W    = 0.2        # weight for boundary loss term
SEED          = 42

# =============================================================================
# DIRECTORY SETUP
# =============================================================================
LOG_DIR   = os.path.join(OUTPUT_DIR, "logs")
CKPT_DIR  = os.path.join(OUTPUT_DIR, "checkpoints")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")

for d in [OUTPUT_DIR, LOG_DIR, CKPT_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# LOGGING — writes to both console and file simultaneously
# =============================================================================
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

# =============================================================================
# SETUP
# =============================================================================
set_determinism(seed=SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

banner("MAMBA-UNET AUTOMATED PIPELINE")
log.info(f"Run ID      : {run_id}")
log.info(f"MONAI       : {monai.__version__}")
log.info(f"PyTorch     : {torch.__version__}")
log.info(f"Device      : {device}")
if torch.cuda.is_available():
    log.info(f"GPU         : {torch.cuda.get_device_name(0)}")
    log.info(f"VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
log.info(f"Output dir  : {OUTPUT_DIR}")

# =============================================================================
# DATA — carve test set once, deterministically
# =============================================================================
if not os.path.exists(DATA_DIR):
    log.error(f"DATA_DIR not found: {DATA_DIR}")
    sys.exit(1)

train_images = sorted(glob.glob(os.path.join(DATA_DIR, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(DATA_DIR, "labelsTr", "*.nii.gz")))

if len(train_images) == 0:
    log.error("No .nii.gz images found in imagesTr/")
    sys.exit(1)

assert len(train_images) == len(train_labels), \
    f"Image/label count mismatch: {len(train_images)} vs {len(train_labels)}"

all_data = [{"image": i, "label": l} for i, l in zip(train_images, train_labels)]

rng      = np.random.default_rng(seed=SEED)
perm     = rng.permutation(len(all_data))
n_test   = max(1, int(TEST_FRACTION * len(all_data)))
test_idx = perm[:n_test]
cv_idx   = perm[n_test:]

test_files = [all_data[i] for i in test_idx]
cv_data    = [all_data[i] for i in cv_idx]

# Save test paths so they are auditable
test_path_file = os.path.join(OUTPUT_DIR, "test_set_paths.txt")
with open(test_path_file, "w") as f:
    for d in test_files:
        f.write(f"{d['image']}  |  {d['label']}\n")

banner("DATASET SPLIT")
log.info(f"Total images     : {len(all_data)}")
log.info(f"Held-out test    : {len(test_files)}")
log.info(f"CV pool          : {len(cv_data)}  ({N_FOLDS} folds)")
log.info(f"Train per fold   : ~{int(len(cv_data) * (N_FOLDS-1) / N_FOLDS)}")
log.info(f"Val per fold     : ~{int(len(cv_data) / N_FOLDS)}")
log.info(f"Test paths saved : {test_path_file}")

# =============================================================================
# TRANSFORMS
# =============================================================================
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
        keys=["image", "label"], label_key="label",
        spatial_size=ROI_SIZE, pos=1, neg=1, num_samples=2,
        image_key="image", image_threshold=0,
    ),
    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
    RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
    RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    RandGaussianSmoothd(keys=["image"], prob=0.2,
                        sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
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

# test_transforms = val_transforms (no augmentation, no cropping)
test_transforms = val_transforms

# =============================================================================
# BOUNDARY LOSS
# =============================================================================
class BoundaryLoss(nn.Module):
    def forward(self, pred_softmax, target):
        t        = target.float()
        eroded   = -F.max_pool3d(-t, kernel_size=3, stride=1, padding=1)
        boundary = (t - eroded).clamp(0, 1)
        pred_fg  = pred_softmax[:, 1:2].clamp(1e-6, 1 - 1e-6)
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

        b = self.bottleneck_conv(e4)
        b = self.mamba_block1(b)
        b = self.mamba_block2(b)

        d4 = self.dec4(b,  self.skip4_mamba(e4))
        d3 = self.dec3(d4, self.skip3_mamba(e3))
        d2 = self.dec2(d3, self.skip2_mamba(e2))
        d1 = self.dec1(d2, self.skip1_mamba(e1))

        return self.final(d1)

# =============================================================================
# TTA INFERENCE
# =============================================================================
def tta_inference(model, inputs, roi_size):
    flip_combos = [
        [], [2], [3], [4],
        [2, 3], [2, 4], [3, 4],
        [2, 3, 4],
    ]
    preds = []
    for axes in flip_combos:
        x = inputs.clone()
        if axes:
            x = torch.flip(x, axes)
        out = sliding_window_inference(
            x, roi_size, sw_batch_size=4, predictor=model,
            overlap=0.5, mode="gaussian",
        )
        out = torch.softmax(out, dim=1)
        if axes:
            out = torch.flip(out, axes)
        preds.append(out)
    return torch.stack(preds).mean(0)

# =============================================================================
# TRAIN ONE FOLD
# =============================================================================
def train_fold(fold_idx, train_files, val_files):
    fold_log = logging.getLogger()
    fold_start = time.time()

    banner(f"FOLD {fold_idx + 1} / {N_FOLDS}  |  train={len(train_files)}  val={len(val_files)}")

    # Per-fold log file
    fold_log_path = os.path.join(LOG_DIR, f"fold{fold_idx+1}_{run_id}.log")
    fh = logging.FileHandler(fold_log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
    fold_log.addHandler(fh)

    train_ds = CacheDataset(train_files, train_transforms, cache_rate=1.0, num_workers=NUM_WORKERS)
    val_ds   = CacheDataset(val_files,   val_transforms,   cache_rate=1.0, num_workers=NUM_WORKERS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,          shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model     = MambaUNet().to(device)
    ema_model = AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(EMA_DECAY))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Model params: {n_params/1e6:.2f}M")

    dice_focal    = DiceFocalLoss(to_onehot_y=True, softmax=True, squared_pred=True)
    boundary_loss = BoundaryLoss()

    optimizer    = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler    = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[WARMUP_EPOCHS])

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    best_metric  = -1
    ckpt_path    = os.path.join(CKPT_DIR, f"best_fold{fold_idx+1}.pth")
    history      = []

    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            probs   = torch.softmax(outputs, dim=1)
            loss    = dice_focal(outputs, labels) + BOUNDARY_W * boundary_loss(probs, labels)
            loss.backward()
            optimizer.step()
            ema_model.update_parameters(model)
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        log.info(f"  [F{fold_idx+1}] Ep {epoch+1:>3}/{MAX_EPOCHS} | loss={epoch_loss:.4f} | lr={lr:.2e}")

        if (epoch + 1) % VAL_INTERVAL == 0:
            ema_model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    vi = val_data["image"].to(device)
                    vl = val_data["label"].to(device)
                    vo = tta_inference(ema_model, vi, ROI_SIZE)
                    vo = [o.argmax(dim=0, keepdim=True) for o in decollate_batch(vo)]
                    vl = decollate_batch(vl)
                    dice_metric(vo, vl)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                history.append({"epoch": epoch + 1, "val_dice": metric, "loss": epoch_loss})

                if metric > best_metric:
                    best_metric = metric
                    torch.save(ema_model.state_dict(), ckpt_path)
                    log.info(f"  [F{fold_idx+1}] *** NEW BEST Val Dice: {metric:.4f}  -> saved")
                else:
                    log.info(f"  [F{fold_idx+1}] Val Dice: {metric:.4f}  (best: {best_metric:.4f})")

    # Save fold history
    hist_path = os.path.join(RESULT_DIR, f"fold{fold_idx+1}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    fold_log.removeHandler(fh)
    fh.close()

    elapsed = (time.time() - fold_start) / 60
    log.info(f"  [F{fold_idx+1}] Done in {elapsed:.1f} min | Best Val Dice: {best_metric:.4f}")
    return best_metric, ckpt_path

# =============================================================================
# FINAL TEST EVALUATION
# Uses an ensemble of all 5 fold checkpoints via TTA
# =============================================================================
def evaluate_test_set(ckpt_paths, test_files):
    banner("FINAL TEST SET EVALUATION")
    log.info(f"  Test cases : {len(test_files)}")
    log.info(f"  Checkpoints: {len(ckpt_paths)} fold models (ensemble + TTA)")

    # Load all fold models
    fold_models = []
    for ckpt in ckpt_paths:
        m = MambaUNet().to(device)
        # EMA checkpoint uses module. prefix
        state = torch.load(ckpt, map_location=device)
        # AveragedModel wraps weights under 'module.' key
        new_state = {}
        for k, v in state.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state[new_key] = v
        m.load_state_dict(new_state)
        m.eval()
        fold_models.append(m)

    log.info(f"  All {len(fold_models)} models loaded successfully")

    test_ds     = CacheDataset(test_files, test_transforms, cache_rate=1.0, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd_metric   = HausdorffDistanceMetric(include_background=False, percentile=95)

    per_case_dice = []
    per_case_hd   = []

    with torch.no_grad():
        for idx, test_data in enumerate(test_loader):
            t_input  = test_data["image"].to(device)
            t_label  = test_data["label"].to(device)

            # Ensemble: average softmax probabilities across all fold models
            all_preds = []
            for m in fold_models:
                pred = tta_inference(m, t_input, ROI_SIZE)  # (1, 2, H, W, D)
                all_preds.append(pred)
            ensemble_pred = torch.stack(all_preds).mean(0)  # average across folds

            # Hard labels for metrics
            pred_hard  = [p.argmax(dim=0, keepdim=True) for p in decollate_batch(ensemble_pred)]
            label_hard = decollate_batch(t_label)

            dice_metric(pred_hard, label_hard)
            hd_metric(pred_hard, label_hard)

            case_dice = dice_metric.aggregate().item()
            case_hd   = hd_metric.aggregate().item()
            dice_metric.reset()
            hd_metric.reset()

            per_case_dice.append(case_dice)
            per_case_hd.append(case_hd)

            img_name = os.path.basename(test_data["image_meta_dict"]["filename_or_obj"][0])
            log.info(f"  Case {idx+1:>2}/{len(test_files)} | {img_name:<30} | Dice={case_dice:.4f} | HD95={case_hd:.2f}mm")

    # Aggregate
    mean_dice = float(np.mean(per_case_dice))
    std_dice  = float(np.std(per_case_dice))
    mean_hd   = float(np.nanmean(per_case_hd))
    std_hd    = float(np.nanstd(per_case_hd))

    return mean_dice, std_dice, mean_hd, std_hd, per_case_dice, per_case_hd

# =============================================================================
# MAIN
# =============================================================================
def main():
    total_start = time.time()

    # ---- 5-FOLD CROSS-VALIDATION ----
    kf         = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds      = list(kf.split(cv_data))
    fold_scores = []
    ckpt_paths  = []

    for fold_idx in range(N_FOLDS):
        train_idx, val_idx = folds[fold_idx]
        fold_train = [cv_data[i] for i in train_idx]
        fold_val   = [cv_data[i] for i in val_idx]

        best_dice, ckpt = train_fold(fold_idx, fold_train, fold_val)
        fold_scores.append(best_dice)
        ckpt_paths.append(ckpt)

    # ---- CV SUMMARY ----
    cv_mean = float(np.mean(fold_scores))
    cv_std  = float(np.std(fold_scores))

    banner("CROSS-VALIDATION SUMMARY")
    for i, score in enumerate(fold_scores):
        log.info(f"  Fold {i+1}: Val Dice = {score:.4f}")
    log.info(f"  CV Mean  : {cv_mean:.4f}")
    log.info(f"  CV Std   : {cv_std:.4f}")
    log.info(f"  CV Result: {cv_mean:.4f} ± {cv_std:.4f}")

    # ---- FINAL TEST EVALUATION ----
    test_dice, test_dice_std, test_hd, test_hd_std, \
        per_case_dice, per_case_hd = evaluate_test_set(ckpt_paths, test_files)

    total_time = (time.time() - total_start) / 60

    # ---- FINAL REPORT ----
    banner("FINAL RESULTS")
    log.info(f"  CV  Dice (5-fold)  : {cv_mean:.4f} ± {cv_std:.4f}")
    log.info(f"  Test Dice          : {test_dice:.4f} ± {test_dice_std:.4f}")
    log.info(f"  Test HD95          : {test_hd:.2f} ± {test_hd_std:.2f} mm")
    log.info("")
    log.info("  ISLES 2022 Leaderboard comparison:")
    log.info("    SEALS   (Rank 1) : Dice = 0.821")
    log.info("    NVAUTO  (Rank 2) : Dice = 0.824")
    log.info("    Factorizer (R3)  : Dice = 0.812")
    log.info(f"   Ours (MambaUNet) : Dice = {test_dice:.4f}  <--- YOUR SCORE")
    log.info("")
    log.info(f"  Total runtime: {total_time:.1f} minutes")

    # ---- SAVE RESULTS JSON ----
    results = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_folds": N_FOLDS,
            "max_epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "roi_size": list(ROI_SIZE),
            "warmup_epochs": WARMUP_EPOCHS,
            "ema_decay": EMA_DECAY,
            "boundary_weight": BOUNDARY_W,
            "seed": SEED,
        },
        "cv_results": {
            "fold_scores": fold_scores,
            "mean": cv_mean,
            "std": cv_std,
        },
        "test_results": {
            "per_case_dice": per_case_dice,
            "per_case_hd95": per_case_hd,
            "mean_dice": test_dice,
            "std_dice": test_dice_std,
            "mean_hd95": test_hd,
            "std_hd95": test_hd_std,
        },
        "runtime_minutes": total_time,
    }

    results_path = os.path.join(RESULT_DIR, f"final_results_{run_id}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Plain-text summary for quick reading
    summary_path = os.path.join(RESULT_DIR, f"summary_{run_id}.txt")
    with open(summary_path, "w") as f:
        f.write(f"MAMBA-UNET FINAL RESULTS  |  Run: {run_id}\n")
        f.write("=" * 50 + "\n\n")
        f.write("CROSS-VALIDATION (5-fold)\n")
        for i, s in enumerate(fold_scores):
            f.write(f"  Fold {i+1}: {s:.4f}\n")
        f.write(f"  Mean ± Std: {cv_mean:.4f} ± {cv_std:.4f}\n\n")
        f.write("HELD-OUT TEST SET\n")
        f.write(f"  Dice  : {test_dice:.4f} ± {test_dice_std:.4f}\n")
        f.write(f"  HD95  : {test_hd:.2f} ± {test_hd_std:.2f} mm\n\n")
        f.write("LEADERBOARD COMPARISON\n")
        f.write("  SEALS    Rank 1 : 0.821\n")
        f.write("  NVAUTO   Rank 2 : 0.824\n")
        f.write("  Ours            : {:.4f}\n".format(test_dice))
        f.write(f"\nRuntime: {total_time:.1f} minutes\n")

    log.info(f"  Full results : {results_path}")
    log.info(f"  Summary      : {summary_path}")
    banner("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()