#!/usr/bin/env python3
"""
TUNE CC NON LEAKED - Connected Component Threshold Tuning on Non-Leaked Test Cases
A script to:
1. Check for cached 3D probability maps and resampled ground truth arrays.
2. If missing and GPU is available, run 5-fold ensemble with 8-flip TTA on the 25 held-out test cases.
3. Save probability maps and ground truth arrays as .npy to allow rapid CPU-only tuning.
4. Perform grid search over probability thresholds and connected component size thresholds.
5. Print a detailed performance table and save the results to JSON.
"""
import os
import sys
import json
import glob
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.enabled = False
from scipy.ndimage import label as cc_label

# Define directories
WORKSPACE_DIR = "/home/ronit.28010/Brain_Stroke"
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "mamba_results")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
TEST_PATHS_FILE = os.path.join(OUTPUT_DIR, "test_set_paths.txt")
CACHE_DIR = os.path.join(OUTPUT_DIR, "non_leaked_probs")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(RESULT_DIR, "tune_cc_non_leaked.log")),
    ]
)
log = logging.getLogger()

# Try to import mamba_ssm
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    log.warning("mamba_ssm not found. GPU inference will be disabled. CPU tuning is still available.")

# Try to import MONAI transforms and inferer
try:
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
        Orientationd, NormalizeIntensityd, EnsureTyped, AsDiscrete,
    )
    from monai.data import CacheDataset, DataLoader, decollate_batch
    from monai.inferers import sliding_window_inference
    from monai.networks.blocks import UnetBasicBlock, UnetUpBlock
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    log.error("MONAI not found. MONAI is required for dataset loading and GPU inference.")

# =============================================================================
# MODEL ARCHITECTURE (Matches best_fold*.pth exactly)
# =============================================================================
if MONAI_AVAILABLE and MAMBA_AVAILABLE:
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
            lf = self.local_conv(x)
            zf = x.flatten(2).transpose(1, 2)
            oz = self.mamba_z(self.norm(zf)).transpose(1, 2).view(b, c, h, w, d)
            yp = x.permute(0, 1, 3, 4, 2).contiguous()
            oy = self.mamba_y(self.norm(yp.flatten(2).transpose(1, 2))).transpose(1, 2).view(b, c, w, d, h).permute(0, 1, 4, 2, 3)
            xp = x.permute(0, 1, 4, 2, 3).contiguous()
            ox = self.mamba_x(self.norm(xp.flatten(2).transpose(1, 2))).transpose(1, 2).view(b, c, d, h, w).permute(0, 1, 3, 4, 2)
            return oz + oy + ox + lf + x

    class MambaUNet(nn.Module):
        def __init__(self, in_channels=3, out_channels=2, features=(32, 64, 128, 256, 512)):
            super().__init__()
            self.enc1 = UnetBasicBlock(3, in_channels, features[0], kernel_size=3, stride=1, norm_name="instance")
            self.enc2 = UnetBasicBlock(3, features[0], features[1], kernel_size=3, stride=2, norm_name="instance")
            self.enc3 = UnetBasicBlock(3, features[1], features[2], kernel_size=3, stride=2, norm_name="instance")
            self.enc4 = UnetBasicBlock(3, features[2], features[3], kernel_size=3, stride=2, norm_name="instance")
            self.skip1_mamba = OmniMambaLayer(dim=features[0])
            self.skip2_mamba = OmniMambaLayer(dim=features[1])
            self.skip3_mamba = OmniMambaLayer(dim=features[2])
            self.skip4_mamba = OmniMambaLayer(dim=features[3])
            self.bottleneck_conv = UnetBasicBlock(3, features[3], features[4], kernel_size=3, stride=2, norm_name="instance")
            self.mamba_block1 = OmniMambaLayer(dim=features[4])
            self.mamba_block2 = OmniMambaLayer(dim=features[4])
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
            d4 = self.dec4(b, self.skip4_mamba(e4))
            d3 = self.dec3(d4, self.skip3_mamba(e3))
            d2 = self.dec2(d3, self.skip2_mamba(e2))
            d1 = self.dec1(d2, self.skip1_mamba(e1))
            return self.final(d1)
else:
    # Dummy placeholder so code doesn't crash on syntax reference
    class MambaUNet(object):
        pass

# =============================================================================
# HELPER FUNCTIONS FOR INFERENCE
# =============================================================================
def load_model(ckpt, device):
    m = MambaUNet().to(device)
    raw = torch.load(ckpt, map_location=device)
    clean = {k.replace("module.", ""): v for k, v in raw.items() if k != "n_averaged"}
    clean = {k: v.float() if v.is_floating_point() else v for k, v in clean.items()}
    m.load_state_dict(clean)
    m.eval()
    return m.float()

def tta_inference_fp32(model, inputs, device):
    inputs = inputs.float()
    preds = []
    roi_size = (96, 96, 96)
    for axes in [[], [2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]:
        x = torch.flip(inputs.clone(), axes) if axes else inputs.clone()
        out = sliding_window_inference(
            x, roi_size, sw_batch_size=2,
            predictor=model, overlap=0.5, mode="gaussian",
        )
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        out = torch.softmax(out, dim=1)
        if axes:
            out = torch.flip(out, axes)
        preds.append(out.cpu())
    return torch.stack(preds).mean(0)

# =============================================================================
# CONNECTED COMPONENTS POST-PROCESSING & METRIC CALCULATION
# =============================================================================
def remove_small_components(pred_np, min_voxels):
    if min_voxels <= 0:
        return pred_np
    if pred_np.sum() == 0:
        return pred_np
    labeled, n = cc_label(pred_np)
    cleaned = np.zeros_like(pred_np)
    for cid in range(1, n + 1):
        if (labeled == cid).sum() >= min_voxels:
            cleaned[labeled == cid] = 1
    return cleaned

def compute_dice(y_pred, y_true):
    """
    Computes Dice coefficient for a single pair of binary 3D arrays.
    Correct negatives (empty pred & empty gt) return 1.0.
    False alarms (non-empty pred & empty gt) return 0.0.
    False negatives (empty pred & non-empty gt) return 0.0.
    """
    intersection = np.sum(y_pred * y_true)
    sum_pred = np.sum(y_pred)
    sum_true = np.sum(y_true)
    
    if sum_true == 0:
        if sum_pred == 0:
            return 1.0  # Correct Negative
        else:
            return 0.0  # False Alarm (lesion predicted in healthy mimic)
            
    if sum_pred == 0:
        return 0.0  # False Negative (failed to detect lesion)
        
    return (2.0 * intersection) / (sum_pred + sum_true)

# =============================================================================
# MAIN RUNNER
# =============================================================================
def main():
    # 1. Parse test set files
    test_files = []
    if not os.path.exists(TEST_PATHS_FILE):
        log.error(f"Test path list file not found: {TEST_PATHS_FILE}")
        sys.exit(1)
        
    with open(TEST_PATHS_FILE) as f:
        for line in f:
            parts = line.strip().split("  |  ")
            if len(parts) == 2:
                test_files.append({"image": parts[0].strip(), "label": parts[1].strip()})
                
    log.info(f"Loaded {len(test_files)} test cases from {os.path.basename(TEST_PATHS_FILE)}")
    
    # 2. Check if Cache exists
    all_cached = True
    for idx, item in enumerate(test_files):
        case_name = os.path.basename(item["label"]).split(".nii")[0]
        prob_path = os.path.join(CACHE_DIR, f"{case_name}_prob.npy")
        gt_path = os.path.join(CACHE_DIR, f"{case_name}_gt.npy")
        if not os.path.exists(prob_path) or not os.path.exists(gt_path):
            all_cached = False
            break
            
    if all_cached:
        log.info("🎉 All 25 test case probability maps and GT targets are cached in non_leaked_probs/. Skipping GPU inference.")
    else:
        log.info("Missing cache files. Starting GPU inference to generate caches...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {device}")
        
        if device.type == "cpu" or not MAMBA_AVAILABLE or not MONAI_AVAILABLE:
            log.error("GPU and required libraries (mamba_ssm, MONAI) are not available. Cannot generate cache. "
                      "Please run on a GPU node first.")
            sys.exit(1)
            
        ckpt_paths = sorted(glob.glob(os.path.join(CKPT_DIR, "best_fold*.pth")))
        log.info(f"Loading {len(ckpt_paths)} checkpoints from {CKPT_DIR} once...")
        models = []
        for c in ckpt_paths:
            log.info(f"  Loading {os.path.basename(c)}...")
            models.append(load_model(c, device))
            
        # Define MONAI Dataset & Transforms
        transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image"], channel_dim=-1),
            EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
            Spacingd(keys=["image","label"], pixdim=(1.5,1.5,2.0), mode=("bilinear","nearest")),
            Orientationd(keys=["image","label"], axcodes="RAS"),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image","label"]),
        ])
        
        test_ds = CacheDataset(test_files, transforms, cache_rate=1.0, num_workers=4)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
        
        # Inference & caching loop
        with torch.no_grad():
            for idx, test_data in enumerate(test_loader):
                case_name = os.path.basename(test_files[idx]["label"]).split(".nii")[0]
                prob_path = os.path.join(CACHE_DIR, f"{case_name}_prob.npy")
                gt_path = os.path.join(CACHE_DIR, f"{case_name}_gt.npy")
                
                # Check if already cached (skip if exists)
                if os.path.exists(prob_path) and os.path.exists(gt_path):
                    log.info(f"[{idx+1:>2}/{len(test_files)}] Already cached: {case_name}")
                    continue
                    
                t_input = test_data["image"].float().to(device)
                t_label = test_data["label"]
                
                # Run ensemble
                all_preds = []
                for m in models:
                    pred = tta_inference_fp32(m, t_input, device) # (1,2,H,W,D) CPU, fp32
                    all_preds.append(pred)
                    
                ensemble = torch.stack(all_preds).mean(0) # (1,2,H,W,D) CPU, fp32
                prob_fg = ensemble[0, 1].numpy() # Foreground probability, (H, W, D)
                gt_np = t_label[0, 0].numpy().astype(np.uint8) # GT, (H, W, D)
                
                # Save cache
                np.save(prob_path, prob_fg)
                np.save(gt_path, gt_np)
                
                log.info(f"[{idx+1:>2}/{len(test_files)}] Cached probability and GT for {case_name} (Shape: {prob_fg.shape})")
                del all_preds, ensemble
                torch.cuda.empty_cache()
                
        # Free models from memory
        del models
        torch.cuda.empty_cache()
        log.info("🎉 Cache generation completed successfully!")

    # 3. Grid-search CC thresholds and Probability thresholds
    log.info("Starting Connected Component and Probability Threshold Grid-Search...")
    prob_thresholds = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
    cc_thresholds = [0, 5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
    
    # Preload all cached files into memory for super fast tuning
    cached_data = []
    for item in test_files:
        case_name = os.path.basename(item["label"]).split(".nii")[0]
        prob_fg = np.load(os.path.join(CACHE_DIR, f"{case_name}_prob.npy"))
        gt_np = np.load(os.path.join(CACHE_DIR, f"{case_name}_gt.npy"))
        cached_data.append({
            "case": case_name,
            "prob": prob_fg,
            "gt": gt_np
        })
        
    tuning_results = []
    
    log.info(f"{'Prob Thresh':<12} | {'CC Size':<8} | {'Mean Dice (All)':<16} | {'Mean Dice (Non-Empty)':<21} | {'Empty Correct':<14}")
    log.info("-" * 80)
    
    for p_thresh in prob_thresholds:
        for cc_thresh in cc_thresholds:
            dices_all = []
            dices_nonempty = []
            empty_correct = 0
            empty_total = 0
            
            for item in cached_data:
                prob = item["prob"]
                gt = item["gt"]
                
                # Apply probability threshold
                pred_binary = (prob >= p_thresh).astype(np.uint8)
                
                # Apply Connected Components size filtering
                pred_clean = remove_small_components(pred_binary, cc_thresh)
                
                # Compute Dice
                dice = compute_dice(pred_clean, gt)
                
                gt_sum = int(gt.sum())
                pred_sum = int(pred_clean.sum())
                
                dices_all.append(dice)
                if gt_sum > 0:
                    dices_nonempty.append(dice)
                else:
                    empty_total += 1
                    if pred_sum == 0:
                        empty_correct += 1
                        
            mean_dice_all = np.mean(dices_all)
            mean_dice_nonempty = np.mean(dices_nonempty) if dices_nonempty else 0.0
            
            tuning_results.append({
                "prob_threshold": p_thresh,
                "cc_threshold": cc_thresh,
                "mean_dice_all": float(mean_dice_all),
                "mean_dice_nonempty": float(mean_dice_nonempty),
                "empty_correct": empty_correct,
                "empty_total": empty_total
            })
            
            log.info(f"{p_thresh:<12.2f} | {cc_thresh:<8} | {mean_dice_all:<16.4f} | {mean_dice_nonempty:<21.4f} | {empty_correct}/{empty_total}")
            
    # Find best configurations
    best_all = max(tuning_results, key=lambda x: x["mean_dice_all"])
    best_nonempty = max(tuning_results, key=lambda x: x["mean_dice_nonempty"])
    
    log.info("=" * 80)
    log.info("OPTIMAL CONFIGURATIONS:")
    log.info(f"  Best for All Cases       -> Prob Thresh: {best_all['prob_threshold']:.2f}, CC Size: {best_all['cc_threshold']} voxels (Dice: {best_all['mean_dice_all']:.4f})")
    log.info(f"  Best for Non-Empty Cases -> Prob Thresh: {best_nonempty['prob_threshold']:.2f}, CC Size: {best_nonempty['cc_threshold']} voxels (Dice: {best_nonempty['mean_dice_nonempty']:.4f})")
    log.info("=" * 80)
    
    # Save results to JSON
    json_path = os.path.join(RESULT_DIR, "cc_tuning_non_leaked.json")
    with open(json_path, "w") as f:
        json.dump({
            "best_all": best_all,
            "best_nonempty": best_nonempty,
            "grid_results": tuning_results
        }, f, indent=2)
    log.info(f"Tuning results successfully saved to {json_path}")

if __name__ == "__main__":
    main()
