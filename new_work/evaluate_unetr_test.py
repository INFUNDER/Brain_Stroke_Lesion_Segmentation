#!/usr/bin/env python3
import os
import sys
import glob
import json
import numpy as np
import torch
import torch.nn as nn
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, NormalizeIntensityd, EnsureTyped, AsDiscrete
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric, ConfusionMatrixMetric
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR

WORKSPACE_DIR = "/home/ronit.28010/Brain_Stroke"
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "mamba_results")
CKPT_PATH = os.path.join(WORKSPACE_DIR, "checkpoints", "best_metric_model.pth")
TEST_PATHS = os.path.join(OUTPUT_DIR, "test_set_paths.txt")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
ROI_SIZE = (96, 96, 96)
VOXEL_VOL_ML = (1.5 * 1.5 * 2.0) / 1000.0 # 0.0045 mL

def compute_dice(pred, gt):
    sum_pred = np.sum(pred)
    sum_gt = np.sum(gt)
    if sum_gt == 0:
        return 1.0 if sum_pred == 0 else 0.0
    if sum_pred == 0:
        return 0.0
    intersection = np.sum(pred * gt)
    return (2.0 * intersection) / (sum_pred + sum_gt)

def main():
    print("🚀 Starting UNETR Test Set Evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load test paths
    test_files = []
    if not os.path.exists(TEST_PATHS):
        print(f"Error: {TEST_PATHS} not found.")
        sys.exit(1)
    with open(TEST_PATHS) as f:
        for line in f:
            parts = line.strip().split("  |  ")
            if len(parts) == 2:
                test_files.append({"image": parts[0].strip(), "label": parts[1].strip()})
    print(f"Loaded {len(test_files)} test cases.")

    # Model definition
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

    # Load weights
    if not os.path.exists(CKPT_PATH):
        print(f"Error: checkpoint {CKPT_PATH} not found.")
        sys.exit(1)
    print(f"Loading weights from {CKPT_PATH}...")
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # Data transforms
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"], channel_dim=-1),
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ])

    test_ds = CacheDataset(test_files, val_transforms, cache_rate=1.0, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95)

    results = []

    with torch.no_grad():
        for idx, test_data in enumerate(test_loader):
            case_img = test_files[idx]["image"]
            case_lbl = test_files[idx]["label"]
            case_name = os.path.basename(case_lbl).split(".nii")[0]

            inputs = test_data["image"].to(device)
            labels = test_data["label"].to(device)

            # Inference (sliding window)
            outputs = sliding_window_inference(inputs, ROI_SIZE, 4, model)
            
            # Post-processing
            pred_discrete = [post_pred(p) for p in decollate_batch(outputs)]
            label_discrete = [post_label(l) for l in decollate_batch(labels)]

            # Monai Dice (which handles NaN as 0.0 or NaNs)
            dice_metric(y_pred=pred_discrete, y=label_discrete)
            monai_dice = dice_metric.aggregate().item()
            dice_metric.reset()

            # Argmax prediction and GT arrays for volume and custom corrected dice
            pred_argmax = outputs.argmax(dim=1).cpu().numpy()[0] # (H, W, D)
            labels_np = labels.cpu().numpy()[0, 0] # (H, W, D)

            # Custom corrected Dice (1.0 for true negative mimic)
            corrected_dice = compute_dice(pred_argmax, labels_np)

            # HD95
            pred_fg_voxels = np.sum(pred_argmax > 0)
            gt_fg_voxels = np.sum(labels_np > 0)

            if pred_fg_voxels > 0 and gt_fg_voxels > 0:
                hd_metric(y_pred=pred_discrete, y=label_discrete)
                case_hd = hd_metric.aggregate().item()
                hd_metric.reset()
            else:
                case_hd = float("nan")

            # Volumes in mL
            pred_vol_ml = float(pred_fg_voxels * VOXEL_VOL_ML)
            gt_vol_ml = float(gt_fg_voxels * VOXEL_VOL_ML)

            print(f"[{idx+1:>2}/{len(test_files)}] {case_name:<25} | Raw Dice={monai_dice:.4f} | Corr Dice={corrected_dice:.4f} | GT Vol={gt_vol_ml:.2f}mL | Pred Vol={pred_vol_ml:.2f}mL")

            results.append({
                "case": case_img,
                "case_name": case_name,
                "raw_dice": monai_dice,
                "corrected_dice": corrected_dice,
                "hd95": case_hd if not np.isnan(case_hd) else None,
                "gt_vol_ml": gt_vol_ml,
                "pred_vol_ml": pred_vol_ml
            })

    # Summary
    raw_dices = [r["raw_dice"] for r in results]
    corr_dices = [r["corrected_dice"] for r in results]
    
    print("\n" + "="*50)
    print("UNETR BASELINE TEST SUMMARY:")
    print(f"Mean Raw Dice      : {np.mean(raw_dices):.4f} +/- {np.std(raw_dices):.4f}")
    print(f"Mean Corrected Dice: {np.mean(corr_dices):.4f} +/- {np.std(corr_dices):.4f}")
    print("="*50)

    # Save to JSON
    out_json = os.path.join(RESULT_DIR, "unetr_test_results.json")
    with open(out_json, "w") as f:
        json.dump({
            "mean_raw_dice": float(np.mean(raw_dices)),
            "std_raw_dice": float(np.std(raw_dices)),
            "mean_corrected_dice": float(np.mean(corr_dices)),
            "std_corrected_dice": float(np.std(corr_dices)),
            "results": results
        }, f, indent=2)
    print(f"Saved results to {out_json}")

if __name__ == "__main__":
    main()
