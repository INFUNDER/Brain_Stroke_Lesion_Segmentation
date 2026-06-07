"""
SMU-Net Training Loop
======================
Full training pipeline for ISLES 2022.
Designed for H100 80GB — uses bf16 AMP throughout.

Usage:
  python train.py --data_root ./ISLES-2022_notformatted \
                  --cache_dir ./cache_preprocessed      \
                  --output_dir ./runs/smunet_fold0      \
                  --fold 0
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from monai.inferers import SlidingWindowInferer

# Local modules
sys.path.insert(0, str(Path(__file__).parent))
from isles22_dataset import (
    discover_cases, ISLES22Preprocessor,
    make_kfold_splits, get_dataloaders,
)
from smu_net import SMUNet, SMUNetLoss, compute_dice, compute_lesion_f1, model_summary

# ─────────────────────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(output_dir: str) -> logging.Logger:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / "train.log"

    logger = logging.getLogger("smunet")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING STEP
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  SMUNetLoss,
    scaler:     GradScaler,
    device:     torch.device,
    epoch:      int,
    logger:     logging.Logger,
    grad_clip:  float = 1.0,
) -> Dict[str, float]:

    model.train()
    total_loss = 0.0
    loss_components = {
        "loss_final": 0.0, "loss_d2": 0.0,
        "loss_d3":    0.0, "loss_lc": 0.0,
    }
    n_steps = len(loader)

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # bf16 AMP — H100 has native bf16 hardware support
        with autocast(dtype=torch.bfloat16):
            outputs = model(images)
            loss, ld = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for k in loss_components:
            loss_components[k] += ld.get(k, 0.0)

        if step % 50 == 0:
            logger.info(
                f"  Epoch {epoch:03d} [{step:4d}/{n_steps}]  "
                f"loss={loss.item():.4f}  "
                f"dice_ce={ld['loss_final']:.4f}  "
                f"lc={ld['loss_lc']:.4f}"
            )

    n = max(n_steps, 1)
    return {
        "train_loss": total_loss / n,
        **{k: v / n for k, v in loss_components.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION STEP (sliding window inference)
# ─────────────────────────────────────────────────────────────────────────────

def validate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    device:    torch.device,
    patch_size: Tuple[int, ...] = (96, 128, 128),
    sw_overlap: float = 0.5,
) -> Dict[str, float]:
    """
    Full-volume sliding-window inference.
    MONAI's SlidingWindowInferer handles padding, overlap averaging,
    and Gaussian weighting automatically.
    """
    model.eval()

    inferer = SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=4,        # run 4 patches per forward pass (fits H100)
        overlap=sw_overlap,
        mode="gaussian",        # Gaussian weighting reduces boundary artifacts
        sigma_scale=0.125,
    )

    dice_scores    = []
    f1_scores      = []

    with torch.no_grad():
        for images, labels, subject in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(dtype=torch.bfloat16):
                logits = inferer(images, model.forward_inference)

            # Cast back to float32 for metric computation
            logits = logits.float()

            dice = compute_dice(logits, labels)
            f1   = compute_lesion_f1(logits, labels)

            dice_scores.append(dice)
            if not np.isnan(f1):
                f1_scores.append(f1)

    return {
        "val_dice": float(np.mean(dice_scores)),
        "val_dice_std": float(np.std(dice_scores)),
        "val_f1":   float(np.mean(f1_scores)) if f1_scores else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST-TIME AUGMENTATION  (single model, not ensemble)
# ─────────────────────────────────────────────────────────────────────────────

def tta_inference(
    model:      nn.Module,
    images:     torch.Tensor,
    inferer:    SlidingWindowInferer,
    device:     torch.device,
) -> torch.Tensor:
    """
    TTA with 4 flips: original + LR flip + UD flip + LR+UD flip.
    Average logits before sigmoid — more numerically stable than
    averaging probabilities.

    This is NOT an ensemble — it's one model run 4 times with augmented
    inputs. Standard practice, ~2% Dice improvement for free.
    """
    model.eval()
    flip_axes = [
        [],          # original
        [2],         # flip D
        [3],         # flip H
        [4],         # flip W
    ]

    logit_sum = None
    with torch.no_grad():
        for axes in flip_axes:
            inp = images.clone()
            for ax in axes:
                inp = torch.flip(inp, dims=[ax])

            with autocast(dtype=torch.bfloat16):
                logit = inferer(inp, model.forward_inference).float()

            # Flip back
            for ax in axes:
                logit = torch.flip(logit, dims=[ax])

            logit_sum = logit if logit_sum is None else logit_sum + logit

    return logit_sum / len(flip_axes)


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    output_dir: str,
    model:      nn.Module,
    optimizer:  torch.optim.Optimizer,
    scheduler:  torch.optim.lr_scheduler.LRScheduler,
    epoch:      int,
    metrics:    Dict,
    is_best:    bool = False,
):
    ckpt = {
        "epoch":     epoch,
        "state_dict": model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
        "metrics":    metrics,
    }
    path = Path(output_dir) / "checkpoint_latest.pth"
    torch.save(ckpt, path)
    if is_best:
        best_path = Path(output_dir) / "checkpoint_best.pth"
        torch.save(ckpt, best_path)


def load_checkpoint(path: str, model: nn.Module,
                    optimizer=None, scheduler=None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    print(f"[checkpoint] Resumed from epoch {ckpt['epoch']} "
          f"(val_dice={ckpt['metrics'].get('val_dice', 'N/A')})")
    return ckpt["epoch"]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(args.output_dir)

    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data ──────────────────────────────────────────────────────────────
    logger.info("Discovering cases...")
    cases = discover_cases(args.data_root)

    prep = ISLES22Preprocessor(cache_dir=args.cache_dir)

    # Load or generate splits
    splits_path = Path(args.output_dir).parent / "isles22_splits.json"
    if splits_path.exists():
        logger.info(f"Loading splits from {splits_path}")
        with open(splits_path) as f:
            split_meta = json.load(f)
        subj_to_case = {c["subject"]: c for c in cases}
        splits = [
            {
                "train": [subj_to_case[s] for s in fold["train"] if s in subj_to_case],
                "val":   [subj_to_case[s] for s in fold["val"]   if s in subj_to_case],
            }
            for fold in split_meta
        ]
    else:
        splits = make_kfold_splits(cases, prep, n_splits=5, seed=42)
        split_meta = [
            {"fold": i,
             "train": [c["subject"] for c in s["train"]],
             "val":   [c["subject"] for c in s["val"]]}
            for i, s in enumerate(splits)
        ]
        with open(splits_path, "w") as f:
            json.dump(split_meta, f, indent=2)
        logger.info(f"Splits saved to {splits_path}")

    fold = splits[args.fold]
    logger.info(f"Fold {args.fold}: {len(fold['train'])} train / {len(fold['val'])} val")

    patch_size = tuple(args.patch_size)
    train_loader, val_loader = get_dataloaders(
        fold, prep,
        patch_size=patch_size,
        patches_per=args.patches_per_volume,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = SMUNet(
        in_channels=4,
        num_classes=1,
        base_features=32,
        mamba_depth=2,
        mamba_d_state=64,
    ).to(device)

    # Compile for H100 (torch.compile uses Triton backend, ~20% speedup)
    if args.compile and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    model_summary(model, input_shape=(args.batch_size, 4, *patch_size))

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
    )

    # Cosine annealing with linear warmup
    def lr_lambda(epoch):
        warmup = 10
        if epoch < warmup:
            return epoch / warmup
        progress = (epoch - warmup) / max(1, args.epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    import math
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Loss ──────────────────────────────────────────────────────────────
    # pos_weight=5 because lesion voxels are ~1-5% of all brain voxels
    criterion = SMUNetLoss(pos_weight=5.0).to(device)

    # AMP scaler — using bf16 which doesn't need dynamic scaling,
    # but scaler handles gradient underflow gracefully
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch += 1

    # ── Training Loop ─────────────────────────────────────────────────────
    best_dice = 0.0
    history = []

    logger.info(f"\nStarting training for {args.epochs} epochs...")
    logger.info(f"Patch size: {patch_size} | Batch: {args.batch_size} | LR: {args.lr}")

    for epoch in range(start_epoch, args.epochs):
        t_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, epoch, logger,
        )
        scheduler.step()

        # Validate every val_interval epochs
        val_metrics = {"val_dice": 0.0, "val_f1": 0.0}
        if epoch % args.val_interval == 0 or epoch == args.epochs - 1:
            val_metrics = validate(
                model, val_loader, device,
                patch_size=patch_size,
                sw_overlap=0.5,
            )

        # Logging
        elapsed = time.time() - t_start
        lr_now  = optimizer.param_groups[0]["lr"]
        is_best = val_metrics["val_dice"] > best_dice
        if is_best:
            best_dice = val_metrics["val_dice"]

        logger.info(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"[{elapsed:.0f}s]  "
            f"lr={lr_now:.2e}  "
            f"train_loss={train_metrics['train_loss']:.4f}  "
            f"val_dice={val_metrics['val_dice']:.4f}  "
            f"val_f1={val_metrics.get('val_f1', float('nan')):.4f}  "
            f"{'★ BEST' if is_best else ''}"
        )

        row = {"epoch": epoch, **train_metrics, **val_metrics}
        history.append(row)

        # Save
        save_checkpoint(
            args.output_dir, model, optimizer, scheduler,
            epoch, val_metrics, is_best=is_best,
        )

        # Save history
        with open(Path(args.output_dir) / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    logger.info(f"\nTraining complete. Best val_dice = {best_dice:.4f}")
    return best_dice


# ─────────────────────────────────────────────────────────────────────────────
# FINAL EVALUATION WITH TTA
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_with_tta(args):
    """
    Load best checkpoint and evaluate with TTA on validation set.
    Run this after training to get final numbers.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cases = discover_cases(args.data_root)
    prep  = ISLES22Preprocessor(cache_dir=args.cache_dir)

    with open(Path(args.output_dir).parent / "isles22_splits.json") as f:
        split_meta = json.load(f)

    subj_to_case = {c["subject"]: c for c in cases}
    val_cases = [subj_to_case[s] for s in split_meta[args.fold]["val"]
                 if s in subj_to_case]

    from torch.utils.data import DataLoader
    from isles22_dataset import ISLES22Dataset

    val_ds = ISLES22Dataset(val_cases, prep, mode="val")
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    model = SMUNet().to(device)
    ckpt_path = Path(args.output_dir) / "checkpoint_best.pth"
    load_checkpoint(str(ckpt_path), model)

    inferer = SlidingWindowInferer(
        roi_size=tuple(args.patch_size),
        sw_batch_size=4,
        overlap=0.5,
        mode="gaussian",
    )

    dice_scores = []
    f1_scores   = []

    model.eval()
    for images, labels, subject in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = tta_inference(model, images, inferer, device)

        dice = compute_dice(logits, labels)
        f1   = compute_lesion_f1(logits, labels)
        dice_scores.append(dice)
        if not np.isnan(f1):
            f1_scores.append(f1)
        print(f"  {subject[0]}  dice={dice:.4f}  f1={f1:.4f}")

    print(f"\nFold {args.fold} TTA Results:")
    print(f"  Mean Dice : {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"  Mean F1   : {np.mean(f1_scores):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("SMU-Net Training")

    # Paths
    p.add_argument("--data_root",   default="./ISLES-2022_notformatted")
    p.add_argument("--cache_dir",   default="./cache_preprocessed")
    p.add_argument("--output_dir",  default="./runs/smunet_fold0")
    p.add_argument("--resume",      default=None, help="path to checkpoint to resume")

    # Training
    p.add_argument("--fold",        type=int,   default=0)
    p.add_argument("--epochs",      type=int,   default=300)
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--val_interval",type=int,   default=10)

    # Data
    p.add_argument("--patch_size",  type=int, nargs=3, default=[96, 128, 128],
                   help="Patch size D H W — updated to match ~73×112×112 volumes")
    p.add_argument("--patches_per_volume", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)

    # Model
    p.add_argument("--compile",     action="store_true",
                   help="Use torch.compile (Triton, ~20pct speedup on H100)")

    # Mode
    p.add_argument("--eval_tta",    action="store_true",
                   help="Run TTA evaluation on best checkpoint instead of training")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.eval_tta:
        evaluate_with_tta(args)
    else:
        train(args)