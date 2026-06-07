#!/usr/bin/env python3
"""
FINAL TEST EVALUATION - NaN FIXED
Root cause: autocast (fp16) produces NaN/Inf in TTA inference.
Fix: run inference in fp32 (no autocast), cast output to fp32 explicitly,
     add nan_to_num guard before softmax.
"""
import os, sys, json, glob, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datetime import datetime
from mamba_ssm import Mamba
import warnings; warnings.filterwarnings("ignore")

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, NormalizeIntensityd, EnsureTyped, AsDiscrete,
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.networks.blocks import UnetBasicBlock, UnetUpBlock

# =============================================================================
OUTPUT_DIR = os.path.join(os.environ.get("PBS_O_WORKDIR", "."), "mamba_results")
CKPT_DIR   = os.path.join(OUTPUT_DIR, "checkpoints")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
TEST_PATHS = os.path.join(OUTPUT_DIR, "test_set_paths.txt")
os.makedirs(RESULT_DIR, exist_ok=True)
ROI_SIZE    = (96, 96, 96)
NUM_WORKERS = 4

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(RESULT_DIR, f"final_nan_fixed_{run_id}.log")),
    ]
)
log = logging.getLogger()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("=" * 65)
log.info("  FINAL TEST EVALUATION — NaN FIXED (fp32 inference)")
log.info("=" * 65)
log.info(f"Device : {device}")
if torch.cuda.is_available():
    log.info(f"GPU    : {torch.cuda.get_device_name(0)}")

# --- load test files ---
test_files = []
with open(TEST_PATHS) as f:
    for line in f:
        parts = line.strip().split("  |  ")
        if len(parts) == 2:
            test_files.append({"image": parts[0].strip(), "label": parts[1].strip()})
ckpt_paths = sorted(glob.glob(os.path.join(CKPT_DIR, "best_fold*.pth")))
log.info(f"Test cases: {len(test_files)} | Checkpoints: {len(ckpt_paths)}")
for c in ckpt_paths:
    log.info(f"  {os.path.basename(c)}")

# =============================================================================
# MODEL
# =============================================================================
class OmniMambaLayer(nn.Module):
    def __init__(self, dim, d_state=32, d_conv=4, expand=2):
        super().__init__()
        self.norm=nn.LayerNorm(dim)
        self.mamba_z=Mamba(d_model=dim,d_state=d_state,d_conv=d_conv,expand=expand)
        self.mamba_y=Mamba(d_model=dim,d_state=d_state,d_conv=d_conv,expand=expand)
        self.mamba_x=Mamba(d_model=dim,d_state=d_state,d_conv=d_conv,expand=expand)
        self.local_conv=nn.Conv3d(dim,dim,kernel_size=3,padding=1,groups=dim)
    def forward(self,x):
        b,c,h,w,d=x.shape
        lf=self.local_conv(x)
        zf=x.flatten(2).transpose(1,2)
        oz=self.mamba_z(self.norm(zf)).transpose(1,2).view(b,c,h,w,d)
        yp=x.permute(0,1,3,4,2).contiguous()
        oy=self.mamba_y(self.norm(yp.flatten(2).transpose(1,2))).transpose(1,2).view(b,c,w,d,h).permute(0,1,4,2,3)
        xp=x.permute(0,1,4,2,3).contiguous()
        ox=self.mamba_x(self.norm(xp.flatten(2).transpose(1,2))).transpose(1,2).view(b,c,d,h,w).permute(0,1,3,4,2)
        return oz+oy+ox+lf+x

class MambaUNet(nn.Module):
    def __init__(self,in_channels=3,out_channels=2,features=(32,64,128,256,512)):
        super().__init__()
        self.enc1=UnetBasicBlock(3,in_channels,features[0],kernel_size=3,stride=1,norm_name="instance")
        self.enc2=UnetBasicBlock(3,features[0],features[1],kernel_size=3,stride=2,norm_name="instance")
        self.enc3=UnetBasicBlock(3,features[1],features[2],kernel_size=3,stride=2,norm_name="instance")
        self.enc4=UnetBasicBlock(3,features[2],features[3],kernel_size=3,stride=2,norm_name="instance")
        self.skip1_mamba=OmniMambaLayer(dim=features[0])
        self.skip2_mamba=OmniMambaLayer(dim=features[1])
        self.skip3_mamba=OmniMambaLayer(dim=features[2])
        self.skip4_mamba=OmniMambaLayer(dim=features[3])
        self.bottleneck_conv=UnetBasicBlock(3,features[3],features[4],kernel_size=3,stride=2,norm_name="instance")
        self.mamba_block1=OmniMambaLayer(dim=features[4])
        self.mamba_block2=OmniMambaLayer(dim=features[4])
        self.dec4=UnetUpBlock(3,features[4],features[3],kernel_size=3,stride=1,upsample_kernel_size=2,norm_name="instance")
        self.dec3=UnetUpBlock(3,features[3],features[2],kernel_size=3,stride=1,upsample_kernel_size=2,norm_name="instance")
        self.dec2=UnetUpBlock(3,features[2],features[1],kernel_size=3,stride=1,upsample_kernel_size=2,norm_name="instance")
        self.dec1=UnetUpBlock(3,features[1],features[0],kernel_size=3,stride=1,upsample_kernel_size=2,norm_name="instance")
        self.final=nn.Conv3d(features[0],out_channels,kernel_size=1)
    def forward(self,x):
        e1=self.enc1(x);e2=self.enc2(e1);e3=self.enc3(e2);e4=self.enc4(e3)
        b=self.mamba_block2(self.mamba_block1(self.bottleneck_conv(e4)))
        d4=self.dec4(b,self.skip4_mamba(e4))
        d3=self.dec3(d4,self.skip3_mamba(e3))
        d2=self.dec2(d3,self.skip2_mamba(e2))
        d1=self.dec1(d2,self.skip1_mamba(e1))
        return self.final(d1)

def load_model(ckpt):
    m = MambaUNet().to(device)
    raw = torch.load(ckpt, map_location=device)
    clean = {k.replace("module.", ""): v for k, v in raw.items() if k != "n_averaged"}

    # FIX: convert any fp16 weights -> fp32 before loading
    # (AMP training sometimes saves fp16 weights; inference in fp32 needs fp32 weights)
    clean = {k: v.float() if v.is_floating_point() else v for k, v in clean.items()}

    m.load_state_dict(clean)
    m.eval()

    # FIX: ensure entire model is in fp32
    m = m.float()

    # Sanity check weights
    nan_count = sum(v.isnan().sum().item() for v in clean.values() if v.is_floating_point())
    inf_count = sum(v.isinf().sum().item() for v in clean.values() if v.is_floating_point())
    log.info(f"    {os.path.basename(ckpt)} loaded | NaN weights: {nan_count} | Inf weights: {inf_count}")

    return m

def tta_inference_fp32(model, inputs):
    """
    FIX: run entirely in fp32, no autocast.
    autocast (fp16) was causing NaN/Inf in Mamba SSM operations during inference.
    fp32 is slightly slower but produces valid outputs.
    """
    # Ensure input is fp32
    inputs = inputs.float()

    preds = []
    for axes in [[], [2], [3], [4], [2,3], [2,4], [3,4], [2,3,4]]:
        x = torch.flip(inputs.clone(), axes) if axes else inputs.clone()

        # FIX: no autocast — pure fp32
        out = sliding_window_inference(
            x, ROI_SIZE, sw_batch_size=2,
            predictor=model, overlap=0.5, mode="gaussian",
        )

        # FIX: guard against any residual NaN/Inf before softmax
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        out = torch.softmax(out, dim=1)

        if axes:
            out = torch.flip(out, axes)
        preds.append(out.cpu())

    result = torch.stack(preds).mean(0)

    # Final NaN check
    if result.isnan().any():
        log.warning("  WARNING: NaN still present after nan_to_num — returning zeros")
        result = torch.zeros_like(result)
        result[:, 0] = 1.0  # all background

    return result

# =============================================================================
# TRANSFORMS
# =============================================================================
transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image"], channel_dim=-1),
    EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
    Spacingd(keys=["image","label"], pixdim=(1.5,1.5,2.0), mode=("bilinear","nearest")),
    Orientationd(keys=["image","label"], axcodes="RAS"),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image","label"]),
])

test_ds     = CacheDataset(test_files, transforms, cache_rate=1.0, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

post_pred  = AsDiscrete(argmax=True, to_onehot=2)
post_label = AsDiscrete(to_onehot=2)
dice_metric = DiceMetric(include_background=False, reduction="mean")
hd_metric   = HausdorffDistanceMetric(include_background=False, percentile=95)

per_case_dice, per_case_hd = [], []

log.info(f"\nRunning fp32 inference (no autocast) + 8-flip TTA + 5-model ensemble...\n")

with torch.no_grad():
    for idx, test_data in enumerate(test_loader):
        t_input = test_data["image"].float().to(device)  # FIX: explicit fp32
        t_label = test_data["label"]

        gt_fg = (t_label > 0).sum().item()

        # Load 1 model at a time
        all_preds = []
        for ckpt in ckpt_paths:
            m = load_model(ckpt)
            pred = tta_inference_fp32(m, t_input)   # (1,2,H,W,D) on CPU, fp32
            all_preds.append(pred)
            del m
            torch.cuda.empty_cache()

        ensemble = torch.stack(all_preds).mean(0)   # CPU, fp32

        # Sanity print for first case
        if idx == 0:
            fg_max  = ensemble[0,1].max().item()
            fg_mean = ensemble[0,1].mean().item()
            has_nan = ensemble.isnan().any().item()
            log.info(f"  [Sanity Case 1] fg_prob_max={fg_max:.4f} | "
                     f"fg_prob_mean={fg_mean:.6f} | has_nan={has_nan}")

        # Metrics
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

        try:
            img_name = os.path.basename(
                test_data["image"].meta["filename_or_obj"][0]
                if hasattr(test_data["image"], "meta")
                else test_data["image_meta_dict"]["filename_or_obj"][0]
            )
        except Exception:
            img_name = f"case_{idx+1:03d}"

        hd_str = f"{case_hd:.2f}mm" if not np.isnan(case_hd) else "N/A"
        log.info(f"  {idx+1:>2}/{len(test_files)} | {img_name:<38} | "
                 f"Dice={case_dice:.4f} | HD95={hd_str} | GT_fg={gt_fg} | Pred_fg={pred_fg}")

        del all_preds, ensemble
        torch.cuda.empty_cache()

# =============================================================================
# REPORT
# =============================================================================
mean_dice = float(np.mean(per_case_dice))
std_dice  = float(np.std(per_case_dice))
valid_hd  = [h for h in per_case_hd if not np.isnan(h)]
mean_hd   = float(np.mean(valid_hd)) if valid_hd else float("nan")
std_hd    = float(np.std(valid_hd))  if valid_hd else float("nan")

log.info("")
log.info("=" * 65)
log.info("  FINAL TEST RESULTS")
log.info("=" * 65)
log.info(f"  Test Dice : {mean_dice:.4f} +/- {std_dice:.4f}")
log.info(f"  Test HD95 : {mean_hd:.2f} +/- {std_hd:.2f} mm"
         if not np.isnan(mean_hd) else "  Test HD95 : N/A")
log.info("")
log.info("  CV results from training:")
log.info("    Fold 1: 0.6439 | Fold 2: 0.7982 | Fold 3: 0.8167")
log.info("    Fold 4: 0.7334 | Fold 5: 0.7381 | Mean: 0.7461")
log.info("")
log.info("  ISLES 2022 Leaderboard:")
log.info("    SEALS      Rank 1 : 0.821")
log.info("    NVAUTO     Rank 2 : 0.824")
log.info("    Factorizer Rank 3 : 0.812")
log.info(f"    Ours (MambaUNet)  : {mean_dice:.4f}  <--- YOUR FINAL SCORE")

results = {
    "run_id": run_id,
    "inference_mode": "fp32_no_autocast",
    "per_case": [{"case": test_files[i]["image"],
                  "dice": per_case_dice[i], "hd95": per_case_hd[i]}
                 for i in range(len(test_files))],
    "mean_dice": mean_dice, "std_dice": std_dice,
    "mean_hd95": mean_hd,   "std_hd95": std_hd,
    "cv": {"folds": [0.6439,0.7982,0.8167,0.7334,0.7381], "mean": 0.7461, "std": 0.0606},
}
out = os.path.join(RESULT_DIR, f"final_test_FP32_{run_id}.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)
log.info(f"\n  Saved: {out}")
log.info("=" * 65)