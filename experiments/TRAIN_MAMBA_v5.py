# =============================================================================
# MAMBA-UNET v5 — OPTIMISED FINAL PIPELINE
# Changes from v4:
#   1. MAX_EPOCHS 400 -> 600  (128^3 needs more epochs to converge)
#   2. Elastic deformation prob 0.3 -> 0.2  (was slowing convergence)
#   3. VAL_INTERVAL 2 -> 5  (saves time at 600 epochs, ~same val frequency)
#   4. LR slightly higher warmup: start_factor 0.01 -> 0.05
#   5. CC min_voxels 10 -> 20  (more aggressive false-positive removal)
#   6. Early stopping: if no improvement for 80 epochs, stop fold early
# =============================================================================

import os, sys, gc, glob, time, json, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import monai

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datetime import datetime
from mamba_ssm import Mamba
from sklearn.model_selection import KFold
from scipy import ndimage

from monai.utils import set_determinism
import warnings; warnings.filterwarnings("ignore")

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, NormalizeIntensityd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, EnsureTyped, SpatialPadd,
    RandGaussianNoised, RandScaleIntensityd, RandShiftIntensityd,
    RandGaussianSmoothd, Rand3DElasticd, AsDiscrete,
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.networks.blocks import UnetBasicBlock, UnetUpBlock
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel
from torch.cuda.amp import GradScaler, autocast

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR      = "/home/ronit.28010/Brain_Stroke/ISLES2022_Formatted"
OUTPUT_DIR    = os.path.join(os.environ.get("PBS_O_WORKDIR", "."), "mamba_results_v5")

ROI_SIZE          = (128, 128, 128)
BATCH_SIZE        = 1
GRAD_ACCUM        = 2
MAX_EPOCHS        = 600          # v4 was 400 — 128^3 needs more
EARLY_STOP_PATIENCE = 80         # stop fold if no improvement for 80 epochs
VAL_INTERVAL      = 5            # validate every 5 epochs (was 2 — saves time)
NUM_WORKERS       = 4
WARMUP_EPOCHS     = 15           # slightly longer warmup for stability
EMA_DECAY         = 0.999
N_FOLDS           = 5
TEST_FRACTION     = 0.10
BOUNDARY_W        = 0.2
MIN_LESION_VOXELS = 20           # v4 was 10 — more aggressive FP removal
SEED              = 42

# =============================================================================
# DIRS + LOGGING
# =============================================================================
LOG_DIR    = os.path.join(OUTPUT_DIR, "logs")
CKPT_DIR   = os.path.join(OUTPUT_DIR, "checkpoints")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
for d in [OUTPUT_DIR, LOG_DIR, CKPT_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
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

def vram():
    if torch.cuda.is_available():
        u = torch.cuda.memory_allocated() / 1e9
        t = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{u:.1f}/{t:.1f}GB"
    return "N/A"

set_determinism(seed=SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

banner("MAMBA-UNET v5 PIPELINE")
log.info(f"Run ID      : {run_id}")
log.info(f"MONAI       : {monai.__version__}")
log.info(f"PyTorch     : {torch.__version__}")
log.info(f"Device      : {device}")
if torch.cuda.is_available():
    log.info(f"GPU         : {torch.cuda.get_device_name(0)}")
    log.info(f"VRAM        : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
log.info(f"ROI         : {ROI_SIZE}")
log.info(f"Epochs      : {MAX_EPOCHS} (early stop patience={EARLY_STOP_PATIENCE})")
log.info(f"Val interval: every {VAL_INTERVAL} epochs")
log.info(f"Eff batch   : {BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}")
log.info(f"CC min vox  : {MIN_LESION_VOXELS}")
log.info(f"Output      : {OUTPUT_DIR}")

# =============================================================================
# DATA
# =============================================================================
if not os.path.exists(DATA_DIR):
    log.error(f"DATA_DIR not found: {DATA_DIR}"); sys.exit(1)

train_images = sorted(glob.glob(os.path.join(DATA_DIR, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(DATA_DIR, "labelsTr", "*.nii.gz")))
assert len(train_images) == len(train_labels)
if len(train_images) == 0:
    log.error("No images found"); sys.exit(1)

all_data = [{"image": i, "label": l} for i, l in zip(train_images, train_labels)]
rng      = np.random.default_rng(seed=SEED)
perm     = rng.permutation(len(all_data))
n_test   = max(1, int(TEST_FRACTION * len(all_data)))
test_files = [all_data[i] for i in perm[:n_test]]
cv_data    = [all_data[i] for i in perm[n_test:]]

test_path_file = os.path.join(OUTPUT_DIR, "test_set_paths.txt")
with open(test_path_file, "w") as f:
    for d in test_files:
        f.write(f"{d['image']}  |  {d['label']}\n")

banner("DATASET SPLIT")
log.info(f"Total        : {len(all_data)}")
log.info(f"Held-out test: {len(test_files)}")
log.info(f"CV pool      : {len(cv_data)}  ({N_FOLDS} folds)")

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
    Rand3DElasticd(                    # v5: prob reduced 0.3->0.2
        keys=["image", "label"],
        sigma_range=(3, 5),
        magnitude_range=(10, 20),
        prob=0.2,
        mode=("bilinear", "nearest"),
    ),
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
test_transforms = val_transforms

# =============================================================================
# CC POST-PROCESSING
# =============================================================================
def remove_small_components(pred_np, min_voxels=MIN_LESION_VOXELS):
    if pred_np.sum() == 0:
        return pred_np
    labeled, n = ndimage.label(pred_np)
    cleaned = np.zeros_like(pred_np)
    for cid in range(1, n + 1):
        if (labeled == cid).sum() >= min_voxels:
            cleaned[labeled == cid] = 1
    return cleaned

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
# MODEL
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
        lf = self.local_conv(x)
        zf = x.flatten(2).transpose(1, 2)
        oz = self.mamba_z(self.norm(zf)).transpose(1, 2).view(b, c, h, w, d)
        yp = x.permute(0,1,3,4,2).contiguous()
        oy = self.mamba_y(self.norm(yp.flatten(2).transpose(1,2))).transpose(1,2).view(b,c,w,d,h).permute(0,1,4,2,3)
        xp = x.permute(0,1,4,2,3).contiguous()
        ox = self.mamba_x(self.norm(xp.flatten(2).transpose(1,2))).transpose(1,2).view(b,c,d,h,w).permute(0,1,3,4,2)
        return oz + oy + ox + lf + x

class MambaUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, features=(32,64,128,256,512)):
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
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3)
        b  = self.mamba_block2(self.mamba_block1(self.bottleneck_conv(e4)))
        d4 = self.dec4(b,  self.skip4_mamba(e4))
        d3 = self.dec3(d4, self.skip3_mamba(e3))
        d2 = self.dec2(d3, self.skip2_mamba(e2))
        d1 = self.dec1(d2, self.skip1_mamba(e1))
        return self.final(d1)

# =============================================================================
# INFERENCE — fp32 always (fixes NaN from v3)
# =============================================================================
def fp32_inference(model, inputs, roi_size):
    inputs = inputs.float()
    out = sliding_window_inference(
        inputs, roi_size, sw_batch_size=2,
        predictor=model, overlap=0.5, mode="gaussian",
    )
    out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
    return torch.softmax(out, dim=1)

def tta_fp32(model, inputs, roi_size):
    preds = []
    for axes in [[], [2], [3], [4], [2,3], [2,4], [3,4], [2,3,4]]:
        x   = torch.flip(inputs.clone(), axes) if axes else inputs.clone()
        out = fp32_inference(model, x, roi_size)
        if axes:
            out = torch.flip(out, axes)
        preds.append(out.cpu())
    return torch.stack(preds).mean(0)

def load_ema_model(ckpt_path):
    m = MambaUNet().to(device)
    raw   = torch.load(ckpt_path, map_location=device)
    clean = {k.replace("module.", ""): v.float() if torch.is_floating_point(v) else v
             for k, v in raw.items() if k != "n_averaged"}
    m.load_state_dict(clean)
    m.eval()
    return m.float()

# =============================================================================
# TRAIN ONE FOLD — with early stopping
# =============================================================================
def train_fold(fold_idx, train_files, val_files):
    fold_start = time.time()
    banner(f"FOLD {fold_idx+1}/{N_FOLDS}  train={len(train_files)}  val={len(val_files)}")
    log.info(f"  VRAM at start: {vram()}")

    fh = logging.FileHandler(os.path.join(LOG_DIR, f"fold{fold_idx+1}_{run_id}.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
    log.addHandler(fh)

    train_ds = CacheDataset(train_files, train_transforms, cache_rate=1.0, num_workers=NUM_WORKERS)
    val_ds   = CacheDataset(val_files,   val_transforms,   cache_rate=1.0, num_workers=NUM_WORKERS)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model     = MambaUNet().to(device)
    ema_model = AveragedModel(model,
                              multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(EMA_DECAY))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Params: {n_params/1e6:.2f}M  | VRAM: {vram()}")

    dice_focal    = DiceFocalLoss(to_onehot_y=True, softmax=True, squared_pred=True)
    boundary_loss = BoundaryLoss()
    scaler        = GradScaler()
    optimizer     = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler     = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.05, end_factor=1.0, total_iters=WARMUP_EPOCHS),
        CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6),
    ], milestones=[WARMUP_EPOCHS])

    post_pred   = AsDiscrete(argmax=True, to_onehot=2)
    post_label  = AsDiscrete(to_onehot=2)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    best_metric    = -1
    no_improve_cnt = 0          # early stopping counter
    ckpt_path      = os.path.join(CKPT_DIR, f"best_fold{fold_idx+1}.pth")
    history        = []

    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            with autocast():
                outputs = model(inputs)
                probs   = torch.softmax(outputs, dim=1)
                loss    = (dice_focal(outputs, labels)
                           + BOUNDARY_W * boundary_loss(probs, labels)) / GRAD_ACCUM

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema_model.update_parameters(model)

            epoch_loss += loss.item() * GRAD_ACCUM

        epoch_loss /= len(train_loader)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        log.info(f"  [F{fold_idx+1}] Ep {epoch+1:>3}/{MAX_EPOCHS} | "
                 f"loss={epoch_loss:.4f} | lr={lr:.2e} | vram={vram()}")

        if (epoch + 1) % VAL_INTERVAL == 0:
            ema_model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    vi = val_data["image"].float().to(device)
                    vl = val_data["label"]
                    vo = fp32_inference(ema_model, vi, ROI_SIZE).cpu()
                    vo_list = [post_pred(p)  for p in decollate_batch(vo)]
                    vl_list = [post_label(l) for l in decollate_batch(vl)]
                    dice_metric(vo_list, vl_list)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                history.append({"epoch": epoch+1, "val_dice": metric, "loss": epoch_loss})

                if metric > best_metric:
                    best_metric    = metric
                    no_improve_cnt = 0
                    torch.save(ema_model.state_dict(), ckpt_path)
                    log.info(f"  [F{fold_idx+1}] *** BEST {metric:.4f} -> saved")
                else:
                    no_improve_cnt += VAL_INTERVAL
                    log.info(f"  [F{fold_idx+1}] Val Dice: {metric:.4f}  "
                             f"(best: {best_metric:.4f}, no-improve: {no_improve_cnt}/{EARLY_STOP_PATIENCE})")

                    if no_improve_cnt >= EARLY_STOP_PATIENCE:
                        log.info(f"  [F{fold_idx+1}] Early stopping triggered at epoch {epoch+1}")
                        break

            torch.cuda.empty_cache()
        else:
            continue
        break  # break outer loop if early stop triggered

    with open(os.path.join(RESULT_DIR, f"fold{fold_idx+1}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    elapsed = (time.time() - fold_start) / 60
    log.info(f"  [F{fold_idx+1}] Done {elapsed:.1f}min | Best Val Dice: {best_metric:.4f}")

    del model, ema_model, optimizer, scheduler, scaler
    del train_ds, val_ds, train_loader, val_loader
    gc.collect(); torch.cuda.empty_cache()
    log.info(f"  VRAM after cleanup: {vram()}")
    log.removeHandler(fh); fh.close()

    return best_metric, ckpt_path

# =============================================================================
# FINAL TEST EVALUATION
# =============================================================================
def evaluate_test_set(ckpt_paths, test_files):
    banner("FINAL TEST SET EVALUATION")
    log.info(f"  Cases: {len(test_files)} | Models: {len(ckpt_paths)} | VRAM: {vram()}")

    test_ds     = CacheDataset(test_files, test_transforms, cache_rate=1.0, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    post_pred   = AsDiscrete(argmax=True, to_onehot=2)
    post_label  = AsDiscrete(to_onehot=2)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd_metric   = HausdorffDistanceMetric(include_background=False, percentile=95)

    per_case = []

    with torch.no_grad():
        for idx, test_data in enumerate(test_loader):
            t_input = test_data["image"].float().to(device)
            t_label = test_data["label"]
            gt_fg   = (t_label > 0).sum().item()

            all_preds = []
            for ckpt in ckpt_paths:
                m    = load_ema_model(ckpt)
                pred = tta_fp32(m, t_input, ROI_SIZE)
                all_preds.append(pred)
                del m; torch.cuda.empty_cache()

            ensemble = torch.stack(all_preds).mean(0)

            # CC post-processing
            pred_binary = (ensemble[0, 1].numpy() >= 0.5).astype(np.uint8)
            pred_clean  = remove_small_components(pred_binary)
            pred_fg     = int(pred_clean.sum())

            pred_tensor = torch.from_numpy(pred_clean).long()
            pred_oh     = torch.stack([1 - pred_tensor, pred_tensor], dim=0).unsqueeze(0)
            label_list  = [post_label(l) for l in decollate_batch(t_label)]
            pred_list   = [pred_oh[0]]

            dice_metric(pred_list, label_list)
            case_dice = dice_metric.aggregate().item()
            dice_metric.reset()

            if gt_fg > 0 and pred_fg > 0:
                hd_metric(pred_list, label_list)
                case_hd = hd_metric.aggregate().item()
                hd_metric.reset()
            else:
                case_hd = float("nan")

            try:
                img_name = os.path.basename(
                    test_data["image"].meta["filename_or_obj"][0]
                    if hasattr(test_data["image"], "meta")
                    else test_data["image_meta_dict"]["filename_or_obj"][0]
                )
            except Exception:
                img_name = f"case_{idx+1:03d}"

            per_case.append({"name": img_name, "dice": case_dice,
                             "hd95": case_hd, "gt_fg": gt_fg, "pred_fg": pred_fg})

            hd_str = f"{case_hd:.2f}mm" if not np.isnan(case_hd) else "N/A"
            log.info(f"  {idx+1:>2}/{len(test_files)} | {img_name:<38} | "
                     f"Dice={case_dice:.4f} | HD95={hd_str} | GT={gt_fg} | Pred={pred_fg}")

            del all_preds, ensemble; torch.cuda.empty_cache()

    all_dice      = [c["dice"] for c in per_case]
    nonempty_dice = [c["dice"] for c in per_case if c["gt_fg"] > 0]
    valid_hd      = [c["hd95"] for c in per_case if not np.isnan(c["hd95"])]
    empty_count   = sum(1 for c in per_case if c["gt_fg"] == 0)

    return per_case, all_dice, nonempty_dice, valid_hd, empty_count

# =============================================================================
# MAIN
# =============================================================================
def main():
    total_start = time.time()

    kf     = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds  = list(kf.split(cv_data))
    fold_scores, ckpt_paths = [], []

    for fold_idx in range(N_FOLDS):
        train_idx, val_idx = folds[fold_idx]
        best_dice, ckpt = train_fold(
            fold_idx,
            [cv_data[i] for i in train_idx],
            [cv_data[i] for i in val_idx],
        )
        fold_scores.append(best_dice)
        ckpt_paths.append(ckpt)

    cv_mean = float(np.mean(fold_scores))
    cv_std  = float(np.std(fold_scores))

    banner("CROSS-VALIDATION SUMMARY")
    for i, s in enumerate(fold_scores):
        log.info(f"  Fold {i+1}: {s:.4f}")
    log.info(f"  CV: {cv_mean:.4f} +/- {cv_std:.4f}")

    per_case, all_dice, nonempty_dice, valid_hd, empty_count = \
        evaluate_test_set(ckpt_paths, test_files)

    mean_all      = float(np.mean(all_dice))
    mean_nonempty = float(np.mean(nonempty_dice)) if nonempty_dice else float("nan")
    std_nonempty  = float(np.std(nonempty_dice))  if nonempty_dice else float("nan")
    mean_hd       = float(np.mean(valid_hd)) if valid_hd else float("nan")
    std_hd        = float(np.std(valid_hd))  if valid_hd else float("nan")
    total_time    = (time.time() - total_start) / 60

    banner("FINAL RESULTS")
    log.info(f"  CV Dice (5-fold)         : {cv_mean:.4f} +/- {cv_std:.4f}")
    log.info(f"  Test Dice (all 25)       : {mean_all:.4f}")
    log.info(f"  Test Dice (non-empty GT) : {mean_nonempty:.4f} +/- {std_nonempty:.4f}  [{len(nonempty_dice)} cases]")
    log.info(f"  Test HD95 (non-empty)    : {mean_hd:.2f} +/- {std_hd:.2f} mm"
             if not np.isnan(mean_hd) else "  Test HD95: N/A")
    log.info(f"  Empty-GT cases excluded  : {empty_count}")
    log.info("")
    log.info("  ISLES 2022 Leaderboard:")
    log.info("    SEALS      Rank 1 : 0.821")
    log.info("    NVAUTO     Rank 2 : 0.824")
    log.info("    Factorizer Rank 3 : 0.812")
    log.info(f"    Ours (MambaUNet)  : {mean_nonempty:.4f}  <--- YOUR SCORE")
    log.info(f"\n  Runtime: {total_time:.1f} min")

    results = {
        "run_id": run_id,
        "config": {
            "roi_size": list(ROI_SIZE), "max_epochs": MAX_EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "val_interval": VAL_INTERVAL, "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM, "warmup_epochs": WARMUP_EPOCHS,
            "ema_decay": EMA_DECAY, "boundary_weight": BOUNDARY_W,
            "min_lesion_voxels": MIN_LESION_VOXELS, "seed": SEED,
        },
        "cv": {"fold_scores": fold_scores, "mean": cv_mean, "std": cv_std},
        "test": {
            "per_case": per_case,
            "mean_dice_all": mean_all,
            "mean_dice_nonempty": mean_nonempty,
            "std_dice_nonempty": std_nonempty,
            "mean_hd95": mean_hd, "std_hd95": std_hd,
            "empty_gt_cases": empty_count,
        },
        "runtime_minutes": total_time,
    }

    rp = os.path.join(RESULT_DIR, f"final_results_{run_id}.json")
    sp = os.path.join(RESULT_DIR, f"summary_{run_id}.txt")

    with open(rp, "w") as f:
        json.dump(results, f, indent=2)

    with open(sp, "w") as f:
        f.write(f"MAMBA-UNET v5  |  {run_id}\n{'='*50}\n\n")
        f.write("CROSS-VALIDATION\n")
        for i, s in enumerate(fold_scores):
            f.write(f"  Fold {i+1}: {s:.4f}\n")
        f.write(f"  Mean: {cv_mean:.4f} +/- {cv_std:.4f}\n\n")
        f.write("TEST SET\n")
        f.write(f"  Dice (all)       : {mean_all:.4f}\n")
        f.write(f"  Dice (non-empty) : {mean_nonempty:.4f} +/- {std_nonempty:.4f}\n")
        f.write(f"  HD95             : {mean_hd:.2f} +/- {std_hd:.2f} mm\n")
        f.write(f"  Empty GT cases   : {empty_count}\n\n")
        f.write("LEADERBOARD\n")
        f.write("  SEALS   Rank 1 : 0.821\n")
        f.write("  NVAUTO  Rank 2 : 0.824\n")
        f.write("  Factorizer R3  : 0.812\n")
        f.write(f"  Ours           : {mean_nonempty:.4f}\n")
        f.write(f"\nRuntime: {total_time:.1f} min\n")

    log.info(f"  Results: {rp}")
    log.info(f"  Summary: {sp}")
    banner("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()