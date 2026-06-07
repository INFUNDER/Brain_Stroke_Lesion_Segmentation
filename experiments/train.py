import os
import glob
import torch
import numpy as np
import monai
from monai.utils import set_determinism
import warnings
warnings.filterwarnings("ignore")
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    Orientationd, NormalizeIntensityd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, EnsureTyped, SpatialPadd
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.networks.nets import UNETR
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric

# =========================
# CONFIGURATION
# =========================
DATA_DIR = "ISLES2022_Formatted"
ROI_SIZE = (96, 96, 96)
BATCH_SIZE = 2
MAX_EPOCHS = 100
VAL_INTERVAL = 2
NUM_WORKERS = 4

set_determinism(seed=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"MONAI Version: {monai.__version__}", flush=True)
print(f"Using Device: {device}", flush=True)

# =========================
# DATA LOADING
# =========================
train_images = sorted(glob.glob(os.path.join(DATA_DIR, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(DATA_DIR, "labelsTr", "*.nii.gz")))

assert len(train_images) > 0, "No training images found"
assert len(train_images) == len(train_labels), "Images and labels count mismatch"

data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(train_images, train_labels)]
split = int(0.8 * len(data_dicts))
train_files, val_files = data_dicts[:split], data_dicts[split:]

print(f"Train samples: {len(train_files)} | Val samples: {len(val_files)}", flush=True)

# =========================
# TRANSFORMS
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
        image_key="image",
        image_threshold=0,
    ),
    RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.1),
    RandRotate90d(keys=["image", "label"], prob=0.1, max_k=3),
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
train_ds = CacheDataset(train_files, train_transforms, cache_rate=1.0,
                        num_workers=NUM_WORKERS)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=NUM_WORKERS)

val_ds = CacheDataset(val_files, val_transforms, cache_rate=1.0,
                      num_workers=NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=1,
                        shuffle=False, num_workers=NUM_WORKERS)

# =========================
# MODEL
# =========================
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
).to(device)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
dice_metric = DiceMetric(include_background=False, reduction="mean")

# =========================
# TRAINING LOOP
# =========================
best_metric = -1

for epoch in range(MAX_EPOCHS):
    model.train()
    epoch_loss = 0

    for batch_data in train_loader:
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

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

                val_outputs = sliding_window_inference(
                    val_inputs, ROI_SIZE, 4, model
                )
                val_outputs = [o.argmax(dim=0, keepdim=True)
                               for o in decollate_batch(val_outputs)]
                val_labels = decollate_batch(val_labels)
                dice_metric(val_outputs, val_labels)

            metric = dice_metric.aggregate().item()
            dice_metric.reset()

            if metric > best_metric:
                best_metric = metric
                torch.save(model.state_dict(), "best_metric_model.pth")
                print(f"⭐ New best Dice: {metric:.4f}", flush=True)
            else:
                print(f"Val Dice: {metric:.4f}", flush=True)
