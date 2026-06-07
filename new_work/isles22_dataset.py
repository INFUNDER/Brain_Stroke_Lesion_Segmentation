"""
ISLES 2022 - Full Data Loading Pipeline
========================================
Handles the raw BIDS layout from your tree:
  sub-strokecaseXXXX/ses-0001/anat/*_FLAIR.nii.gz
  sub-strokecaseXXXX/ses-0001/dwi/*_dwi.nii.gz
  sub-strokecaseXXXX/ses-0001/dwi/*_adc.nii.gz
  derivatives/sub-strokecaseXXXX/ses-0001/*_msk.nii.gz

Key steps performed:
  1. Path discovery with duplicate .nii / .nii.gz deduplication
  2. Per-volume z-score normalization (per modality, brain-masked)
  3. Rigid FLAIR→DWI registration via SimpleITK (NOT MNI - preserves small lesions)
  4. Physics-guided DWI×ADC gating channel (pre-computed, stored as 4th channel)
  5. 3D patch sampling with lesion-centered oversampling (50/50)
  6. MONAI-based augmentation pipeline tuned for stroke MRI
  7. 5-fold cross-validation splits (stratified by lesion volume)
  8. DataLoader ready for SMU-Net training
"""

import os
import glob
import json
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import SimpleITK as sitk

# MONAI for medical augmentations
from monai.transforms import (
    Compose,
    RandFlipd,
    RandAffined,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    NormalizeIntensityd,
    ToTensord,
    RandCropByPosNegLabeld,
    SpatialPadd,
    CenterSpatialCropd,
    Spacingd,
    Orientationd,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
)
from monai.data import CacheDataset, PersistentDataset
from sklearn.model_selection import KFold


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PATH DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def discover_cases(data_root: str) -> List[Dict[str, str]]:
    """
    Walk the BIDS tree and return a list of dicts, one per valid case.
    Skips cases missing any of the 4 required files.
    Handles the .nii / .nii.gz duplicate issue (prefers .nii.gz).
    """
    data_root = Path(data_root)
    cases = []
    missing_log = []

    subject_dirs = sorted(data_root.glob("sub-strokecase*"))
    # Filter to only subject dirs (not derivatives)
    subject_dirs = [d for d in subject_dirs if d.is_dir() and "deriv" not in str(d)]

    for subj_dir in subject_dirs:
        subj_id = subj_dir.name  # e.g. sub-strokecase0001
        ses_dir = subj_dir / "ses-0001"

        if not ses_dir.exists():
            missing_log.append(f"{subj_id}: missing ses-0001")
            continue

        # ── FLAIR ──────────────────────────────────────────────────────────
        flair_path = _find_nii(ses_dir / "anat", f"{subj_id}_ses-0001_FLAIR")
        # ── DWI ────────────────────────────────────────────────────────────
        dwi_path   = _find_nii(ses_dir / "dwi",  f"{subj_id}_ses-0001_dwi")
        # ── ADC ────────────────────────────────────────────────────────────
        adc_path   = _find_nii(ses_dir / "dwi",  f"{subj_id}_ses-0001_adc")
        # ── MASK ───────────────────────────────────────────────────────────
        mask_path  = _find_nii(
            data_root / "derivatives" / subj_id / "ses-0001",
            f"{subj_id}_ses-0001_msk"
        )

        if None in (flair_path, dwi_path, adc_path, mask_path):
            missing_log.append(
                f"{subj_id}: flair={flair_path is not None} "
                f"dwi={dwi_path is not None} "
                f"adc={adc_path is not None} "
                f"mask={mask_path is not None}"
            )
            continue

        cases.append({
            "subject":  subj_id,
            "flair":    str(flair_path),
            "dwi":      str(dwi_path),
            "adc":      str(adc_path),
            "mask":     str(mask_path),
        })

    print(f"[discover_cases] Found {len(cases)} complete cases "
          f"({len(missing_log)} incomplete)")
    if missing_log:
        print("  Skipped:", "\n  ".join(missing_log))

    return cases


def _find_nii(directory: Path, stem: str) -> Optional[Path]:
    """
    Find stem.nii.gz (preferred) or stem.nii in directory.
    Returns None if neither exists.
    """
    gz = directory / f"{stem}.nii.gz"
    plain = directory / f"{stem}.nii"
    if gz.exists():
        return gz
    if plain.exists():
        return plain
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PREPROCESSING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def register_flair_to_dwi(flair_path: str, dwi_path: str) -> np.ndarray:
    """
    Rigid registration of FLAIR into DWI space using SimpleITK.
    Returns the resampled FLAIR array in DWI voxel space.

    Why rigid (not affine)? MRI acquisitions of the same patient are
    already skull-stripped and close in space. Affine would distort
    lesion boundaries. Rigid is sufficient and fast (~1-2s per case).
    """
    fixed  = sitk.ReadImage(dwi_path,   sitk.sitkFloat32)
    moving = sitk.ReadImage(flair_path, sitk.sitkFloat32)

    # Initialize with center-of-mass alignment
    initial_tf = sitk.CenteredTransformInitializer(
        fixed, moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.01)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial_tf, inPlace=False)

    final_tf = reg.Execute(fixed, moving)

    flair_registered = sitk.Resample(
        moving, fixed, final_tf,
        sitk.sitkLinear, 0.0,
        moving.GetPixelID()
    )
    return sitk.GetArrayFromImage(flair_registered)  # (D, H, W)


def zscore_normalize(volume: np.ndarray,
                     brain_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Z-score normalization within the brain mask.
    If no mask provided, uses non-zero voxels (skull-stripped data).
    Clips to [-5, 5] to handle outlier voxels from acquisition artifacts.
    """
    if brain_mask is None:
        brain_mask = volume > 0

    brain_voxels = volume[brain_mask]
    if brain_voxels.size == 0:
        return volume.astype(np.float32)

    mean = brain_voxels.mean()
    std  = brain_voxels.std()
    if std < 1e-8:
        std = 1.0

    normalized = (volume - mean) / std
    normalized = np.clip(normalized, -5.0, 5.0)

    # Zero out background (outside brain)
    normalized[~brain_mask] = 0.0
    return normalized.astype(np.float32)


def compute_physics_gate(dwi: np.ndarray,
                          adc: np.ndarray) -> np.ndarray:
    """
    Physics-guided DWI×ADC gating channel.

    True ischemic tissue = DWI bright + ADC dark (restricted diffusion).
    T2 shine-through = DWI bright + ADC normal/bright (NOT ischemic).

    Gate = sigmoid(DWI_norm) * (1 - norm(ADC_norm to [0,1]))

    This pre-computes the physical prior that the model would otherwise
    have to learn from scratch. Reduces false positives significantly.
    """
    # Both inputs are already z-score normalized → sigmoid maps to (0,1)
    dwi_prob  = 1.0 / (1.0 + np.exp(-dwi))          # sigmoid

    # ADC: normalize to [0,1] within brain
    brain = adc != 0
    adc_norm = np.zeros_like(adc)
    if brain.any():
        a_min, a_max = adc[brain].min(), adc[brain].max()
        if a_max > a_min:
            adc_norm[brain] = (adc[brain] - a_min) / (a_max - a_min)

    # Low ADC = high confidence of true ischemia
    gate = dwi_prob * (1.0 - adc_norm)
    return gate.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PREPROCESSED CASE LOADER
# ─────────────────────────────────────────────────────────────────────────────

class ISLES22Preprocessor:
    """
    Loads one case, runs registration + normalization + gating,
    returns a dict with numpy arrays ready for the Dataset.

    Can optionally save/load from a cache directory to avoid
    re-running registration every epoch.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def process(self, case: Dict[str, str]) -> Dict[str, np.ndarray]:
        subj = case["subject"]

        # ── Check cache ──────────────────────────────────────────────────
        if self.cache_dir:
            cache_path = self.cache_dir / f"{subj}.npz"
            if cache_path.exists():
                data = np.load(cache_path)
                return {
                    "image": data["image"],  # (4, D, H, W)
                    "label": data["label"],  # (1, D, H, W)
                    "subject": subj,
                    "lesion_voxels": int(data["lesion_voxels"]),
                }

        # ── Load raw volumes ─────────────────────────────────────────────
        dwi_nib  = nib.load(case["dwi"])
        adc_nib  = nib.load(case["adc"])
        mask_nib = nib.load(case["mask"])

        dwi_arr  = dwi_nib.get_fdata(dtype=np.float32)   # (H, W, D)
        adc_arr  = adc_nib.get_fdata(dtype=np.float32)
        mask_arr = mask_nib.get_fdata(dtype=np.float32)

        # ── Register FLAIR → DWI space ──────────────────────────────────
        # SimpleITK returns (D, H, W), nibabel returns (H, W, D)
        # We'll work in (H, W, D) then transpose to (D, H, W) at the end
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flair_registered = register_flair_to_dwi(case["flair"], case["dwi"])
            # register returns (D, H, W) → transpose to (H, W, D)
            flair_arr = flair_registered.transpose(2, 1, 0)

        # ── Brain mask from DWI (non-zero after skull strip) ─────────────
        brain_mask = dwi_arr > 0

        # ── Z-score normalize each modality ─────────────────────────────
        dwi_norm   = zscore_normalize(dwi_arr,   brain_mask)
        adc_norm   = zscore_normalize(adc_arr,   brain_mask)
        flair_norm = zscore_normalize(flair_arr, brain_mask)

        # ── Physics gate (4th channel) ───────────────────────────────────
        gate = compute_physics_gate(dwi_norm, adc_norm)

        # ── Stack to (4, H, W, D) then → (4, D, H, W) ───────────────────
        image = np.stack([dwi_norm, adc_norm, flair_norm, gate], axis=0)
        image = image.transpose(0, 3, 2, 1)   # (4, D, H, W)
        label = mask_arr.transpose(2, 1, 0)[None]  # (1, D, H, W)
        label = (label > 0.5).astype(np.float32)

        lesion_voxels = int(label.sum())

        # ── Save to cache ─────────────────────────────────────────────────
        if self.cache_dir:
            np.savez_compressed(
                cache_path,
                image=image,
                label=label,
                lesion_voxels=np.array(lesion_voxels)
            )

        return {
            "image":          image,
            "label":          label,
            "subject":        subj,
            "lesion_voxels":  lesion_voxels,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────

PATCH_SIZE = (96, 96, 96)   # fits comfortably on H100 80GB with batch=2

class ISLES22Dataset(Dataset):
    """
    Patch-sampling dataset for training.

    Lesion-centered oversampling: 50% of patches are centered on a random
    foreground (lesion) voxel, 50% are random. This is critical because
    many cases have very small punctiform lesions that would be rarely
    sampled otherwise.
    """

    def __init__(
        self,
        cases:         List[Dict],
        preprocessor:  ISLES22Preprocessor,
        patch_size:    Tuple[int, ...] = PATCH_SIZE,
        patches_per_volume: int = 4,
        mode:          str = "train",    # "train" | "val" | "test"
        pos_fraction:  float = 0.5,
    ):
        self.cases       = cases
        self.prep        = preprocessor
        self.patch_size  = patch_size
        self.patches_per = patches_per_volume
        self.mode        = mode
        self.pos_frac    = pos_fraction

        # ── Augmentation pipeline (train only) ───────────────────────────
        self.train_aug = Compose([
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandAffined(
                keys=["image", "label"],
                mode=["bilinear", "nearest"],
                prob=0.3,
                rotate_range=(0.26, 0.26, 0.26),   # ±15°
                scale_range=(0.1, 0.1, 0.1),
                translate_range=(5, 5, 5),
                padding_mode="border",
            ),
            RandGaussianNoised(keys=["image"], prob=0.15, std=0.05),
            RandScaleIntensityd(keys=["image"], factors=0.2, prob=0.3),
            RandAdjustContrastd(keys=["image"], gamma=(0.8, 1.2), prob=0.2),
            RandGaussianSmoothd(
                keys=["image"], prob=0.1,
                sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)
            ),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ])

    def __len__(self):
        if self.mode == "train":
            return len(self.cases) * self.patches_per
        return len(self.cases)

    def __getitem__(self, idx):
        case_idx = idx % len(self.cases)
        case_data = self.prep.process(self.cases[case_idx])

        image = torch.from_numpy(case_data["image"])  # (4, D, H, W)
        label = torch.from_numpy(case_data["label"])  # (1, D, H, W)

        if self.mode == "train":
            image, label = self._sample_patch(image, label)
            sample = {"image": image, "label": label}
            sample = self.train_aug(sample)
            return sample["image"], sample["label"]

        else:
            # Val/test: return full volume (inference done with sliding window)
            return image, label, case_data["subject"]

    def _sample_patch(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample one patch. With probability pos_frac, center on a lesion voxel.
        Falls back to random if no lesion voxels found in this crop.
        """
        D, H, W = image.shape[1:]
        pd, ph, pw = self.patch_size

        # Ensure volume is large enough; pad if needed
        pad_d = max(0, pd - D)
        pad_h = max(0, ph - H)
        pad_w = max(0, pw - W)
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            image = torch.nn.functional.pad(
                image, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=0
            )
            label = torch.nn.functional.pad(
                label, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=0
            )
            D, H, W = image.shape[1:]

        use_foreground = (
            random.random() < self.pos_frac
            and label.sum() > 0
        )

        if use_foreground:
            # Pick random lesion voxel as patch center
            fg_coords = label[0].nonzero(as_tuple=False)  # (N, 3)
            chosen = fg_coords[random.randint(0, len(fg_coords) - 1)]
            cd, ch, cw = chosen[0].item(), chosen[1].item(), chosen[2].item()

            d0 = max(0, min(cd - pd // 2, D - pd))
            h0 = max(0, min(ch - ph // 2, H - ph))
            w0 = max(0, min(cw - pw // 2, W - pw))
        else:
            d0 = random.randint(0, D - pd)
            h0 = random.randint(0, H - ph)
            w0 = random.randint(0, W - pw)

        patch_img = image[:, d0:d0+pd, h0:h0+ph, w0:w0+pw].clone()
        patch_lbl = label[:, d0:d0+pd, h0:h0+ph, w0:w0+pw].clone()
        return patch_img, patch_lbl


# ─────────────────────────────────────────────────────────────────────────────
# 5.  STRATIFIED 5-FOLD CROSS-VALIDATION SPLITS
# ─────────────────────────────────────────────────────────────────────────────

def make_kfold_splits(
    cases: List[Dict],
    preprocessor: ISLES22Preprocessor,
    n_splits: int = 5,
    seed: int = 42,
) -> List[Dict]:
    """
    Returns list of dicts: [{"train": [...], "val": [...]}, ...]
    Stratified by lesion volume quartile so each fold has a
    representative mix of small/large lesions.
    """
    print("[make_kfold_splits] Computing lesion volumes for stratification...")

    volumes = []
    for case in cases:
        data = preprocessor.process(case)
        volumes.append(data["lesion_voxels"])

    volumes = np.array(volumes)

    # Stratify by quartile
    quartiles = np.digitize(volumes, np.percentile(volumes, [25, 50, 75]))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Use quartile as pseudo-label for stratified split
    # (KFold doesn't support stratification directly, so we do it manually)
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    splits = []
    indices = np.arange(len(cases))
    for train_idx, val_idx in skf.split(indices, quartiles):
        splits.append({
            "train": [cases[i] for i in train_idx],
            "val":   [cases[i] for i in val_idx],
        })

    print(f"[make_kfold_splits] Created {n_splits} folds:")
    for i, s in enumerate(splits):
        print(f"  Fold {i}: train={len(s['train'])} val={len(s['val'])}")

    return splits


# ─────────────────────────────────────────────────────────────────────────────
# 6.  DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    fold:          Dict,               # one element from make_kfold_splits
    preprocessor:  ISLES22Preprocessor,
    patch_size:    Tuple = PATCH_SIZE,
    patches_per:   int = 4,
    batch_size:    int = 2,
    num_workers:   int = 8,            # H100 node typically has 32+ cores
    pin_memory:    bool = True,
    persistent_workers: bool = True,
) -> Tuple[DataLoader, DataLoader]:

    train_ds = ISLES22Dataset(
        cases=fold["train"],
        preprocessor=preprocessor,
        patch_size=patch_size,
        patches_per_volume=patches_per,
        mode="train",
        pos_fraction=0.5,
    )
    val_ds = ISLES22Dataset(
        cases=fold["val"],
        preprocessor=preprocessor,
        patch_size=patch_size,
        mode="val",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,               # full volumes for sliding-window inference
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    print(f"[get_dataloaders] "
          f"Train: {len(train_ds)} patches from {len(fold['train'])} volumes | "
          f"Val: {len(val_ds)} volumes")

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# 7.  QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check(data_root: str, cache_dir: str = "./cache_preprocessed"):
    """
    Run this once to verify the pipeline end-to-end before training.
    Processes 3 cases and prints shapes/stats.
    """
    import time

    print("=" * 60)
    print("ISLES 2022 Data Pipeline - Sanity Check")
    print("=" * 60)

    # Discover cases
    cases = discover_cases(data_root)
    print(f"\nTotal cases: {len(cases)}")
    print(f"First case: {cases[0]}")

    # Preprocess a few
    prep = ISLES22Preprocessor(cache_dir=cache_dir)

    for i, case in enumerate(cases[:3]):
        t0 = time.time()
        data = prep.process(case)
        elapsed = time.time() - t0

        img = data["image"]
        lbl = data["label"]
        print(f"\n[Case {case['subject']}] ({elapsed:.1f}s)")
        print(f"  image shape : {img.shape}  dtype: {img.dtype}")
        print(f"  label shape : {lbl.shape}  dtype: {lbl.dtype}")
        print(f"  lesion voxels: {data['lesion_voxels']}")
        print(f"  image stats  | "
              f"DWI  min={img[0].min():.2f} max={img[0].max():.2f} mean={img[0].mean():.2f}")
        print(f"               | "
              f"ADC  min={img[1].min():.2f} max={img[1].max():.2f} mean={img[1].mean():.2f}")
        print(f"               | "
              f"FLAIR min={img[2].min():.2f} max={img[2].max():.2f} mean={img[2].mean():.2f}")
        print(f"               | "
              f"gate min={img[3].min():.2f} max={img[3].max():.2f} mean={img[3].mean():.2f}")

    # Test patch sampling
    print("\n[Patch sampling test]")
    ds = ISLES22Dataset(
        cases=cases[:5],
        preprocessor=prep,
        patch_size=(96, 96, 96),
        patches_per_volume=2,
        mode="train",
    )
    patch_img, patch_lbl = ds[0]
    print(f"  patch image shape : {patch_img.shape}")
    print(f"  patch label shape : {patch_lbl.shape}")
    print(f"  label positive ratio: {patch_lbl.float().mean():.4f}")

    # Test dataloader
    print("\n[DataLoader test]")
    splits = make_kfold_splits(cases, prep, n_splits=5)
    train_loader, val_loader = get_dataloaders(splits[0], prep, batch_size=2)

    batch_img, batch_lbl = next(iter(train_loader))
    print(f"  batch image shape: {batch_img.shape}")  # (2, 4, 96, 96, 96)
    print(f"  batch label shape: {batch_lbl.shape}")  # (2, 1, 96, 96, 96)
    print(f"  batch image on: {'GPU' if batch_img.is_cuda else 'CPU'}")
    print(f"  batch dtype: {batch_img.dtype}")

    print("\n✓ Pipeline OK — ready for training")
    return cases, splits, prep


# ─────────────────────────────────────────────────────────────────────────────
# 8.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    DATA_ROOT  = sys.argv[1] if len(sys.argv) > 1 else "isless2022dataset/ISLES-2022_notformatted"
    CACHE_DIR  = sys.argv[2] if len(sys.argv) > 2 else "./cache_preprocessed"

    cases, splits, prep = sanity_check(DATA_ROOT, CACHE_DIR)

    # Save split metadata for reproducibility
    split_meta = []
    for fold_i, fold in enumerate(splits):
        split_meta.append({
            "fold": fold_i,
            "train": [c["subject"] for c in fold["train"]],
            "val":   [c["subject"] for c in fold["val"]],
        })
    with open("isles22_splits.json", "w") as f:
        json.dump(split_meta, f, indent=2)
    print("\nSplit metadata saved to isles22_splits.json")