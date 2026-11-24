"""Evaluation metrics, computed per subject in native image space.

The four ISLES'22 ranking metrics (Dice, absolute volume difference, absolute
lesion-count difference, lesion-wise F1) follow the definitions of the
challenge's public evaluation code: lesions are 26-connected components, a
reference lesion is detected if any predicted voxel overlaps it, and a predicted
lesion is a false positive if it overlaps no reference voxel.

Empty references are handled explicitly and identically for every model:
Dice and lesion F1 are 1.0 when both masks are empty and 0.0 when only one is.
Results on empty-reference subjects are always reported separately as well.
"""
from typing import Optional, Sequence

import cc3d
import numpy as np
from scipy import ndimage

CONNECTIVITY = 26


def _as_bool(a) -> np.ndarray:
    return np.asarray(a).astype(bool)


def dice(ref, pred, empty_value: float = 1.0) -> float:
    ref, pred = _as_bool(ref), _as_bool(pred)
    denom = int(ref.sum()) + int(pred.sum())
    if denom == 0:
        return float(empty_value)
    return float(2.0 * np.logical_and(ref, pred).sum() / denom)


def volume_ml(mask, voxel_ml: float) -> float:
    return float(_as_bool(mask).sum() * voxel_ml)


def absolute_volume_difference_ml(ref, pred, voxel_ml: float) -> float:
    return abs(volume_ml(ref, voxel_ml) - volume_ml(pred, voxel_ml))


def lesion_count(mask) -> int:
    return int(cc3d.connected_components(_as_bool(mask).astype(np.uint8), connectivity=CONNECTIVITY, return_N=True)[1])


def absolute_lesion_count_difference(ref, pred) -> int:
    return abs(lesion_count(ref) - lesion_count(pred))


def lesion_detection_counts(ref, pred) -> dict:
    """Lesion-wise true positives, false negatives and false positives."""
    ref, pred = _as_bool(ref), _as_bool(pred)
    ref_lab, n_ref = cc3d.connected_components(ref.astype(np.uint8), connectivity=CONNECTIVITY, return_N=True)
    pred_lab, n_pred = cc3d.connected_components(pred.astype(np.uint8), connectivity=CONNECTIVITY, return_N=True)
    detected = np.unique(ref_lab[pred])          # reference lesions touched by any predicted voxel
    tp = int((detected > 0).sum())
    matched_pred = np.unique(pred_lab[ref])      # predicted lesions touching any reference voxel
    fp = int(n_pred - (matched_pred > 0).sum())
    return {"tp": tp, "fn": int(n_ref - tp), "fp": fp, "n_ref": int(n_ref), "n_pred": int(n_pred)}


def lesion_f1(ref, pred, empty_value: float = 1.0) -> float:
    c = lesion_detection_counts(ref, pred)
    if c["tp"] + c["fp"] + c["fn"] == 0:
        return float(empty_value)
    return float(c["tp"] / (c["tp"] + 0.5 * (c["fp"] + c["fn"])))


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(mask.ndim, 1), border_value=0)


def hd95_mm(ref, pred, spacing: Sequence[float]) -> Optional[float]:
    """Symmetric 95th-percentile surface distance in mm; None if either mask is empty."""
    ref, pred = _as_bool(ref), _as_bool(pred)
    if not ref.any() or not pred.any():
        return None
    s_ref, s_pred = _surface(ref), _surface(pred)
    d_to_ref = ndimage.distance_transform_edt(~s_ref, sampling=spacing)
    d_to_pred = ndimage.distance_transform_edt(~s_pred, sampling=spacing)
    d = np.concatenate([d_to_ref[s_pred], d_to_pred[s_ref]])
    return float(np.percentile(d, 95))


def evaluate_case(ref, pred, spacing: Sequence[float]) -> dict:
    """All per-subject metrics. `spacing` is the native voxel size in mm."""
    ref, pred = _as_bool(ref), _as_bool(pred)
    if ref.shape != pred.shape:
        raise ValueError(f"shape mismatch {ref.shape} vs {pred.shape}")
    voxel_ml = float(np.prod(spacing)) / 1000.0
    c = lesion_detection_counts(ref, pred)
    ref_ml, pred_ml = volume_ml(ref, voxel_ml), volume_ml(pred, voxel_ml)
    return {
        "ref_empty": int(not ref.any()),
        "pred_empty": int(not pred.any()),
        "dice": dice(ref, pred),
        "ref_ml": ref_ml,
        "pred_ml": pred_ml,
        "volume_error_ml": pred_ml - ref_ml,
        "abs_volume_diff_ml": abs(pred_ml - ref_ml),
        "abs_lesion_count_diff": abs(c["n_ref"] - c["n_pred"]),
        "lesion_f1": lesion_f1(ref, pred),
        "lesion_tp": c["tp"], "lesion_fn": c["fn"], "lesion_fp": c["fp"],
        "n_ref_lesions": c["n_ref"], "n_pred_lesions": c["n_pred"],
        "hd95_mm": hd95_mm(ref, pred, spacing),
    }
