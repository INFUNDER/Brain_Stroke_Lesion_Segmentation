"""Conformal intervals for lesion volume.

Three nonconformity scores are compared (PROTOCOL.md, section 8). Each defines a
family of intervals that grows with a scalar q; the calibrated q is the
finite-sample conformal quantile of the scores of calibration subjects that the
model has never seen. Under exchangeability of calibration and test subjects the
interval then contains the true volume with probability at least 1 - alpha.

absolute    score |V - Vhat|                 interval Vhat +/- q
logratio    score |log((V+e)/(Vhat+e))|      interval (Vhat+e)*exp(-/+q) - e
threshold   nested intervals [Vhat(0.5+q), Vhat(0.5-q)], where Vhat(t) is the volume of
            voxels whose lesion probability is at least t. The score of a subject is the
            smallest q whose interval contains its true volume.

Volumes are in mL. Lower bounds are clipped at zero.
"""
import math
from typing import Sequence, Tuple

import numpy as np

EPS_ML = 0.1                     # offset of the log-ratio score; keeps tiny volumes finite
N_BINS = 1000                    # resolution of the stored probability histogram
MAX_THRESHOLD_SCORE = 0.5        # threshold score at which the interval becomes [0, inf)


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th smallest calibration score; +inf if n is too small."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    return float("inf") if k > n else float(s[k - 1])


# ---------------------------------------------------------------- absolute residual
def score_absolute(v_ref: float, v_hat: float) -> float:
    return abs(v_ref - v_hat)


def interval_absolute(v_hat: float, q: float) -> Tuple[float, float]:
    return max(0.0, v_hat - q), v_hat + q


# ---------------------------------------------------------------- log-ratio residual
def score_logratio(v_ref: float, v_hat: float, eps: float = EPS_ML) -> float:
    return abs(math.log((v_ref + eps) / (v_hat + eps)))


def interval_logratio(v_hat: float, q: float, eps: float = EPS_ML) -> Tuple[float, float]:
    if math.isinf(q):
        return 0.0, float("inf")
    return max(0.0, (v_hat + eps) * math.exp(-q) - eps), (v_hat + eps) * math.exp(q) - eps


# ---------------------------------------------------------------- threshold-nested intervals
def probability_histogram(lesion_probability: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Voxel counts of the lesion probability in n_bins equal bins on [0, 1]."""
    idx = np.minimum((np.asarray(lesion_probability, dtype=np.float64) * n_bins).astype(np.int64), n_bins - 1)
    return np.bincount(idx.ravel(), minlength=n_bins).astype(np.int64)


def volume_at_bin(hist: np.ndarray, voxel_ml: float) -> np.ndarray:
    """vol[b] = volume of voxels with probability >= b / n_bins, for b = 0..n_bins (vol[n_bins] = 0)."""
    tail = np.concatenate([np.cumsum(hist[::-1])[::-1], [0]])
    return tail.astype(np.float64) * voxel_ml


def _half_width_bins(n_bins: int) -> int:
    return n_bins // 2


def point_volume(hist: np.ndarray, voxel_ml: float) -> float:
    """Volume at the arg-max decision (probability >= 0.5)."""
    return float(volume_at_bin(hist, voxel_ml)[_half_width_bins(len(hist))])


def score_threshold(v_ref: float, hist: np.ndarray, voxel_ml: float) -> float:
    """Smallest q (multiple of 1/n_bins) with Vhat(0.5+q) <= v_ref <= Vhat(0.5-q); 0.5 if none below 0.5."""
    n = len(hist)
    mid = _half_width_bins(n)
    vol = volume_at_bin(hist, voxel_ml)
    for j in range(0, mid):                       # q = j / n; thresholds mid+j and mid-j stay inside (0, 1)
        if vol[mid + j] <= v_ref <= vol[mid - j]:
            return j / n
    return MAX_THRESHOLD_SCORE


def interval_threshold(hist: np.ndarray, voxel_ml: float, q: float) -> Tuple[float, float]:
    n = len(hist)
    mid = _half_width_bins(n)
    if math.isinf(q) or q >= MAX_THRESHOLD_SCORE:
        return 0.0, float("inf")
    j = int(round(q * n))
    vol = volume_at_bin(hist, voxel_ml)
    return float(vol[mid + j]), float(vol[mid - j])


# ---------------------------------------------------------------- calibration schemes
def leave_one_out_quantiles(scores: Sequence[float], groups: Sequence, alpha: float) -> np.ndarray:
    """For each subject, the conformal quantile of the other subjects in the same group.

    With group = cross-validation fold, every calibration subject was unseen by the
    model that produced the target subject's prediction.
    """
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)
    out = np.empty(len(scores))
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        for i in idx:
            out[i] = conformal_quantile(scores[idx[idx != i]], alpha)
    return out


def coverage(lower: Sequence[float], upper: Sequence[float], v_ref: Sequence[float]) -> float:
    lower, upper, v_ref = (np.asarray(a, dtype=float) for a in (lower, upper, v_ref))
    return float(np.mean((v_ref >= lower) & (v_ref <= upper)))


# ---------------------------------------------------------------- adaptive (normalised log-ratio)
ADAPTIVE_DELTA = 0.1          # fixed in PROTOCOL.md amendment 2026-11-22 (third entry)


def spread(hist: np.ndarray, voxel_ml: float, low: float = 0.1, high: float = 0.9, eps: float = EPS_ML) -> float:
    """log((V(p>=low) + eps) / (V(p>=high) + eps)): how much the volume depends on the cut-off."""
    n = len(hist)
    vol = volume_at_bin(hist, voxel_ml)
    return float(math.log((vol[int(round(low * n))] + eps) / (vol[int(round(high * n))] + eps)))


def score_adaptive(v_ref: float, v_hat: float, s: float, delta: float = ADAPTIVE_DELTA) -> float:
    return score_logratio(v_ref, v_hat) / (s + delta)


def interval_adaptive(v_hat: float, s: float, q: float, delta: float = ADAPTIVE_DELTA) -> Tuple[float, float]:
    return interval_logratio(v_hat, q * (s + delta))
