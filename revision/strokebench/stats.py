"""Paired, subject-level statistics for the backbone comparison (PROTOCOL.md, section 6)."""
from typing import Dict, Optional, Sequence

import numpy as np
from scipy import stats as sps

EQUIVALENCE_MARGIN = 0.02      # absolute Dice, fixed in the protocol
N_BOOTSTRAP = 10_000
SEED = 20261115


def paired_bootstrap(diff: Sequence[float], strata: Optional[Sequence] = None, n_boot: int = N_BOOTSTRAP,
                     seed: int = SEED) -> Dict[str, float]:
    """Mean paired difference with percentile bootstrap intervals.

    Subjects are resampled with replacement within strata (cross-validation folds),
    so every bootstrap sample keeps the fold sizes of the real design.
    """
    diff = np.asarray(diff, dtype=float)
    strata = np.zeros(len(diff), dtype=int) if strata is None else np.asarray(strata)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == g) for g in np.unique(strata)]
    sums = np.zeros(n_boot)
    for idx in groups:
        draws = rng.integers(0, len(idx), size=(n_boot, len(idx)))
        sums += diff[idx][draws].sum(axis=1)
    means = sums / len(diff)
    lo95, hi95 = np.percentile(means, [2.5, 97.5])
    lo90, hi90 = np.percentile(means, [5.0, 95.0])
    return {"mean_diff": float(diff.mean()), "ci95_low": float(lo95), "ci95_high": float(hi95),
            "ci90_low": float(lo90), "ci90_high": float(hi90), "n": int(len(diff))}


def equivalent(boot: Dict[str, float], margin: float = EQUIVALENCE_MARGIN) -> bool:
    """Two one-sided tests at alpha = 0.05: the 90% interval lies inside +/- margin."""
    return bool(boot["ci90_low"] > -margin and boot["ci90_high"] < margin)


def wilcoxon_two_sided(diff: Sequence[float]) -> float:
    diff = np.asarray(diff, dtype=float)
    if np.allclose(diff, 0):
        return 1.0
    return float(sps.wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue)


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, returned in the input order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def compare_to_reference(reference: Dict[str, float], others: Dict[str, Dict[str, float]],
                         folds: Dict[str, int]) -> Dict[str, dict]:
    """reference / others[model]: subject -> Dice. Returns one row per model, Holm-corrected across models."""
    subjects = sorted(reference)
    strata = np.array([folds[s] for s in subjects])
    rows, p_raw = {}, []
    for model, values in others.items():
        if sorted(values) != subjects:
            raise ValueError(f"{model}: subjects differ from the reference")
        diff = np.array([values[s] - reference[s] for s in subjects])
        boot = paired_bootstrap(diff, strata)
        p = wilcoxon_two_sided(diff)
        p_raw.append(p)
        rows[model] = {**boot, "equivalent_within_margin": equivalent(boot), "wilcoxon_p": p,
                       "share_subjects_better": float(np.mean(diff > 0)), "median_abs_diff": float(np.median(np.abs(diff)))}
    for model, adj in zip(rows, holm(p_raw) if p_raw else []):
        rows[model]["wilcoxon_p_holm"] = float(adj)
    return rows
