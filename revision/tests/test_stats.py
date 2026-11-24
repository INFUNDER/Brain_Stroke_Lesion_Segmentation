import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from strokebench import stats as S


def test_holm_matches_statsmodels():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.uniform(0, 0.2, size=6)
        np.testing.assert_allclose(S.holm(p), multipletests(p, method="holm")[1], rtol=1e-12)


def test_bootstrap_interval_covers_true_difference():
    rng = np.random.default_rng(1)
    hits = 0
    for trial in range(200):
        diff = rng.normal(0.01, 0.08, size=247)
        b = S.paired_bootstrap(diff, strata=np.arange(247) % 5, n_boot=1000, seed=trial)
        hits += b["ci95_low"] <= 0.01 <= b["ci95_high"]
    assert 0.90 <= hits / 200 <= 0.99
    assert b["ci90_low"] >= b["ci95_low"] and b["ci90_high"] <= b["ci95_high"]


def test_equivalence_decision():
    rng = np.random.default_rng(2)
    strata = np.arange(247) % 5
    same = S.paired_bootstrap(rng.normal(0.000, 0.05, size=247), strata)
    worse = S.paired_bootstrap(rng.normal(-0.05, 0.05, size=247), strata)
    noisy = S.paired_bootstrap(rng.normal(0.000, 0.40, size=247), strata)
    assert S.equivalent(same) is True
    assert S.equivalent(worse) is False          # a real 0.05 Dice loss is not "equivalent"
    assert S.equivalent(noisy) is False          # too uncertain to claim equivalence


def test_equivalence_false_positive_rate_is_controlled():
    """A model that is truly worse by exactly the margin must rarely be called equivalent."""
    rng = np.random.default_rng(3)
    called = 0
    for trial in range(300):
        diff = rng.normal(-S.EQUIVALENCE_MARGIN, 0.08, size=247)
        called += S.equivalent(S.paired_bootstrap(diff, n_boot=1000, seed=trial))
    assert called / 300 <= 0.09                  # nominal 0.05


def test_compare_to_reference_pairs_by_subject():
    rng = np.random.default_rng(4)
    subjects = [f"s{i:03d}" for i in range(100)]
    folds = {s: i % 5 for i, s in enumerate(subjects)}
    ref = {s: float(rng.uniform(0.5, 0.9)) for s in subjects}
    better = {s: v + 0.05 for s, v in ref.items()}
    same = dict(ref)
    out = S.compare_to_reference(ref, {"better": better, "same": same}, folds)
    assert out["better"]["mean_diff"] == pytest.approx(0.05)
    assert out["better"]["wilcoxon_p_holm"] < 0.001 and out["better"]["equivalent_within_margin"] is False
    assert out["same"]["mean_diff"] == 0.0 and out["same"]["wilcoxon_p"] == 1.0 and out["same"]["equivalent_within_margin"] is True
    with pytest.raises(ValueError):
        S.compare_to_reference(ref, {"short": {s: 0.5 for s in subjects[:-1]}}, folds)
