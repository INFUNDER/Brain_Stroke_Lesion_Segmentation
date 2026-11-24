"""Conformal code is validated on synthetic data only (PROTOCOL.md, section 8)."""
import math

import numpy as np
import pytest

from strokebench import conformal as C


def test_quantile_rank_and_small_sample():
    s = [0.1, 0.5, 0.2, 0.9, 0.3, 0.4, 0.8, 0.6, 0.7]            # n = 9
    assert C.conformal_quantile(s, 0.10) == 0.9                  # ceil(10 * 0.9) = 9th smallest
    assert C.conformal_quantile(s, 0.20) == 0.8                  # 8th smallest
    assert math.isinf(C.conformal_quantile(s, 0.05))             # would need the 10th of 9
    assert math.isinf(C.conformal_quantile([], 0.1))


def test_absolute_and_logratio_intervals_invert_their_scores():
    rng = np.random.default_rng(0)
    for _ in range(200):
        v, vh = rng.uniform(0, 200, size=2)
        s = C.score_absolute(v, vh)
        lo, hi = C.interval_absolute(vh, s)
        assert lo - 1e-9 <= v <= hi + 1e-9
        lo, hi = C.interval_absolute(vh, s * 0.99 - 1e-6)
        assert not (lo <= v <= hi) or s < 1e-5
        s = C.score_logratio(v, vh)
        lo, hi = C.interval_logratio(vh, s)
        assert lo - 1e-9 <= v <= hi + 1e-9
    assert C.interval_absolute(1.0, 5.0)[0] == 0.0
    assert C.interval_logratio(3.0, float("inf")) == (0.0, float("inf"))


def test_probability_histogram_and_volume_curve():
    p = np.array([0.0, 0.2, 0.4999, 0.5, 0.75, 1.0])
    h = C.probability_histogram(p, n_bins=1000)
    assert h.sum() == 6 and h[0] == 1 and h[999] == 1 and h[500] == 1 and h[499] == 1
    vol = C.volume_at_bin(h, voxel_ml=2.0)
    assert vol[0] == 12.0 and vol[1000] == 0.0
    assert vol[500] == 6.0                                       # p >= 0.5 : 0.5, 0.75, 1.0
    assert C.point_volume(h, 2.0) == 6.0
    assert np.all(np.diff(vol) <= 0)


def test_threshold_score_is_smallest_covering_width():
    rng = np.random.default_rng(1)
    for _ in range(100):
        p = rng.beta(0.3, 0.3, size=4000)
        h = C.probability_histogram(p)
        v_ref = rng.uniform(0, 4000)
        s = C.score_threshold(v_ref, h, voxel_ml=1.0)
        lo, hi = C.interval_threshold(h, 1.0, s)
        assert lo <= v_ref <= hi
        if 0 < s < C.MAX_THRESHOLD_SCORE:
            lo, hi = C.interval_threshold(h, 1.0, s - 1 / C.N_BINS)
            assert not (lo <= v_ref <= hi)
    # nestedness: wider q never shrinks the interval
    h = C.probability_histogram(rng.beta(0.3, 0.3, size=4000))
    widths = [np.subtract(*C.interval_threshold(h, 1.0, j / C.N_BINS)[::-1]) for j in range(0, 500, 25)]
    assert all(b >= a for a, b in zip(widths, widths[1:]))
    assert C.interval_threshold(h, 1.0, 0.5) == (0.0, float("inf"))


def _synthetic_cohort(rng, n):
    """Subjects with heteroscedastic volume errors and soft probability maps."""
    v_ref = np.exp(rng.normal(1.5, 1.2, size=n))                              # mL
    v_hat = np.maximum(0.0, v_ref * np.exp(rng.normal(0.0, 0.25, size=n)) + rng.normal(0, 0.3, size=n))
    hists = []
    for vr in v_ref:
        n_core = int(vr * 100)                                                # voxel = 0.01 mL
        sharp = rng.uniform(2, 12)
        p = np.concatenate([rng.beta(sharp, 1.0, size=n_core), rng.beta(1.0, sharp, size=n_core + 50)])
        hists.append(C.probability_histogram(p))
    return v_ref, v_hat, hists


@pytest.mark.parametrize("alpha", [0.2, 0.1])
def test_marginal_coverage_on_exchangeable_data(alpha):
    rng = np.random.default_rng(42)
    covered = {"absolute": [], "logratio": [], "threshold": []}
    for trial in range(40):
        v_ref, v_hat, hists = _synthetic_cohort(rng, 250)
        folds = np.repeat(np.arange(5), 50)
        s_abs = [C.score_absolute(a, b) for a, b in zip(v_ref, v_hat)]
        s_log = [C.score_logratio(a, b) for a, b in zip(v_ref, v_hat)]
        s_thr = [C.score_threshold(a, h, 0.01) for a, h in zip(v_ref, hists)]
        q_abs = C.leave_one_out_quantiles(s_abs, folds, alpha)
        q_log = C.leave_one_out_quantiles(s_log, folds, alpha)
        q_thr = C.leave_one_out_quantiles(s_thr, folds, alpha)
        iv = [C.interval_absolute(vh, q) for vh, q in zip(v_hat, q_abs)]
        covered["absolute"].append(C.coverage([i[0] for i in iv], [i[1] for i in iv], v_ref))
        iv = [C.interval_logratio(vh, q) for vh, q in zip(v_hat, q_log)]
        covered["logratio"].append(C.coverage([i[0] for i in iv], [i[1] for i in iv], v_ref))
        iv = [C.interval_threshold(h, 0.01, q) for h, q in zip(hists, q_thr)]
        covered["threshold"].append(C.coverage([i[0] for i in iv], [i[1] for i in iv], v_ref))
    for name, c in covered.items():
        mean = float(np.mean(c))
        assert mean >= 1 - alpha - 0.01, (name, mean)            # guarantee: at least 1 - alpha
        assert mean <= 1 - alpha + 0.04, (name, mean)            # and not vacuously wide


def test_leave_one_out_uses_only_same_group():
    scores = np.array([1.0, 2.0, 3.0, 100.0, 200.0, 300.0])
    groups = np.array([0, 0, 0, 1, 1, 1])
    q = C.leave_one_out_quantiles(scores, groups, alpha=0.5)     # n = 2 -> ceil(3 * 0.5) = 2nd smallest
    assert q.tolist() == [3.0, 3.0, 2.0, 300.0, 300.0, 200.0]


def test_decision_baselines_on_constructed_cohort():
    from strokebench.volumes import decision_baselines
    # 6 subjects, threshold 70 mL; estimates chosen so the rules are hand-checkable
    v_ref = [10.0, 65.0, 75.0, 68.0, 72.0, 200.0]
    v_hat = [12.0, 72.0, 68.0, 69.0, 80.0, 190.0]        # subjects 2 and 3 are on the wrong side of 70
    rows = [{"v_ref": a, "v_hat": b} for a, b in zip(v_ref, v_hat)]
    lower = [5, 60, 60, 60, 75, 150]                       # interval flags subjects 2, 3, 4 (contain 70)
    upper = [20, 85, 80, 80, 90, 250]
    out = decision_baselines(rows, lower, upper)["70.0"]
    assert out["point_errors"] == 2
    assert out["interval"] == {"flagged": 3, "errors_left": 0, "errors_caught": 2}
    # three estimates closest to 70: 69, 68, 72 -> subjects 4, 3, 2 (same set by ratio and by mL)
    assert out["ratio_matched"]["flagged"] == 3 and out["ratio_matched"]["errors_left"] == 0
    assert out["ml_matched"]["overlap_with_interval"] == 3
    # +/-10% of 70 = [63, 77]: subjects 2, 3, 4 flagged; +/-50% = [35, 105]: adds subject 5 (80)
    assert out["fixed_band_10pct"]["flagged"] == 3
    assert out["fixed_band_50pct"]["flagged"] == 4


def test_spread_and_adaptive_interval():
    # 100 voxels at p=0.95, 100 at p=0.5, 100 at p=0.2, voxel 0.01 mL
    p = np.concatenate([np.full(100, 0.95), np.full(100, 0.5), np.full(100, 0.2)])
    h = C.probability_histogram(p)
    s = C.spread(h, 0.01)
    assert s == pytest.approx(math.log((3.0 + 0.1) / (1.0 + 0.1)))
    v_hat = C.point_volume(h, 0.01)                              # p >= 0.5 -> 2.0 mL
    assert v_hat == pytest.approx(2.0)
    for v_ref in (0.5, 2.0, 7.0):
        sc = C.score_adaptive(v_ref, v_hat, s)
        lo, hi = C.interval_adaptive(v_hat, s, sc)
        assert lo - 1e-9 <= v_ref <= hi + 1e-9                  # the score inverts the interval
    lo1, hi1 = C.interval_adaptive(v_hat, 0.0, 1.0)
    lo2, hi2 = C.interval_adaptive(v_hat, 1.0, 1.0)
    assert lo2 < lo1 and hi2 > hi1                              # larger spread -> wider interval


def test_adaptive_coverage_on_exchangeable_data():
    rng = np.random.default_rng(7)
    cov = []
    for _ in range(40):
        n = 250
        s = rng.uniform(0.05, 1.0, size=n)
        v_ref = np.exp(rng.normal(1.5, 1.2, size=n))
        v_hat = (v_ref + C.EPS_ML) * np.exp(rng.normal(0, 0.4 * s)) - C.EPS_ML   # error scales with spread
        folds = np.repeat(np.arange(5), 50)
        sc = [C.score_adaptive(a, b, c) for a, b, c in zip(v_ref, v_hat, s)]
        q = C.leave_one_out_quantiles(sc, folds, 0.1)
        iv = [C.interval_adaptive(b, c, qq) for b, c, qq in zip(v_hat, s, q)]
        cov.append(C.coverage([i[0] for i in iv], [i[1] for i in iv], v_ref))
    assert 0.89 <= float(np.mean(cov)) <= 0.94
