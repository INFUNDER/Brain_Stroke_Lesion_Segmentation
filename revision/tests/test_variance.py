import numpy as np

from strokebench.variance import decompose, r2_subject_means


def test_decomposition_on_constructed_matrix():
    subj = np.array([0.2, 0.5, 0.9])[:, None]
    back = np.array([0.0, 0.1])[None, :]
    x = subj + back                                   # no interaction
    s = decompose(x)
    assert abs(s["residual"]) < 1e-12
    assert abs(s["subject"] + s["backbone"] - 1) < 1e-12
    assert s["subject"] > s["backbone"]
    x2 = np.repeat(np.array([[0.3, 0.7]]), 4, axis=0)  # only the backbone differs
    assert abs(decompose(x2)["backbone"] - 1) < 1e-12


def test_r2_perfect_fit():
    lv = np.linspace(0, 3, 20)
    proto = np.array(["A"] * 10 + ["B"] * 10)
    means = 0.5 + 0.1 * lv
    r = r2_subject_means(means, proto, lv)
    assert abs(r["log_volume"] - 1) < 1e-9 and abs(r["protocol+log_volume"] - 1) < 1e-9
