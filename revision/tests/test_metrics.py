"""Hand-checkable cases for every metric, including the empty-mask edge cases
that produced the scoring confusion in the first manuscript."""
import numpy as np
import pytest

from strokebench import metrics as M


def blank(shape=(12, 12, 12)):
    return np.zeros(shape, dtype=np.uint8)


def test_dice_known_overlap():
    ref, pred = blank(), blank()
    ref[2:6, 2:6, 2:6] = 1            # 64 voxels
    pred[4:8, 2:6, 2:6] = 1           # 64 voxels, 32 shared
    assert M.dice(ref, pred) == pytest.approx(2 * 32 / 128)


def test_dice_empty_cases():
    ref, pred = blank(), blank()
    assert M.dice(ref, pred) == 1.0                    # both empty: correct rejection
    pred[0, 0, 0] = 1
    assert M.dice(ref, pred) == 0.0                    # false alarm on empty reference
    assert M.dice(pred, ref) == 0.0                    # missed lesion


def test_volume_uses_native_voxel_size():
    ref, pred = blank(), blank()
    ref[0:5, 0:4, 0:1] = 1                             # 20 voxels
    pred[0:5, 0:2, 0:1] = 1                            # 10 voxels
    voxel_ml = (1.797 * 1.797 * 4.8) / 1000
    assert M.absolute_volume_difference_ml(ref, pred, voxel_ml) == pytest.approx(10 * voxel_ml)


def test_lesion_count_uses_26_connectivity():
    m = blank()
    m[1, 1, 1] = 1
    m[2, 2, 2] = 1                                     # touches only at a corner
    assert M.lesion_count(m) == 1
    m[6, 6, 6] = 1
    assert M.lesion_count(m) == 2


def test_lesion_f1_counts():
    ref, pred = blank(), blank()
    ref[1:3, 1:3, 1:3] = 1                             # lesion 1: detected
    ref[8:10, 8:10, 8:10] = 1                          # lesion 2: missed
    pred[2:4, 2:4, 2:4] = 1                            # overlaps lesion 1
    pred[5, 5, 5] = 1                                  # false positive
    c = M.lesion_detection_counts(ref, pred)
    assert (c["tp"], c["fn"], c["fp"]) == (1, 1, 1)
    assert M.lesion_f1(ref, pred) == pytest.approx(1 / (1 + 0.5 * 2))


def test_one_prediction_covering_two_lesions():
    ref, pred = blank(), blank()
    ref[1:3, 1:3, 1:3] = 1
    ref[5:7, 1:3, 1:3] = 1
    pred[1:7, 1:3, 1:3] = 1                            # one blob bridges both
    c = M.lesion_detection_counts(ref, pred)
    assert (c["tp"], c["fn"], c["fp"]) == (2, 0, 0)
    assert M.absolute_lesion_count_difference(ref, pred) == 1


def test_lesion_f1_empty_cases():
    ref, pred = blank(), blank()
    assert M.lesion_f1(ref, pred) == 1.0
    pred[3, 3, 3] = 1
    assert M.lesion_f1(ref, pred) == 0.0


def test_hd95_matches_brute_force():
    rng = np.random.default_rng(0)
    spacing = (1.5, 1.5, 4.8)
    for _ in range(5):
        ref = ndimage_blob(rng)
        pred = np.roll(ref, shift=rng.integers(0, 3), axis=int(rng.integers(0, 3)))
        got = M.hd95_mm(ref, pred, spacing)
        s_ref, s_pred = M._surface(ref.astype(bool)), M._surface(pred.astype(bool))
        a = np.argwhere(s_ref) * np.array(spacing)
        b = np.argwhere(s_pred) * np.array(spacing)
        d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
        want = np.percentile(np.concatenate([d.min(0), d.min(1)]), 95)
        assert got == pytest.approx(want)


def ndimage_blob(rng):
    m = np.zeros((14, 14, 10), dtype=np.uint8)
    x, y, z = rng.integers(3, 8, size=3)
    m[x:x + 4, y:y + 3, z:z + 2] = 1
    return m


def test_hd95_identical_and_empty():
    m = blank()
    m[2:5, 2:5, 2:5] = 1
    assert M.hd95_mm(m, m, (1, 1, 1)) == 0.0
    assert M.hd95_mm(m, blank(), (1, 1, 1)) is None


def test_evaluate_case_consistency():
    ref, pred = blank(), blank()
    ref[2:6, 2:6, 2:6] = 1
    pred[2:6, 2:6, 2:5] = 1
    out = M.evaluate_case(ref, pred, spacing=(2.0, 2.0, 2.0))
    assert out["ref_ml"] == pytest.approx(64 * 0.008)
    assert out["volume_error_ml"] == pytest.approx(-16 * 0.008)
    assert out["dice"] == pytest.approx(2 * 48 / (64 + 48))
    assert out["lesion_f1"] == 1.0
    with pytest.raises(ValueError):
        M.evaluate_case(ref, pred[:-1], spacing=(1, 1, 1))
