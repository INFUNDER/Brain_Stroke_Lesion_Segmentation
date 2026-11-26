import numpy as np
import pytest

from strokebench.counterfactual import edit


@pytest.fixture
def volume():
    rng = np.random.default_rng(0)
    adc = np.zeros((16, 16, 16), np.float32); dwi = np.zeros_like(adc)
    adc[2:14, 2:14, 2:14] = rng.normal(900, 30, (12, 12, 12)); dwi[2:14, 2:14, 2:14] = rng.normal(100, 5, (12, 12, 12))
    lesion = np.zeros(adc.shape, bool); lesion[6:9, 6:9, 6:9] = True
    adc[lesion] = 450; dwi[lesion] = 300
    return dwi, adc, lesion


def test_none_is_identity(volume):
    dwi, adc, lesion = volume
    d, a = edit(dwi, adc, lesion, "none")
    assert np.array_equal(d, dwi) and np.array_equal(a, adc)


def test_cf_a_normalises_adc_only_in_lesion(volume):
    dwi, adc, lesion = volume
    d, a = edit(dwi, adc, lesion, "cf_a")
    assert np.array_equal(d, dwi)
    assert np.allclose(a[lesion], np.median(adc[(adc > 0) & ~lesion]))
    assert np.array_equal(a[~lesion], adc[~lesion])


def test_cf_b_normalises_dwi_only_in_lesion(volume):
    dwi, adc, lesion = volume
    d, a = edit(dwi, adc, lesion, "cf_b")
    assert np.array_equal(a, adc) and np.array_equal(d[~lesion], dwi[~lesion])
    assert np.allclose(d[lesion], np.median(dwi[(adc > 0) & ~lesion]))


def test_cf_c_d_flatten_one_channel_inside_brain(volume):
    dwi, adc, lesion = volume
    brain = adc > 0
    d, a = edit(dwi, adc, lesion, "cf_c")
    assert np.array_equal(d, dwi) and len(np.unique(a[brain])) == 1 and np.all(a[~brain] == 0)
    d, a = edit(dwi, adc, lesion, "cf_d")
    assert np.array_equal(a, adc) and len(np.unique(d[brain])) == 1 and np.array_equal(d[~brain], dwi[~brain])
