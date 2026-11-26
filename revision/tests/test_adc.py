"""ADC unit harmonisation (PROTOCOL.md, amendment 2026-11-26, second entry)."""
import numpy as np
import pytest

from strokebench import adc


@pytest.mark.parametrize("stored, factor, unit", [(0.00085, 1e6, "mm2/s"), (0.85, 1e3, "1e-3 mm2/s"), (850.0, 1.0, "1e-6 mm2/s")])
def test_each_scale_maps_to_the_same_values(stored, factor, unit):
    rng = np.random.default_rng(0)
    brain = rng.normal(1.0, 0.15, size=(20, 20, 20)).clip(0.05) * stored
    vol = np.zeros((30, 30, 30))
    vol[5:25, 5:25, 5:25] = brain                       # background zeros must be ignored
    assert adc.detect_scale(vol) == (factor, unit)
    h = adc.harmonise(vol)
    assert np.median(h[h > 0]) == pytest.approx(np.median(brain) * factor)
    assert 700 < np.median(h[h > 0]) < 1000             # normal brain ~ 850 x 1e-6 mm2/s


def test_lesion_threshold_is_unit_independent():
    lesion = np.full((4, 4, 4), 500.0)                  # 500 x 1e-6 mm2/s, below the 620 rule
    brain = np.full((10, 10, 10), 900.0)
    brain[:4, :4, :4] = lesion
    for f in (1e-6, 1e-3, 1.0):
        assert (adc.harmonise(brain * f) < 620).sum() == 64


def test_empty_map_raises():
    with pytest.raises(ValueError):
        adc.detect_scale(np.zeros((3, 3, 3)))
