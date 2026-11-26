import numpy as np

from strokebench.physics_rule import rule_masks


def test_rules_on_synthetic_brain():
    rng = np.random.default_rng(0)
    adc = np.zeros((20, 20, 20)); dwi = np.zeros_like(adc)
    adc[2:18, 2:18, 2:18] = rng.normal(0.9, 0.05, (16, 16, 16))       # stored in 1e-3 mm2/s
    dwi[2:18, 2:18, 2:18] = rng.normal(100, 5, (16, 16, 16))
    adc[8:11, 8:11, 8:11] = 0.45; dwi[8:11, 8:11, 8:11] = 300        # true infarct: low ADC, bright DWI
    adc[3:5, 3:5, 3:5] = 0.45                                          # low ADC, normal DWI
    m = rule_masks(dwi, adc)
    assert m["R-ADC"][8:11, 8:11, 8:11].all() and m["R-ADC"][3:5, 3:5, 3:5].all()
    assert m["R-ADC+DWI"][8:11, 8:11, 8:11].all() and not m["R-ADC+DWI"][3:5, 3:5, 3:5].any()
    assert not m["R-ADC"][0].any()                                     # background excluded
