#!/usr/bin/env python3
"""ADC unit harmonisation (PROTOCOL.md, amendment 2026-11-26, second entry).

The raw ISLES 2022 ADC maps are stored in three unit scales. The median ADC of brain
tissue is about 0.0008-0.001 in mm^2/s, about 1 in 10^-3 mm^2/s and about 1000 in
10^-6 mm^2/s, so the scale of a subject is read off its median brain ADC.

Every analysis that uses absolute ADC values converts to 10^-6 mm^2/s first.
Network training is not affected: nnU-Net z-scores each subject.

    python -m strokebench.adc        # writes revision/data/adc_scale.csv
"""
import csv

import numpy as np

from strokebench import paths

TARGET_UNIT = "1e-6 mm2/s"
# (upper bound on the median brain ADC, factor to 10^-6 mm^2/s, name of the stored unit)
SCALES = [(0.01, 1e6, "mm2/s"), (10.0, 1e3, "1e-3 mm2/s"), (float("inf"), 1.0, "1e-6 mm2/s")]


def brain_median(adc: np.ndarray) -> float:
    vals = np.asarray(adc, dtype=np.float64)
    vals = vals[vals > 0]
    if vals.size == 0:
        raise ValueError("ADC map has no positive voxels")
    return float(np.median(vals))


def detect_scale(adc: np.ndarray):
    """(factor to 10^-6 mm^2/s, stored unit) from the median of the positive voxels."""
    m = brain_median(adc)
    for upper, factor, unit in SCALES:
        if m < upper:
            return factor, unit
    raise AssertionError("unreachable")


def harmonise(adc: np.ndarray) -> np.ndarray:
    """ADC in 10^-6 mm^2/s, whatever unit it was stored in."""
    factor, _ = detect_scale(adc)
    return np.asarray(adc, dtype=np.float64) * factor


def main() -> None:
    import SimpleITK as sitk
    paths.set_nnunet_env()
    _, ds_name = paths.DATASETS["dwi_adc"]
    images = paths.NNUNET_RAW / ds_name / "imagesTr"
    cases = list(csv.DictReader(open(paths.CASES_CSV)))
    rows = []
    for r in cases:
        adc = sitk.GetArrayFromImage(sitk.ReadImage(str(images / f"{r['case']}_0001.nii.gz")))
        factor, unit = detect_scale(adc)
        m = brain_median(adc)
        rows.append({"case": r["case"], "protocol": r["protocol"], "vendor": r["vendor"], "stored_unit": unit,
                     "factor_to_1e-6": factor, "median_brain_adc_stored": m, "median_brain_adc_1e-6": m * factor})
    out = paths.DATA / "adc_scale.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    harmonised = np.array([r["median_brain_adc_1e-6"] for r in rows])
    print(f"wrote {out}")
    for unit in [s[2] for s in SCALES]:
        sel = [r for r in rows if r["stored_unit"] == unit]
        by = {}
        for r in sel:
            key = (r["protocol"], r["vendor"] or "unknown")
            by[key] = by.get(key, 0) + 1
        print(f"  {unit:11s} n={len(sel):3d}  by (protocol, vendor): {dict(sorted(by.items()))}")
    print(f"  median brain ADC after harmonisation: {np.median(harmonised):.0f} x 1e-6 mm2/s "
          f"(range {harmonised.min():.0f}-{harmonised.max():.0f})")


if __name__ == "__main__":
    main()
