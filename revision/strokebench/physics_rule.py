#!/usr/bin/env python3
"""Physics-rule baselines, no training (PROTOCOL.md, amendment 2026-11-26, second entry).

R-ADC       ADC < 620 x 10^-6 mm^2/s inside the brain (ADC > 0)
R-ADC+DWI   the same, and DWI above the brain mean + 2 SD (brain = ADC > 0)

Thresholds are fixed in the protocol and not tuned. Applied in native space to the
harmonised ADC (strokebench.adc) and scored against the corrected labels.

    python -m strokebench.physics_rule
"""
import csv
import json
from multiprocessing import Pool

import numpy as np

from strokebench import adc as adc_units
from strokebench import metrics, paths, stats

ADC_THRESHOLD = 620.0            # x 10^-6 mm^2/s
DWI_SD = 2.0
VOLUME_BINS = [(0, 1, "<1 mL"), (1, 10, "1-10 mL"), (10, 70, "10-70 mL"), (70, float("inf"), ">70 mL")]


def rule_masks(dwi: np.ndarray, adc_stored: np.ndarray) -> dict:
    adc = adc_units.harmonise(adc_stored)
    brain = adc_stored > 0
    low_adc = brain & (adc < ADC_THRESHOLD)
    d = dwi[brain].astype(np.float64)
    bright = dwi > d.mean() + DWI_SD * d.std()
    return {"R-ADC": low_adc, "R-ADC+DWI": low_adc & bright}


def score_case(case: str) -> list:
    import SimpleITK as sitk
    paths.set_nnunet_env()
    _, ds = paths.DATASETS["dwi_adc"]
    root = paths.NNUNET_RAW / ds
    img = sitk.ReadImage(str(root / "imagesTr" / f"{case}_0000.nii.gz"))
    dwi = sitk.GetArrayFromImage(img)
    adc = sitk.GetArrayFromImage(sitk.ReadImage(str(root / "imagesTr" / f"{case}_0001.nii.gz")))
    ref = sitk.GetArrayFromImage(sitk.ReadImage(str(root / "labelsTr" / f"{case}.nii.gz"))) > 0
    voxel_ml = float(np.prod(img.GetSpacing())) / 1000.0
    out = []
    for rule, pred in rule_masks(dwi, adc).items():
        out.append({"case": case, "rule": rule, "ref_empty": int(ref.sum() == 0), "ref_ml": metrics.volume_ml(ref, voxel_ml),
                    "pred_ml": metrics.volume_ml(pred, voxel_ml), "dice": metrics.dice(ref, pred),
                    "lesion_f1": metrics.lesion_f1(ref, pred),
                    "abs_volume_diff_ml": metrics.absolute_volume_difference_ml(ref, pred, voxel_ml)})
    return out


def main() -> None:
    cases = [r["case"] for r in csv.DictReader(open(paths.CASES_CSV))]
    with Pool(16) as pool:
        rows = [r for rs in pool.map(score_case, cases) for r in rs]
    out_csv = paths.REVISION / "results" / "per_case" / "physics_rules_dwi_adc.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    nn = {r["case"]: float(r["dice"]) for r in csv.DictReader(open(paths.REVISION / "results" / "per_case" / "nnunet_dwi_adc.csv"))}
    summary = {}
    for rule in ("R-ADC", "R-ADC+DWI"):
        pos = [r for r in rows if r["rule"] == rule and not r["ref_empty"]]
        s = {k: float(np.mean([r[k] for r in pos])) for k in ("dice", "lesion_f1", "abs_volume_diff_ml")}
        s["dice_median"] = float(np.median([r["dice"] for r in pos]))
        s["by_volume"] = {lab: float(np.mean([r["dice"] for r in pos if lo <= r["ref_ml"] < hi])) for lo, hi, lab in VOLUME_BINS}
        diff = np.array([r["dice"] - nn[r["case"]] for r in pos])
        s["vs_nnunet"] = {k: v for k, v in stats.paired_bootstrap(diff).items() if not k.startswith("_")}
        summary[rule] = s
    path = paths.REVISION / "results" / "physics_rules_dwi_adc.json"
    json.dump({"adc_threshold_1e-6_mm2_s": ADC_THRESHOLD, "dwi_sd": DWI_SD, "summary": summary}, open(path, "w"), indent=1)
    for rule, s in summary.items():
        v = s["vs_nnunet"]
        print(f"{rule:10s} Dice {s['dice']:.3f} (median {s['dice_median']:.3f})  F1 {s['lesion_f1']:.3f}  AVD {s['abs_volume_diff_ml']:.1f} mL  "
              f"vs nnU-Net {v['mean_diff']:+.3f} [{v['ci95_low']:+.3f}, {v['ci95_high']:+.3f}]")
        print("           by volume: " + ", ".join(f"{k} {x:.3f}" for k, x in s["by_volume"].items()))
    print(f"wrote {out_csv} and {path}")


if __name__ == "__main__":
    main()
