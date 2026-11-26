#!/usr/bin/env python3
"""E2: physics counterfactuals (PROTOCOL.md, amendment 2026-11-26, second entry).

Each lesion-positive subject is re-predicted by its own out-of-fold model after one edit
of the input, with the inference settings nnU-Net used for the cross-validation
predictions (sliding window, step 0.5, Gaussian weighting, mirroring, final checkpoint):

  none   unedited input (control; must reproduce the cross-validation prediction)
  cf_a   ADC inside the reference lesion -> median ADC of the brain outside the lesion
         (bright DWI with normal ADC, as in T2 shine-through or pseudo-normalisation)
  cf_b   DWI inside the reference lesion -> median DWI of the brain outside the lesion
         (low ADC with normal DWI)
  cf_c   ADC of every brain voxel -> the brain median ADC (DWI only)
  cf_d   DWI of every brain voxel -> the brain median DWI (ADC only)

The brain is the set of voxels with ADC > 0; voxels outside it keep their values, so
the head outline is unchanged. All edits are made in the stored units, so no unit
conversion is needed (nnU-Net z-scores each subject).

    python -m strokebench.counterfactual --make-inputs
    python -m strokebench.counterfactual --predict --models nnunet [--conditions none cf_a ...]
    python -m strokebench.counterfactual --score
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from strokebench import config, metrics, paths, stats

CONDITIONS = ["none", "cf_a", "cf_b", "cf_c", "cf_d"]
WORK = paths.NNUNET / "work" / "counterfactual"
DATASET = "dwi_adc"


def edit(dwi: np.ndarray, adc: np.ndarray, lesion: np.ndarray, condition: str):
    """Return the edited (dwi, adc) for one condition."""
    dwi, adc = dwi.astype(np.float32).copy(), adc.astype(np.float32).copy()
    brain = adc > 0
    outside = brain & ~lesion
    if condition == "none":
        pass
    elif condition == "cf_a":
        adc[lesion & brain] = np.median(adc[outside])
    elif condition == "cf_b":
        dwi[lesion & brain] = np.median(dwi[outside])
    elif condition == "cf_c":
        adc[brain] = np.median(adc[brain])
    elif condition == "cf_d":
        dwi[brain] = np.median(dwi[brain])
    else:
        raise ValueError(condition)
    return dwi, adc


def raw_dir() -> Path:
    return paths.NNUNET_RAW / paths.DATASETS[DATASET][1]


def lesion_positive_cases() -> list:
    return [r["case"] for r in csv.DictReader(open(paths.CASES_CSV)) if int(r["lesion_voxels"]) > 0]


def make_inputs() -> None:
    import SimpleITK as sitk
    for cond in CONDITIONS:
        (WORK / "inputs" / cond).mkdir(parents=True, exist_ok=True)
    for case in lesion_positive_cases():
        dimg = sitk.ReadImage(str(raw_dir() / "imagesTr" / f"{case}_0000.nii.gz"))
        aimg = sitk.ReadImage(str(raw_dir() / "imagesTr" / f"{case}_0001.nii.gz"))
        lesion = sitk.GetArrayFromImage(sitk.ReadImage(str(raw_dir() / "labelsTr" / f"{case}.nii.gz"))) > 0
        dwi, adc = sitk.GetArrayFromImage(dimg), sitk.GetArrayFromImage(aimg)
        for cond in CONDITIONS:
            d, a = edit(dwi, adc, lesion, cond)
            for arr, ref, ch in ((d, dimg, 0), (a, aimg, 1)):
                out = sitk.GetImageFromArray(arr)
                out.CopyInformation(ref)
                sitk.WriteImage(out, str(WORK / "inputs" / cond / f"{case}_{ch:04d}.nii.gz"))
    print(f"wrote inputs for {len(lesion_positive_cases())} subjects x {len(CONDITIONS)} conditions to {WORK / 'inputs'}")


def predict(models: list, conditions: list) -> None:
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from strokebench.evaluate import result_folder
    paths.set_nnunet_env()
    splits = json.load(open(paths.SPLITS_JSON))
    positive = set(lesion_positive_cases())
    for model in models:
        folder = result_folder(model, DATASET, paths.NNUNET_RESULTS)
        for fold in range(5):
            cases = sorted(c for c in splits[fold]["val"] if c in positive)
            predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                        perform_everything_on_device=True, device=torch.device("cuda"),
                                        verbose=False, verbose_preprocessing=False, allow_tqdm=False)
            predictor.initialize_from_trained_model_folder(str(folder), use_folds=(fold,), checkpoint_name="checkpoint_final.pth")
            for cond in conditions:
                out_dir = WORK / "predictions" / model / cond
                out_dir.mkdir(parents=True, exist_ok=True)
                todo = [c for c in cases if not (out_dir / f"{c}.nii.gz").exists()]
                if not todo:
                    continue
                inputs = [[str(WORK / "inputs" / cond / f"{c}_{ch:04d}.nii.gz") for ch in (0, 1)] for c in todo]
                predictor.predict_from_files(inputs, [str(out_dir / c) for c in todo], save_probabilities=False,
                                             overwrite=False, num_processes_preprocessing=2,
                                             num_processes_segmentation_export=2)
                print(f"{model} fold {fold} {cond}: {len(todo)} subjects predicted", flush=True)


def score() -> None:
    import SimpleITK as sitk
    from strokebench.evaluate import result_folder
    splits = json.load(open(paths.SPLITS_JSON))
    fold_of = {c: f for f in range(5) for c in splits[f]["val"]}
    rows, summary = [], {}
    pred_root = WORK / "predictions"
    for model_dir in sorted(p for p in pred_root.iterdir() if p.is_dir()):
        model = model_dir.name
        cv_folder = result_folder(model, DATASET, paths.NNUNET_RESULTS)
        for case in lesion_positive_cases():
            files = {c: model_dir / c / f"{case}.nii.gz" for c in CONDITIONS}
            if not all(f.exists() for f in files.values()):
                continue
            ref_img = sitk.ReadImage(str(raw_dir() / "labelsTr" / f"{case}.nii.gz"))
            ref = sitk.GetArrayFromImage(ref_img) > 0
            pred = {c: sitk.GetArrayFromImage(sitk.ReadImage(str(f))) > 0 for c, f in files.items()}
            cv = sitk.GetArrayFromImage(sitk.ReadImage(str(cv_folder / f"fold_{fold_of[case]}" / "validation" / f"{case}.nii.gz"))) > 0
            inside0 = int((pred["none"] & ref).sum())
            row = {"model": model, "case": case, "fold": fold_of[case], "control_matches_cv_dice": metrics.dice(cv, pred["none"]),
                   "pred_in_lesion_voxels_none": inside0}
            for c in CONDITIONS:
                row[f"dice_{c}"] = metrics.dice(ref, pred[c])
            for c in ("cf_a", "cf_b"):
                row[f"retained_{c}"] = float((pred[c] & ref).sum() / inside0) if inside0 > 0 else float("nan")
                # voxels newly predicted outside the lesion by the edit (should be ~0)
                row[f"new_outside_voxels_{c}"] = int((pred[c] & ~pred["none"] & ~ref).sum())
            rows.append(row)
        mrows = [r for r in rows if r["model"] == model]
        if not mrows:
            continue
        s = {"n": len(mrows), "control_matches_cv_dice_min": float(min(r["control_matches_cv_dice"] for r in mrows))}
        rng = np.random.default_rng(stats.SEED)
        for c in ("cf_a", "cf_b"):
            v = np.array([r[f"retained_{c}"] for r in mrows if np.isfinite(r[f"retained_{c}"])])
            boots = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(2000)]
            s[f"retained_{c}"] = {"n": int(len(v)), "median": float(np.median(v)), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                  "share_below_half": float(np.mean(v < 0.5))}
        for c in CONDITIONS:
            s[f"dice_{c}"] = float(np.mean([r[f"dice_{c}"] for r in mrows]))
        summary[model] = s
    with open(paths.REVISION / "results" / "per_case" / "counterfactual_dwi_adc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(summary, open(paths.REVISION / "results" / "counterfactual_dwi_adc.json", "w"), indent=1)
    for m, s in summary.items():
        a, b = s["retained_cf_a"], s["retained_cf_b"]
        print(f"{m:14s} n={s['n']}  control check min Dice {s['control_matches_cv_dice_min']:.3f}")
        print(f"   CF-A (normal ADC in lesion): median retained {a['median']:.2f} [{a['ci95'][0]:.2f}, {a['ci95'][1]:.2f}], below half in {100 * a['share_below_half']:.0f}%")
        print(f"   CF-B (normal DWI in lesion): median retained {b['median']:.2f} [{b['ci95'][0]:.2f}, {b['ci95'][1]:.2f}], below half in {100 * b['share_below_half']:.0f}%")
        print("   Dice: " + ", ".join(f"{c} {s[f'dice_{c}']:.3f}" for c in CONDITIONS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-inputs", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--models", nargs="+", default=["nnunet"], choices=list(config.ALL_MODELS))
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    args = ap.parse_args()
    if args.make_inputs:
        make_inputs()
    if args.predict:
        predict(args.models, args.conditions)
    if args.score:
        score()


if __name__ == "__main__":
    main()
