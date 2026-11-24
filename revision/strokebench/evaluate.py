#!/usr/bin/env python3
"""Score a model's held-out predictions, one row per subject, in native space.

Reads the validation predictions that nnU-Net writes for each fold and compares
them with the repaired reference masks. Before scoring it checks that every
prediction belongs to a subject that was NOT in that fold's training set, and
that no subject is scored twice across the cross-validation folds.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from strokebench import config, metrics, paths

CV_FOLDS = (0, 1, 2, 3, 4)


def result_folder(model: str, dataset: str, results_root: Path, trainer_suffix: str = "", trainer: str = "") -> Path:
    plans, configuration, default_trainer = config.ALL_MODELS[model]
    name = (trainer or default_trainer) + trainer_suffix
    return results_root / paths.DATASETS[dataset][1] / f"{name}__{plans}__{configuration}"


def load_cases() -> dict:
    with open(paths.CASES_CSV) as f:
        return {r["case"]: r for r in csv.DictReader(f)}


def score_fold(folder: Path, fold: int, dataset: str, splits: list, cases: dict) -> list:
    val_dir = folder / f"fold_{fold}" / "validation"
    label_dir = paths.NNUNET_RAW / paths.DATASETS[dataset][1] / "labelsTr"
    allowed, forbidden = set(splits[fold]["val"]), set(splits[fold]["train"])
    files = sorted(val_dir.glob("*.nii.gz"))
    found = {f.name[:-7] for f in files}
    if found & forbidden:
        raise RuntimeError(f"fold {fold}: predictions exist for training subjects {sorted(found & forbidden)[:5]}")
    if found != allowed:
        raise RuntimeError(f"fold {fold}: {len(allowed - found)} validation subjects have no prediction, "
                           f"{len(found - allowed)} unexpected files in {val_dir}")
    rows = []
    for f in files:
        case = f.name[:-7]
        pred_img, ref_img = nib.load(f), nib.load(label_dir / f.name)
        if pred_img.shape != ref_img.shape or not np.allclose(pred_img.affine, ref_img.affine, atol=1e-3):
            raise RuntimeError(f"{case}: prediction and reference are on different grids")
        spacing = [float(z) for z in ref_img.header.get_zooms()[:3]]
        m = metrics.evaluate_case(np.asanyarray(ref_img.dataobj) > 0, np.asanyarray(pred_img.dataobj) > 0, spacing)
        rows.append({"case": case, "fold": fold, "protocol": cases[case]["protocol"], **m})
    return rows


def summarise(rows: list) -> dict:
    pos = [r for r in rows if not r["ref_empty"]]
    neg = [r for r in rows if r["ref_empty"]]
    arr = lambda key, rs: np.array([r[key] for r in rs], dtype=float)
    out = {"n": len(rows), "n_lesion_positive": len(pos), "n_empty_reference": len(neg)}
    if pos:
        out.update({
            "dice_mean": float(arr("dice", pos).mean()), "dice_median": float(np.median(arr("dice", pos))),
            "lesion_f1_mean": float(arr("lesion_f1", pos).mean()),
            "abs_volume_diff_ml_mean": float(arr("abs_volume_diff_ml", pos).mean()),
            "abs_lesion_count_diff_mean": float(arr("abs_lesion_count_diff", pos).mean()),
            "missed_entirely": int(sum(r["pred_empty"] for r in pos)),
        })
    if neg:
        out["empty_reference_correctly_empty"] = int(sum(r["pred_empty"] for r in neg))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(config.ALL_MODELS))
    ap.add_argument("--dataset", default="dwi_adc", choices=list(paths.DATASETS))
    ap.add_argument("--folds", type=int, nargs="+", default=list(CV_FOLDS))
    ap.add_argument("--results-root", default=str(paths.NNUNET_RESULTS))
    ap.add_argument("--trainer-suffix", default="", help='"Debug" to score a smoke-test run')
    ap.add_argument("--trainer", default="", help="score a run made with a non-default trainer (replicates, optimiser sensitivity)")
    ap.add_argument("--out-dir", default=str(paths.REVISION / "results" / "per_case"))
    args = ap.parse_args()

    splits = json.load(open(paths.NNUNET_PREPROCESSED / paths.DATASETS[args.dataset][1] / "splits_final.json"))
    assert splits == json.load(open(paths.SPLITS_JSON)), "installed folds differ from revision/data/splits_final.json"
    cases = load_cases()
    folder = result_folder(args.model, args.dataset, Path(args.results_root), args.trainer_suffix, args.trainer)

    rows = []
    for fold in args.folds:
        rows += score_fold(folder, fold, args.dataset, splits, cases)
    cv_rows = [r for r in rows if r["fold"] in CV_FOLDS]
    names = [r["case"] for r in cv_rows]
    assert len(names) == len(set(names)), "a subject was scored in more than one cross-validation fold"

    os.makedirs(args.out_dir, exist_ok=True)
    variant = "".join(f"_{v}" for v in (args.trainer, args.trainer_suffix.lower()) if v)
    tag = f"{args.model}{variant}_{args.dataset}"
    out_csv = Path(args.out_dir) / f"{tag}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for fold in args.folds:
        print(f"fold {fold}: {json.dumps(summarise([r for r in rows if r['fold'] == fold]))}")
    if set(CV_FOLDS) <= set(args.folds):
        print(f"cross-validation, all subjects: {json.dumps(summarise(cv_rows))}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
