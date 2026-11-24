#!/usr/bin/env python3
"""Create the one set of folds that every model uses.

Folds 0-4  five-fold cross-validation, stratified by protocol group and lesion
           volume tertile. Every subject is predicted exactly once by a model
           that never saw it, so the evaluation set is all 250 subjects.
Fold 5     protocol shift: train on thin-slice (A) minus N_SOURCE_CAL held-out thin-slice
           subjects; evaluate on those held-out subjects and on all thick-slice
           subjects (B1+B2). The held-out thin-slice subjects are the source-protocol
           calibration set for the volume intervals, so they must be unseen by the model.
Fold 6     protocol shift: train on thick-slice, evaluate on thin-slice.
Fold 7     size-matched control for fold 6: train on a random thin-slice subset
           of the same size as the thick-slice group, evaluate on everything else.

The split is a pure function of revision/data/cases.csv and SEED.
"""
import csv
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold

from strokebench import paths

SEED = 20261115
N_FOLDS = 5
N_SOURCE_CAL = 40


def load_cases() -> list:
    with open(paths.CASES_CSV) as f:
        return list(csv.DictReader(f))


def strata(rows: list) -> list:
    vol = np.array([float(r["lesion_ml"]) for r in rows])
    positive = vol > 0
    t1, t2 = np.percentile(vol[positive], [100 / 3, 200 / 3])
    tertile = np.where(vol <= t1, 0, np.where(vol <= t2, 1, 2))   # empty masks fall in the smallest tertile
    return [f"{r['protocol']}_{t}" for r, t in zip(rows, tertile)]


def make() -> list:
    rows = load_cases()
    cases = np.array([r["case"] for r in rows])
    labels = np.array(strata(rows))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = []
    for train_idx, val_idx in skf.split(cases, labels):
        splits.append({"train": sorted(cases[train_idx].tolist()), "val": sorted(cases[val_idx].tolist())})

    thin = sorted(r["case"] for r in rows if r["protocol"] == "A")
    thick = sorted(r["case"] for r in rows if r["protocol"] != "A")
    # fold 5: hold out a volume-stratified thin-slice calibration set
    thin_idx = np.array([i for i, r in enumerate(rows) if r["protocol"] == "A"])
    hold = StratifiedKFold(n_splits=round(len(thin) / N_SOURCE_CAL), shuffle=True, random_state=SEED + 5)
    _, cal_local = next(hold.split(thin_idx, labels[thin_idx]))
    thin_cal = sorted(cases[thin_idx[cal_local]].tolist())
    thin_train = sorted(set(thin) - set(thin_cal))
    splits.append({"train": thin_train, "val": sorted(thin_cal + thick)})          # fold 5
    splits.append({"train": thick, "val": thin})                                   # fold 6
    rng = np.random.default_rng(SEED)
    thin_subset = sorted(rng.choice(thin, size=len(thick), replace=False).tolist())
    rest = sorted(set(cases.tolist()) - set(thin_subset))
    splits.append({"train": thin_subset, "val": rest})                             # fold 7
    return splits


def main() -> None:
    splits = make()
    rows = {r["case"]: r for r in load_cases()}
    with open(paths.SPLITS_JSON, "w") as f:
        json.dump(splits, f, indent=1)
    for k, s in enumerate(splits):
        tr, va = s["train"], s["val"]
        vv = [float(rows[c]["lesion_ml"]) for c in va]
        prot = {g: sum(rows[c]["protocol"] == g for c in va) for g in ("A", "B1", "B2")}
        print(f"fold {k}: train {len(tr):3d} val {len(va):3d} | val protocols {prot} | val median lesion {np.median(vv):.2f} mL | empty in val {sum(v == 0 for v in vv)}")
    print(f"wrote {paths.SPLITS_JSON}")


if __name__ == "__main__":
    main()
