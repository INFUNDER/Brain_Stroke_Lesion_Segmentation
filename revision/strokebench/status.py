#!/usr/bin/env python3
"""Show how far every training run has got:  python -m strokebench.status

One row per model, one column per fold: "done", "ep 120/250", or "-" (not started).
Finished folds also show nnU-Net's own validation Dice (lesion class) for a quick look;
the study's results always come from strokebench.evaluate.
"""
import glob
import json
import os
import re

from strokebench import config, paths


def fold_state(folder: str, total_epochs: int) -> str:
    summary = os.path.join(folder, "validation", "summary.json")
    if os.path.isfile(summary):
        try:
            dice = json.load(open(summary))["foreground_mean"]["Dice"]
            return f"done {dice:.3f}"
        except Exception:
            return "done"
    logs = sorted(glob.glob(os.path.join(folder, "training_log_*.txt")), key=os.path.getmtime)
    last = -1
    for log in logs:
        with open(log, errors="ignore") as f:
            for m in re.finditer(r"Epoch (\d+)\s*$", f.read(), flags=re.M):
                last = max(last, int(m.group(1)))
    if last < 0:
        return "-"
    if os.path.isfile(os.path.join(folder, "checkpoint_final.pth")):
        return "validating"
    return f"ep {last + 1}/{total_epochs}"


def main() -> None:
    schedules = {"StrokeTrainerLong": config.NUM_EPOCHS_LONG}
    for dataset, (_, ds_name) in paths.DATASETS.items():
        root = paths.NNUNET_RESULTS / ds_name
        if not root.is_dir():
            continue
        print(f"\n{ds_name}")
        print(f"{'model':26s} " + " ".join(f"{'fold ' + str(k):>12s}" for k in range(8)))
        seen = set()
        for model, (plans, configuration, trainer) in config.ALL_MODELS.items():
            base = root / f"{trainer}__{plans}__{configuration}"
            seen.add(base.name)
            if not base.is_dir():
                continue
            total = schedules.get(trainer, config.NUM_EPOCHS)
            cells = [fold_state(str(base / f"fold_{k}"), total) if (base / f"fold_{k}").is_dir() else "-" for k in range(8)]
            print(f"{model:26s} " + " ".join(f"{c:>12s}" for c in cells))
        for other in sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in seen):
            trainer = other.split("__")[0]
            total = schedules.get(trainer, config.NUM_EPOCHS)
            cells = [fold_state(str(root / other / f"fold_{k}"), total) if (root / other / f"fold_{k}").is_dir() else "-" for k in range(8)]
            print(f"{other[:26]:26s} " + " ".join(f"{c:>12s}" for c in cells))


if __name__ == "__main__":
    main()
