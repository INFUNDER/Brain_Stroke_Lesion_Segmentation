#!/usr/bin/env python3
"""Protocol-shift analysis (PROTOCOL.md, section 7).

Uses the cross-validation scores (results/per_case) and the shift-fold scores
(results/per_case_shift, folds 5-7). All contrasts are paired by subject:

  thick-slice cost  fold 5 (trained on thin-slice only) minus cross-validation, on the 54 thick-slice subjects
  thin-slice check  the same difference on the 40 held-out thin-slice subjects of fold 5 (should be near 0)
  protocol effect   fold 6 (trained on 54 thick-slice) minus fold 7 (trained on 54 thin-slice),
                    on the thin-slice subjects that neither trained on
  size effect       fold 7 minus cross-validation on those same subjects (54 vs about 200 training subjects)

Between-model contrasts (difference of paired differences against nnU-Net) are exploratory.
"""
import csv
import json
from pathlib import Path

import numpy as np

from strokebench import paths, stats

MODELS = ["nnunet", "resenc_m", "umamba_bot", "omnimamba_nn", "swinunetr"]


def load(path: Path, key="case"):
    if not path.exists():
        return None
    out = {}
    for r in csv.DictReader(open(path)):
        out.setdefault(int(r["fold"]), {})[r[key]] = r
    return out


def contrast(a: dict, b: dict, cases):
    cases = sorted(c for c in cases if c in a and c in b and not int(a[c]["ref_empty"]))
    diff = np.array([float(a[c]["dice"]) - float(b[c]["dice"]) for c in cases])
    boot = stats.paired_bootstrap(diff)
    return {**boot, "n": len(cases), "mean_a": float(np.mean([float(a[c]["dice"]) for c in cases])),
            "mean_b": float(np.mean([float(b[c]["dice"]) for c in cases])), "_diff": dict(zip(cases, diff))}


def main() -> None:
    rows = {r["case"]: r for r in csv.DictReader(open(paths.CASES_CSV))}
    splits = json.load(open(paths.SPLITS_JSON))
    thick = [c for c in rows if rows[c]["protocol"] != "A"]
    thin_cal = [c for c in splits[5]["val"] if rows[c]["protocol"] == "A"]
    thin_unseen = [c for c in splits[7]["val"] if rows[c]["protocol"] == "A"]
    out = {}
    for m in MODELS:
        cv = load(paths.REVISION / "results" / "per_case" / f"{m}_dwi_adc.csv")
        sh = load(paths.REVISION / "results" / "per_case_shift" / f"{m}_dwi_adc.csv")
        if not cv or not sh or not {5, 6, 7} <= set(sh):
            continue
        cv_all = {c: r for f in cv.values() for c, r in f.items()}
        out[m] = {
            "thick_slice_cost": contrast(sh[5], cv_all, thick),
            "thin_slice_check": contrast(sh[5], cv_all, thin_cal),
            "protocol_effect": contrast(sh[6], sh[7], thin_unseen),
            "size_effect": contrast(sh[7], cv_all, thin_unseen),
        }
    ref = out.get("nnunet")
    print(f"{'model':12s} {'contrast':18s} {'n':>4s} {'Dice A':>7s} {'Dice B':>7s} {'A - B':>8s}  95% CI")
    for m, block in out.items():
        for name, r in block.items():
            print(f"{m:12s} {name:18s} {r['n']:4d} {r['mean_a']:7.3f} {r['mean_b']:7.3f} {r['mean_diff']:+8.4f}  "
                  f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]")
        if ref and m != "nnunet":
            for name in ("thick_slice_cost", "protocol_effect"):
                d, dr = block[name]["_diff"], ref[name]["_diff"]
                common = sorted(set(d) & set(dr))
                b = stats.paired_bootstrap(np.array([d[c] - dr[c] for c in common]))
                block[name]["vs_nnunet"] = {k: b[k] for k in ("mean_diff", "ci95_low", "ci95_high")}
                print(f"{'':12s} {name + ' vs nnU-Net':30s} {b['mean_diff']:+8.4f}  [{b['ci95_low']:+.4f}, {b['ci95_high']:+.4f}]  (exploratory)")
    for block in out.values():
        for r in block.values():
            r.pop("_diff")
    with open(paths.REVISION / "results" / "shift_dwi_adc.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
