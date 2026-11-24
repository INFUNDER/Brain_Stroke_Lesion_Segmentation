#!/usr/bin/env python3
"""Backbone comparison on out-of-fold predictions (PROTOCOL.md, section 6).

Reads revision/results/per_case/<model>_<dataset>.csv for every model that has all
five cross-validation folds, and writes:
  results/comparison_<dataset>.json   paired differences against the reference
  results/strata_<dataset>.csv        Dice by protocol group and lesion-volume stratum

Holm correction is applied over the comparisons that exist when the script runs.
Until all six pre-specified comparisons exist, the output is marked "interim".
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from strokebench import config, paths, stats

REFERENCE = "nnunet"
PRESPECIFIED = [m for m in config.MODELS if m != REFERENCE]          # the six comparisons of section 6
VOLUME_BINS = [(0, 1, "<1 mL"), (1, 10, "1-10 mL"), (10, 70, "10-70 mL"), (70, float("inf"), ">70 mL")]
METRICS = ["dice", "lesion_f1", "abs_volume_diff_ml", "abs_lesion_count_diff"]


def load(model: str, dataset: str, folder: Path):
    path = folder / f"{model}_{dataset}.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    if sorted({int(r["fold"]) for r in rows}) != [0, 1, 2, 3, 4] or len(rows) != paths.N_CASES:
        return None
    return {r["case"]: r for r in rows}


def mean_ci(values, n_boot=10_000, seed=stats.SEED):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dwi_adc")
    ap.add_argument("--per-case", default=str(paths.REVISION / "results" / "per_case"))
    ap.add_argument("--out-dir", default=str(paths.REVISION / "results"))
    args = ap.parse_args()
    folder = Path(args.per_case)

    data = {m: d for m in config.ALL_MODELS if (d := load(m, args.dataset, folder)) is not None}
    # second independent run of the reference (run-to-run noise floor, PROTOCOL.md section 6)
    if (d := load(f"{REFERENCE}_StrokeTrainerReplicate2", args.dataset, folder)) is not None:
        data[f"{REFERENCE}_replicate2"] = d
    if REFERENCE not in data:
        raise SystemExit(f"reference model '{REFERENCE}' has no complete cross-validation results")
    ref = data[REFERENCE]
    positive = sorted(c for c, r in ref.items() if not int(r["ref_empty"]))
    folds = {c: int(ref[c]["fold"]) for c in positive}
    dice = {m: {c: float(d[c]["dice"]) for c in positive} for m, d in data.items()}

    comparisons = [m for m in PRESPECIFIED if m in data]
    result = stats.compare_to_reference(dice[REFERENCE], {m: dice[m] for m in comparisons}, folds)
    supplementary = {}
    for m in data:
        if m not in comparisons and m != REFERENCE:          # long-schedule and ablation models: not in the Holm family
            supplementary[m] = stats.compare_to_reference(dice[REFERENCE], {m: dice[m]}, folds)[m]
            supplementary[m].pop("wilcoxon_p_holm", None)

    summary = {}
    for m, d in data.items():
        pos = [d[c] for c in positive]
        summary[m] = {k: mean_ci([float(r[k]) for r in pos])[0] for k in METRICS}
        summary[m]["dice_median"] = float(np.median([float(r["dice"]) for r in pos]))
        summary[m]["missed_entirely"] = int(sum(int(r["pred_empty"]) for r in pos))

    status = "final" if set(PRESPECIFIED) <= set(comparisons) else "interim"
    out = {"status": status, "reference": REFERENCE, "n_lesion_positive": len(positive),
           "equivalence_margin": stats.EQUIVALENCE_MARGIN,
           "holm_family": comparisons, "missing_from_family": [m for m in PRESPECIFIED if m not in data],
           "summary": summary, "prespecified": result, "supplementary": supplementary}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / f"comparison_{args.dataset}.json", "w") as f:
        json.dump(out, f, indent=1)

    strata_rows = []
    for m, d in data.items():
        for group in ("A", "B1", "B2"):
            cs = [c for c in positive if ref[c]["protocol"] == group]
            mu, lo, hi = mean_ci([dice[m][c] for c in cs])
            strata_rows.append({"model": m, "stratum": f"protocol {group}", "n": len(cs), "dice": mu, "ci_low": lo, "ci_high": hi})
        for low, high, label in VOLUME_BINS:
            cs = [c for c in positive if low <= float(ref[c]["ref_ml"]) < high]
            mu, lo, hi = mean_ci([dice[m][c] for c in cs])
            strata_rows.append({"model": m, "stratum": f"volume {label}", "n": len(cs), "dice": mu, "ci_low": lo, "ci_high": hi})
    with open(Path(args.out_dir) / f"strata_{args.dataset}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(strata_rows[0]))
        w.writeheader()
        w.writerows(strata_rows)

    print(f"[{status}] {len(positive)} lesion-positive subjects, reference = {REFERENCE}")
    if status == "interim":
        print(f"  not yet in the Holm family: {out['missing_from_family']}")
    print(f"\n{'model':16s} {'Dice':>6s} {'median':>7s} {'F1':>6s} {'AVD mL':>7s} {'ALCD':>6s} {'missed':>6s}")
    for m in data:
        s = summary[m]
        print(f"{m:16s} {s['dice']:6.3f} {s['dice_median']:7.3f} {s['lesion_f1']:6.3f} {s['abs_volume_diff_ml']:7.2f} "
              f"{s['abs_lesion_count_diff']:6.2f} {s['missed_entirely']:6d}")
    print(f"\nPaired Dice difference vs {REFERENCE} (95% CI; equivalent = 90% CI within +/-{stats.EQUIVALENCE_MARGIN})")
    for label, block in (("pre-specified", result), ("supplementary", supplementary)):
        for m, r in block.items():
            p = f"p_holm={r['wilcoxon_p_holm']:.3g}" if "wilcoxon_p_holm" in r else f"p={r['wilcoxon_p']:.3g} (uncorrected)"
            print(f"  [{label}] {m:14s} {r['mean_diff']:+.4f} [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]  "
                  f"90% [{r['ci90_low']:+.4f}, {r['ci90_high']:+.4f}]  equivalent={r['equivalent_within_margin']}  {p}  "
                  f"better in {100 * r['share_subjects_better']:.0f}% of subjects")


if __name__ == "__main__":
    main()
