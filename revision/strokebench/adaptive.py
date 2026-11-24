#!/usr/bin/env python3
"""Adaptive (per-patient) volume intervals, PROTOCOL.md amendment 2026-11-22 (third entry).

Writes results/adaptive_volumes.json and prints the pre-specified success criterion.
"""
import csv
import json

import numpy as np
from scipy.stats import spearmanr

from strokebench import conformal as C, paths
from strokebench import volumes as V

MODELS = ["nnunet", "resenc_m", "umamba_bot", "omnimamba_nn", "unetr"]
SPREADS = {"primary_0.1_0.9": (0.1, 0.9), "sensitivity_0.25_0.75": (0.25, 0.75)}


def matched(dist, n, wrong):
    flag = np.zeros(len(dist), dtype=bool)
    flag[np.argsort(dist, kind="stable")[:n]] = True
    return V._rule_outcome(flag, wrong)


def analyse(rows, low, high):
    v = np.array([r["v_ref"] for r in rows])
    vh = np.array([r["v_hat"] for r in rows])
    s = np.array([C.spread(r["hist"], r["voxel_ml"], low, high) for r in rows])
    err = np.array([C.score_logratio(a, b) for a, b in zip(v, vh)])
    rho = spearmanr(s, err)
    scores = [C.score_adaptive(a, b, c) for a, b, c in zip(v, vh, s)]
    q = C.leave_one_out_quantiles(scores, [r["fold"] for r in rows], V.DECISION_ALPHA)
    iv = [C.interval_adaptive(b, c, qq) for b, c, qq in zip(vh, s, q)]
    lo, hi = [x[0] for x in iv], [x[1] for x in iv]
    cov = {"overall": V.summarise_coverage(rows, lo, hi)}
    for g in ("A", "B1", "B2"):
        idx = [i for i, r in enumerate(rows) if r["protocol"] == g]
        cov[f"protocol {g}"] = V.summarise_coverage([rows[i] for i in idx], [lo[i] for i in idx], [hi[i] for i in idx])
    for a, b, label in V.VOLUME_BINS:
        idx = [i for i, r in enumerate(rows) if a <= r["v_ref"] < b]
        cov[f"volume {label}"] = V.summarise_coverage([rows[i] for i in idx], [lo[i] for i in idx], [hi[i] for i in idx])
    decisions = {}
    lo, hi = np.array(lo), np.array(hi)
    for t in V.THRESHOLDS_ML:
        wrong = (vh > t) != (v > t)
        flag = (lo <= t) & (hi > t)
        n = int(flag.sum())
        decisions[str(t)] = {
            "point_errors": int(wrong.sum()),
            "adaptive": V._rule_outcome(flag, wrong),
            "ratio_matched": matched(np.abs(np.log((vh + C.EPS_ML) / (t + C.EPS_ML))), n, wrong),
            "ml_matched": matched(np.abs(vh - t), n, wrong),
        }
    return {"spearman_spread_vs_error": float(rho.correlation), "spearman_p": float(rho.pvalue),
            "coverage_90": cov, "decision_90": decisions}


def main():
    cases = {r["case"]: r for r in csv.DictReader(open(paths.CASES_CSV))}
    out, pooled = {}, {}
    for m in MODELS:
        rows = V.subjects(m, [0, 1, 2, 3, 4], cases)
        out[m] = {name: analyse(rows, lo, hi) for name, (lo, hi) in SPREADS.items()}
        fixed = json.load(open(paths.REVISION / "results" / f"volumes_{m}.json"))["cross_validation"]["logratio"]["0.1"]
        for name in SPREADS:
            r = out[m][name]
            c = r["coverage_90"]
            print(f"{m:13s} {name:22s} rho(spread, error)={r['spearman_spread_vs_error']:+.2f} (p={r['spearman_p']:.1g}) | "
                  f"median rel width {c['overall']['median_relative_width']:.2f} (fixed log-ratio {fixed['overall']['median_relative_width']:.2f}) | "
                  + ", ".join(f"{k.split(' ', 1)[1]} {v['coverage']:.2f}" for k, v in c.items() if k != "overall"))
            for t, d in r["decision_90"].items():
                for rule in ("adaptive", "ratio_matched", "ml_matched"):
                    acc = pooled.setdefault(name, {}).setdefault(rule, [0, 0, 0])
                    acc[0] += d[rule]["flagged"]; acc[1] += d[rule]["errors_caught"]; acc[2] += d[rule]["errors_left"]
    print("\nPooled over 5 models x 4 thresholds, 90% level: flagged / errors caught / errors left")
    verdict = {}
    for name, rules in pooled.items():
        for rule, (f, c, e) in rules.items():
            print(f"  {name:22s} {rule:14s} {f:5d} / {c:3d} / {e:3d}")
        best_baseline = max(rules["ratio_matched"][1], rules["ml_matched"][1])
        verdict[name] = {"adaptive_caught": rules["adaptive"][1], "best_matched_baseline_caught": best_baseline,
                         "success": rules["adaptive"][1] > best_baseline}
    print("\nPre-specified criterion (primary spread only):", verdict["primary_0.1_0.9"])
    with open(paths.REVISION / "results" / "adaptive_volumes.json", "w") as f:
        json.dump({"per_model": out, "pooled": pooled, "criterion": verdict}, f, indent=1)


if __name__ == "__main__":
    main()
