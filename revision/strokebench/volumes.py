#!/usr/bin/env python3
"""Calibrated lesion-volume intervals on real model outputs (PROTOCOL.md, section 8).

Step 1 (--extract): for every subject and fold, reduce nnU-Net's exported lesion
probability map to a 1000-bin histogram and cache it in results/volume_hist/<model>.npz.
Step 2 (default): coverage, width, protocol-shift and decision analyses from the cache.

Writes results/volumes_<model>.json and prints a summary. Uses only out-of-fold predictions.
"""
import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from strokebench import conformal as C, config, paths

ALPHAS = (0.20, 0.10, 0.05)
THRESHOLDS_ML = (21.0, 31.0, 51.0, 70.0)            # fixed in PROTOCOL.md amendment 2026-11-22
DECISION_ALPHA = 0.10
SCORES = ("absolute", "logratio", "threshold")
VOLUME_BINS = [(0, 1, "<1 mL"), (1, 10, "1-10 mL"), (10, 70, "10-70 mL"), (70, float("inf"), ">70 mL")]
HIST_DIR = paths.REVISION / "results" / "volume_hist"


def result_dir(model: str) -> Path:
    plans, configuration, trainer = config.ALL_MODELS[model]
    return paths.NNUNET_RESULTS / paths.DATASETS["dwi_adc"][1] / f"{trainer}__{plans}__{configuration}"


def _hist_one(path: str) -> np.ndarray:
    p = np.load(path)["probabilities"][1]
    return C.probability_histogram(p)


def extract(model: str, folds, workers: int) -> None:
    splits = json.load(open(paths.SPLITS_JSON))
    jobs = []
    for fold in folds:
        for case in splits[fold]["val"]:
            path = result_dir(model) / f"fold_{fold}" / "validation" / f"{case}.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            jobs.append((fold, case, str(path)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        hists = list(ex.map(_hist_one, [j[2] for j in jobs], chunksize=4))
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    out = HIST_DIR / f"{model}.npz"
    old = dict(np.load(out)) if out.exists() else {}
    for (fold, case, _), h in zip(jobs, hists):
        old[f"{fold}/{case}"] = h
    np.savez_compressed(out, **old)
    print(f"{model}: cached {len(jobs)} histograms for folds {list(folds)} -> {out}")


def exact_ci(k: int, n: int):
    if n == 0:
        return float("nan"), float("nan")
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    return float(ci.low), float(ci.high)


def score_and_interval(kind: str, v_ref: float, v_hat: float, hist, voxel_ml: float):
    if kind == "absolute":
        return C.score_absolute(v_ref, v_hat), lambda q: C.interval_absolute(v_hat, q)
    if kind == "logratio":
        return C.score_logratio(v_ref, v_hat), lambda q: C.interval_logratio(v_hat, q)
    return C.score_threshold(v_ref, hist, voxel_ml), lambda q: C.interval_threshold(hist, voxel_ml, q)


def subjects(model: str, folds, cases: dict):
    data = np.load(HIST_DIR / f"{model}.npz")
    splits = json.load(open(paths.SPLITS_JSON))
    out = []
    for fold in folds:
        for case in splits[fold]["val"]:
            h = data[f"{fold}/{case}"]
            vox = float(cases[case]["voxel_ml"])
            out.append({"case": case, "fold": fold, "protocol": cases[case]["protocol"], "voxel_ml": vox, "hist": h,
                        "v_ref": float(cases[case]["lesion_ml"]), "v_hat": C.point_volume(h, vox)})
    return out


def summarise_coverage(rows, lower, upper):
    v = np.array([r["v_ref"] for r in rows])
    lo, hi = np.array(lower), np.array(upper)
    hit = (v >= lo) & (v <= hi)
    width = hi - lo
    finite = np.isfinite(width)
    rel = width[finite & (v > 0)] / v[finite & (v > 0)]
    k, n = int(hit.sum()), len(rows)
    c_lo, c_hi = exact_ci(k, n)
    return {"n": n, "coverage": k / n if n else float("nan"), "ci_low": c_lo, "ci_high": c_hi,
            "median_width_ml": float(np.median(width[finite])) if finite.any() else float("inf"),
            "median_relative_width": float(np.median(rel)) if len(rel) else float("nan"),
            "n_unbounded": int((~finite).sum())}


def cv_analysis(rows):
    """Split conformal within each cross-validation fold (leave-one-out calibration)."""
    out = {}
    groups = [r["fold"] for r in rows]
    for kind in SCORES:
        pairs = [score_and_interval(kind, r["v_ref"], r["v_hat"], r["hist"], r["voxel_ml"]) for r in rows]
        scores = [p[0] for p in pairs]
        out[kind] = {}
        for alpha in ALPHAS:
            q = C.leave_one_out_quantiles(scores, groups, alpha)
            iv = [pairs[i][1](q[i]) for i in range(len(rows))]
            lo, hi = [x[0] for x in iv], [x[1] for x in iv]
            block = {"overall": summarise_coverage(rows, lo, hi)}
            for g in ("A", "B1", "B2"):
                idx = [i for i, r in enumerate(rows) if r["protocol"] == g]
                block[f"protocol {g}"] = summarise_coverage([rows[i] for i in idx], [lo[i] for i in idx], [hi[i] for i in idx])
            for a, b, label in VOLUME_BINS:
                idx = [i for i, r in enumerate(rows) if a <= r["v_ref"] < b]
                block[f"volume {label}"] = summarise_coverage([rows[i] for i in idx], [lo[i] for i in idx], [hi[i] for i in idx])
            out[kind][str(alpha)] = block
            if alpha == DECISION_ALPHA:
                out[kind]["_intervals_90"] = (lo, hi)
    return out


def mondrian_analysis(rows):
    """Group-wise calibration by protocol, pooling out-of-fold scores (cross-conformal, weaker guarantee)."""
    out = {}
    thick = np.array([r["protocol"] != "A" for r in rows])
    for kind in SCORES:
        pairs = [score_and_interval(kind, r["v_ref"], r["v_hat"], r["hist"], r["voxel_ml"]) for r in rows]
        scores = np.array([p[0] for p in pairs])
        out[kind] = {}
        for alpha in ALPHAS:
            q = C.leave_one_out_quantiles(scores, thick.astype(int), alpha)
            iv = [pairs[i][1](q[i]) for i in range(len(rows))]
            block = {}
            for name, mask in (("thin-slice", ~thick), ("thick-slice", thick)):
                idx = np.flatnonzero(mask)
                block[name] = summarise_coverage([rows[i] for i in idx], [iv[i][0] for i in idx], [iv[i][1] for i in idx])
            out[kind][str(alpha)] = block
    return out


def shift_analysis(rows, n_draws: int = 200, seed: int = 20261122):
    """Fold 5: calibrate on held-out thin-slice subjects, evaluate on thick-slice subjects."""
    src = [r for r in rows if r["protocol"] == "A"]
    tgt = [r for r in rows if r["protocol"] != "A"]
    rng = np.random.default_rng(seed)
    out = {}
    for kind in SCORES:
        sp = [score_and_interval(kind, r["v_ref"], r["v_hat"], r["hist"], r["voxel_ml"]) for r in src]
        tp = [score_and_interval(kind, r["v_ref"], r["v_hat"], r["hist"], r["voxel_ml"]) for r in tgt]
        out[kind] = {}
        for alpha in ALPHAS:
            q = C.conformal_quantile([p[0] for p in sp], alpha)
            iv = [p[1](q) for p in tp]
            block = {"source_calibrated": summarise_coverage(tgt, [x[0] for x in iv], [x[1] for x in iv])}
            for k in (10, 20, 30):
                covs, widths = [], []
                for _ in range(n_draws):
                    cal = rng.choice(len(tgt), size=k, replace=False)
                    rest = np.setdiff1d(np.arange(len(tgt)), cal)
                    qk = C.conformal_quantile([tp[i][0] for i in cal], alpha)
                    ivk = [tp[i][1](qk) for i in rest]
                    s = summarise_coverage([tgt[i] for i in rest], [x[0] for x in ivk], [x[1] for x in ivk])
                    covs.append(s["coverage"])
                    widths.append(s["median_relative_width"])
                block[f"target_calibrated_k{k}"] = {"mean_coverage": float(np.mean(covs)),
                                                    "coverage_p05": float(np.percentile(covs, 5)),
                                                    "median_relative_width": float(np.nanmedian(widths))}
            out[kind][str(alpha)] = block
    return out


def decision_analysis(rows, lower, upper):
    out = {}
    v = np.array([r["v_ref"] for r in rows])
    vh = np.array([r["v_hat"] for r in rows])
    lo, hi = np.array(lower), np.array(upper)
    for t in THRESHOLDS_ML:
        truth = v > t
        point_err = int(np.sum((vh > t) != truth))
        above, below = lo > t, hi <= t
        determined = above | below
        wrong = int(np.sum(above & ~truth) + np.sum(below & truth))
        out[str(t)] = {"n_truly_above": int(truth.sum()), "point_errors": point_err,
                       "interval_determined": int(determined.sum()), "interval_errors": wrong,
                       "undetermined": int((~determined).sum()),
                       "point_errors_among_undetermined": int(np.sum(((vh > t) != truth) & ~determined))}
    return out


FIXED_BANDS = (0.10, 0.20, 0.50)


def _rule_outcome(flag: np.ndarray, point_wrong: np.ndarray) -> dict:
    return {"flagged": int(flag.sum()), "errors_left": int((point_wrong & ~flag).sum()),
            "errors_caught": int((point_wrong & flag).sum())}


def decision_baselines(rows, lower, upper) -> dict:
    """Compare the interval rule with simple flagging rules (PROTOCOL.md amendment 2026-11-22, exploratory)."""
    v = np.array([r["v_ref"] for r in rows])
    vh = np.array([r["v_hat"] for r in rows])
    lo, hi = np.array(lower), np.array(upper)
    out = {}
    for t in THRESHOLDS_ML:
        wrong = (vh > t) != (v > t)
        interval_flag = (lo <= t) & (hi > t)
        n = int(interval_flag.sum())
        res = {"point_errors": int(wrong.sum()), "interval": _rule_outcome(interval_flag, wrong)}
        for name, dist in (("ratio_matched", np.abs(np.log((vh + C.EPS_ML) / (t + C.EPS_ML)))),
                           ("ml_matched", np.abs(vh - t))):
            flag = np.zeros(len(vh), dtype=bool)
            flag[np.argsort(dist, kind="stable")[:n]] = True
            res[name] = _rule_outcome(flag, wrong)
            res[name]["overlap_with_interval"] = int((flag & interval_flag).sum())
        for b in FIXED_BANDS:
            res[f"fixed_band_{int(b * 100)}pct"] = _rule_outcome(np.abs(vh - t) <= b * t, wrong)
        out[str(t)] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["nnunet", "resenc_m", "umamba_bot", "omnimamba_nn", "unetr"])
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cases = {r["case"]: r for r in csv.DictReader(open(paths.CASES_CSV))}

    if args.extract:
        for m in args.models:
            folds = [0, 1, 2, 3, 4] + ([5] if (result_dir(m) / "fold_5" / "validation" / "summary.json").exists() else [])
            extract(m, folds, args.workers)
        return

    for m in args.models:
        rows = subjects(m, [0, 1, 2, 3, 4], cases)
        point = np.array([r["v_hat"] for r in rows])
        ref = np.array([r["v_ref"] for r in rows])
        cv = cv_analysis(rows)
        res = {"model": m, "n": len(rows), "point_mae_ml": float(np.mean(np.abs(point - ref))),
               "cross_validation": {k: {a: b for a, b in v.items() if not a.startswith("_")} for k, v in cv.items()},
               "mondrian_by_protocol": mondrian_analysis(rows),
               "decision": {k: decision_analysis(rows, *cv[k]["_intervals_90"]) for k in SCORES},
               "decision_baselines": decision_baselines(rows, *cv["logratio"]["_intervals_90"])}
        hist = np.load(HIST_DIR / f"{m}.npz")
        if any(key.startswith("5/") for key in hist.files):
            res["protocol_shift_fold5"] = shift_analysis(subjects(m, [5], cases))
        with open(paths.REVISION / "results" / f"volumes_{m}.json", "w") as f:
            json.dump(res, f, indent=1)

        print(f"\n===== {m}  (n = {len(rows)}, point-estimate mean absolute error {res['point_mae_ml']:.2f} mL)")
        print(f"{'score':10s} {'level':>6s} {'coverage':>9s} {'95% CI':>15s} {'med width mL':>13s} {'med rel width':>14s}")
        for k in SCORES:
            for a in ALPHAS:
                o = res["cross_validation"][k][str(a)]["overall"]
                print(f"{k:10s} {1 - a:6.0%} {o['coverage']:9.3f} [{o['ci_low']:.3f}, {o['ci_high']:.3f}] "
                      f"{o['median_width_ml']:13.2f} {o['median_relative_width']:14.2f}")
        print("coverage at 90% by stratum:  " + " | ".join(
            f"{k}: " + ", ".join(f"{s.split(' ', 1)[1]} {res['cross_validation'][k]['0.1'][s]['coverage']:.2f}"
                                 for s in res["cross_validation"][k]["0.1"] if s != "overall") for k in SCORES))


if __name__ == "__main__":
    main()
