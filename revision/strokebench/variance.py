#!/usr/bin/env python3
"""E4c: how much of the variation in Dice is due to the subject and how much to the backbone
(PROTOCOL.md, amendment 2026-11-26, second entry).

Two-way decomposition without replication over the subjects x backbones matrix of
per-subject Dice (lesion-positive subjects, backbones with complete cross-validation):

  SS_total = SS_subject + SS_backbone + SS_residual

The residual is the subject-by-backbone interaction (which backbone suits which subject).
The share of subject variance explained by protocol group and log reference volume is the
R^2 of an ordinary least-squares fit to the subject means. Bootstrap CIs resample subjects.

    python -m strokebench.variance
"""
import csv
import json

import numpy as np

from strokebench import config, paths, stats

FOLDER = paths.REVISION / "results" / "per_case"


def decompose(x: np.ndarray) -> dict:
    """x: subjects x backbones. Shares of the total sum of squares."""
    n, k = x.shape
    grand = x.mean()
    ss_total = float(((x - grand) ** 2).sum())
    ss_subject = float(k * ((x.mean(axis=1) - grand) ** 2).sum())
    ss_backbone = float(n * ((x.mean(axis=0) - grand) ** 2).sum())
    ss_resid = ss_total - ss_subject - ss_backbone
    return {"subject": ss_subject / ss_total, "backbone": ss_backbone / ss_total, "residual": ss_resid / ss_total}


def r2_subject_means(means: np.ndarray, protocol: np.ndarray, log_vol: np.ndarray) -> dict:
    groups = sorted(set(protocol))
    design = [np.ones(len(means))] + [(protocol == g).astype(float) for g in groups[1:]]
    out = {}
    for label, cols in (("protocol", design), ("log_volume", [np.ones(len(means)), log_vol]),
                        ("protocol+log_volume", design + [log_vol])):
        X = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(X, means, rcond=None)
        resid = means - X @ beta
        out[label] = float(1 - (resid ** 2).sum() / ((means - means.mean()) ** 2).sum())
    return out


def main() -> None:
    models = [m for m in config.MODELS if (FOLDER / f"{m}_dwi_adc.csv").exists()]
    data = {m: {r["case"]: r for r in csv.DictReader(open(FOLDER / f"{m}_dwi_adc.csv"))} for m in models}
    models = [m for m in models if len(data[m]) == paths.N_CASES]
    ref = data[models[0]]
    cases = sorted(c for c, r in ref.items() if not int(r["ref_empty"]))
    x = np.array([[float(data[m][c]["dice"]) for m in models] for c in cases])
    protocol = np.array([ref[c]["protocol"] for c in cases])
    log_vol = np.log(np.array([float(ref[c]["ref_ml"]) for c in cases]) + 0.1)

    point = decompose(x)
    point_r2 = r2_subject_means(x.mean(axis=1), protocol, log_vol)
    rng = np.random.default_rng(stats.SEED)
    boots, boots_r2 = [], []
    for _ in range(2000):
        idx = rng.integers(0, len(cases), len(cases))
        boots.append(decompose(x[idx]))
        boots_r2.append(r2_subject_means(x[idx].mean(axis=1), protocol[idx], log_vol[idx]))
    ci = lambda vals: [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    out = {"models": models, "n_subjects": len(cases),
           "shares": {k: {"estimate": v, "ci95": ci([b[k] for b in boots])} for k, v in point.items()},
           "subject_variance_explained": {k: {"estimate": v, "ci95": ci([b[k] for b in boots_r2])} for k, v in point_r2.items()},
           "sd_dice": {"between_subject_means": float(x.mean(axis=1).std(ddof=1)),
                       "between_backbone_means": float(x.mean(axis=0).std(ddof=1))}}
    path = paths.REVISION / "results" / "variance_dwi_adc.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"{len(cases)} lesion-positive subjects x {len(models)} backbones: {models}")
    for k, v in out["shares"].items():
        print(f"  share of Dice variance due to {k:9s} {100 * v['estimate']:5.1f}%  [{100 * v['ci95'][0]:.1f}, {100 * v['ci95'][1]:.1f}]")
    for k, v in out["subject_variance_explained"].items():
        print(f"  subject-mean variance explained by {k:20s} R2={v['estimate']:.2f}  [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]")
    print(f"  SD of subject means {out['sd_dice']['between_subject_means']:.3f}; SD of backbone means {out['sd_dice']['between_backbone_means']:.3f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
