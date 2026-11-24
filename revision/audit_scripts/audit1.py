#!/usr/bin/env python3
"""Read-only audit: split overlap, raw vs formatted labels, per-case re-analysis."""
import os, glob, json, re, sys
import numpy as np
import nibabel as nib
from scipy import stats
from scipy.ndimage import label as cc

ROOT = "/home/ronit.28010/Brain_Stroke"
RES = f"{ROOT}/mamba_results"

# --- A. which test subjects did the UNETR script train/validate on?
imgs = sorted(glob.glob(f"{ROOT}/ISLES2022_Formatted/imagesTr/*.nii.gz"))
names = [os.path.basename(p).split(".nii")[0].replace("_0000", "") for p in imgs]
print("n images", len(imgs), "first", names[:3], "last", names[-3:])
split = int(0.8 * len(imgs))
unetr_train = set(names[:split])
unetr_val = set(names[split:])
test = []
for line in open(f"{RES}/test_set_paths.txt"):
    parts = line.strip().split("  |  ")
    if len(parts) == 2:
        test.append(os.path.basename(parts[1]).split(".nii")[0])
print("test n", len(test), test[:3])
in_tr = sorted(set(test) & unetr_train)
in_val = sorted(set(test) & unetr_val)
print(f"A. test subjects in UNETR TRAIN: {len(in_tr)}; in UNETR VAL (checkpoint selection): {len(in_val)}; neither: {len(set(test) - unetr_train - unetr_val)}")
print("   in val:", in_val)

# --- B. empty labels: formatted vs raw; dataset geometry
lbls = sorted(glob.glob(f"{ROOT}/ISLES2022_Formatted/labelsTr/*.nii.gz"))
raw_root = f"{ROOT}/isless2022dataset/ISLES-2022_notformatted"
print("raw listing:", sorted(os.listdir(raw_root))[:4], len(os.listdir(raw_root)))
sample_sub = sorted(d for d in os.listdir(raw_root) if d.startswith("sub-"))[0]
for dp, dn, fn in os.walk(os.path.join(raw_root, sample_sub)):
    print("  ", dp.replace(raw_root, ""), fn[:6])

rows = []
for l in lbls:
    name = os.path.basename(l).split(".nii")[0]
    nii = nib.load(l)
    arr = np.asanyarray(nii.dataobj)
    vox = float(np.prod(nii.header.get_zooms()[:3]))
    fg = int((arr > 0).sum())
    ncc = int(cc(arr > 0)[1]) if fg else 0
    # raw mask lookup
    candidates = glob.glob(f"{raw_root}/**/{name}*msk*.nii*", recursive=True) + glob.glob(f"{raw_root}/derivatives/{name}/**/*.nii*", recursive=True)
    raw_fg = None
    if candidates:
        r = nib.load(candidates[0])
        raw_fg = int((np.asanyarray(r.dataobj) > 0).sum())
    rows.append(dict(name=name, fg=fg, vox_mm3=vox, ml=fg * vox / 1000, n_lesions=ncc, raw_fg=raw_fg))
empty = [r for r in rows if r["fg"] == 0]
print(f"B. formatted empties: {len(empty)} / {len(rows)}")
print("   names:", [r["name"] for r in empty])
print("   raw fg for those:", [r["raw_fg"] for r in empty])
mismatch = [(r["name"], r["fg"], r["raw_fg"]) for r in rows if r["raw_fg"] is not None and r["raw_fg"] != r["fg"]]
print("   raw != formatted fg count:", len(mismatch), mismatch[:5])
print("   raw label found for", sum(r['raw_fg'] is not None for r in rows))
vols = np.array([r["ml"] for r in rows if r["fg"] > 0])
print(f"   lesion vol mL: median {np.median(vols):.2f} IQR {np.percentile(vols,25):.2f}-{np.percentile(vols,75):.2f}, min {vols.min():.3f} max {vols.max():.1f}; >70 mL: {(vols>70).sum()}")
print("   n empties in test:", [r["name"] for r in empty if r["name"] in test])
json.dump(rows, open(sys.argv[1] + "/case_table.json", "w"), indent=0)

# --- C. image geometry (formatted)
for p in imgs[:3]:
    n = nib.load(p)
    print("C.", os.path.basename(p), n.shape, n.header.get_zooms())

# --- D. per-case reanalysis of the saved test-set probabilities and UNETR results
un = json.load(open(f"{RES}/results/unetr_test_results.json"))["results"]
un = {r["case_name"]: r for r in un}

def dsc(p, g):
    a, b = p.sum(), g.sum()
    if b == 0:
        return np.nan
    if a == 0:
        return 0.0
    return 2 * (p & g).sum() / (a + b)

out = []
for c in sorted(test):
    prob = np.load(f"{RES}/non_leaked_probs/{c}_prob.npy")
    gt = np.load(f"{RES}/non_leaked_probs/{c}_gt.npy") > 0
    p50 = prob >= 0.5
    p45 = prob >= 0.45
    out.append(dict(case=c, gt=int(gt.sum()), d50=dsc(p50, gt), d45=dsc(p45, gt),
                    fp45=int(p45.sum()) if gt.sum() == 0 else None,
                    fp50=int(p50.sum()) if gt.sum() == 0 else None,
                    unet_d=un[c]["corrected_dice"] if gt.sum() > 0 else np.nan,
                    unet_vox=un[c]["pred_vol_ml"]/0.0045,
                    leak=("train" if c in unetr_train else "val")))
pos = [o for o in out if o["gt"] > 0]
neg = [o for o in out if o["gt"] == 0]
print("D. lesion-positive n", len(pos), " empty-GT n", len(neg))
print("   empty-GT: Mamba FP voxels @0.45:", [(o['case'], o['fp45']) for o in neg], " @0.50:", [o['fp50'] for o in neg], " UNETR FP voxels:", [round(o['unet_vox']) for o in neg])
m50 = np.array([o["d50"] for o in pos]); m45 = np.array([o["d45"] for o in pos]); u = np.array([o["unet_d"] for o in pos])
print(f"   lesion-positive mean Dice: OmniMamba@0.50 {m50.mean():.4f}  @0.45 {m45.mean():.4f}  UNETR {u.mean():.4f}")
print(f"   per-case Omni@0.45 minus UNETR: mean {np.mean(m45-u):+.4f}; Omni better in {(m45>u).sum()}/{len(u)}")
print("   two-sided Wilcoxon (Omni@0.45 vs UNETR, lesion-positive): p=", stats.wilcoxon(m45, u, alternative='two-sided').pvalue)
print("   one-sided, UNETR > Omni: p=", stats.wilcoxon(u, m45, alternative='greater').pvalue)
lk = np.array([o['leak'] == 'train' for o in pos])
print(f"   UNETR Dice on test cases it TRAINED on: {u[lk].mean():.4f} (n={lk.sum()}); on cases in its VAL set: {u[~lk].mean():.4f} (n={(~lk).sum()})")
print(f"   OmniMamba@0.45 on same subsets: {m45[lk].mean():.4f} / {m45[~lk].mean():.4f}")
small = np.array([o['gt'] * 0.0045 < 5 for o in pos])
print(f"   <5 mL lesions n={small.sum()}: Omni {m45[small].mean():.4f} UNETR {u[small].mean():.4f}; >=5 mL n={(~small).sum()}: Omni {m45[~small].mean():.4f} UNETR {u[~small].mean():.4f}")
json.dump(out, open(sys.argv[1] + "/per_case.json", "w"), indent=0, default=float)
