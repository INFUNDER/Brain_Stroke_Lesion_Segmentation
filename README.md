# Stroke lesion segmentation on ISLES 2022: a leakage-free backbone benchmark

This repository holds a stroke lesion segmentation study on the ISLES 2022 MRI dataset (250 subjects, DWI and ADC). It started as the **OmniMamba-UNet** paper: a 3-D U-Net with tri-planar Mamba (selective state space) layers. That version was audited and found to have invalid results. The study has been re-done from scratch under a pre-registered protocol, and the new work is in [`revision/`](revision/).

> **Status (2026-11-24): results are interim.** Six of the seven pre-specified models are trained and scored. The original OmniMamba design and two protocol-shift runs are still training.

---

## What the revised study asks

1. **Does the backbone matter?** It compares seven backbones under one nnU-Net v2 pipeline, with the same folds, preprocessing, patch size and training budget, using five-fold cross-validation on all 250 subjects. The seven are: nnU-Net, ResEnc-M, U-Mamba-Bot, OmniMamba (two variants), SwinUNETR and UNETR.
2. **Does scanner protocol matter?** 196 subjects have thin-slice scans and 54 have thick-slice scans (4 mm or more). Dedicated splits train on one protocol and test on the other.
3. **How far can a predicted lesion volume be trusted?** Conformal prediction gives each subject an interval around the predicted volume. The study checks it against treatment-eligibility thresholds of 21, 31, 51 and 70 mL.

The analysis plan was fixed in [`revision/PROTOCOL.md`](revision/PROTOCOL.md) before any full training run. Later changes are recorded as dated amendments at the end of that file.

## Why the first version was withdrawn

The full audit is in [`revision/AUDIT.md`](revision/AUDIT.md). The main problems:

- **47 of 250 labels were blank.** Those ISLES masks are stored as `int16` with a scale factor, so their lesion value reads as 0.99999…, and an integer cast turns it into 0. Almost all of them were thick-slice scans (43 of 54). The fix is to binarise with `> 0.5`, which [`revision/01_build_corrected_labels.py`](revision/01_build_corrected_labels.py) does. **Anyone using ISLES 2022 should check for this.**
- **The "lesion-free mimic" cohort did not exist.** Three of the four were real lesions with blanked masks.
- **The UNETR baseline had seen the test subjects** during training or checkpoint selection.
- **The post-processing settings were tuned on the test set.**
- **The parameter counts in the paper were wrong.** The real counts are 35.0M against 127.4M, not 12.3M against 101.9M.

None of the numbers, checkpoints or labels from that version are used in the revision.

---

## Interim results

Cross-validation on 247 subjects with a lesion, all models trained for 250 epochs. Differences are paired against nnU-Net; "equivalent" means the 90% CI lies within ±0.02 Dice.

| Model | Mean Dice | Difference vs nnU-Net [95% CI] | Holm p | Equivalent |
|---|---|---|---|---|
| U-Mamba-Bot | 0.791 | +0.009 [−0.001, +0.019] | 0.025 | yes |
| OmniMamba (in nnU-Net) | 0.790 | +0.008 [+0.000, +0.016] | 0.36 | yes |
| ResEnc-M | 0.786 | +0.003 [−0.003, +0.011] | 0.36 | yes |
| nnU-Net (reference) | 0.782 | — | — | — |
| SwinUNETR | 0.776 | −0.006 [−0.016, +0.003] | 0.36 | yes |
| UNETR | 0.739 | −0.043 [−0.055, −0.033] | <0.001 | no |
| OmniMamba (original design) | training | | | |

Supplementary runs, which are outside the Holm family:

- **Run-to-run noise:** a second nnU-Net run with a different seed differs by only −0.002 Dice.
- **Longer training:** 1000-epoch runs of nnU-Net (0.777) and ResEnc-M (0.778) score no higher than their 250-epoch runs.

**What this shows so far:**

- Five of the six backbones scored so far are within about 0.01 Dice of a well-tuned nnU-Net. That gap is close to run-to-run noise.
- UNETR is clearly worse.
- No backbone, including OmniMamba, is a meaningful improvement.

**Protocol shift (nnU-Net):**

- Models trained on thin-slice scans segment thick-slice scans about as well as thin-slice ones.
- The reverse direction is weaker. Training on the 54 thick-slice subjects and testing on thin-slice scans scores 0.63. Training on 54 randomly chosen thin-slice subjects scores 0.72.

**Volume intervals:**

- **Log-ratio score:** coverage within each subgroup is 0.85–0.93, against 0.90 targeted. A typical interval runs from about 0.55 to 1.8 times the predicted volume.
- **Absolute (mL) score:** these intervals cover only 32% of lesions over 70 mL.
- **Treatment thresholds:** at the same number of flagged patients, the intervals catch no more threshold errors than a simple ranking by distance from the threshold.
- **Adaptive intervals:** these scale with each subject's probability spread. They met their pre-registered test narrowly, but a bootstrap does not rule out chance.

Machine-readable results are in [`revision/results/`](revision/results/).

---

## Repository layout

| Path | Contents |
|---|---|
| [`revision/`](revision/) | **The current study.** Code, protocol, audit, job scripts, tests and results. See [`revision/README.md`](revision/README.md) |
| `revision/strokebench/` | Networks, nnU-Net trainers, metrics, folds, statistics, conformal intervals |
| `revision/results/` | Per-subject scores, model comparison, protocol-shift and volume analyses |
| `paper.tex`, `paper.pdf` | Manuscript of the first version. **Its results are invalid** (see above) |
| `new_work/`, `experiments/`, `hpc_scripts/`, `evaluate.py`, `run_final_test.py`, `tune_cc_non_leaked.py`, `figures/` | Code and figures of the first version, kept for the record |

Not in the repository: the ISLES 2022 images and masks ([download from the challenge](https://isles22.grand-challenge.org/)), the corrected masks (`ISLES2022_Corrected/`, rebuilt by a script), and nnU-Net's preprocessed data and trained models (`nnunet/`).

## Reproducing

```bash
bash revision/env/setup_env.sh
```

```bash
source revision/env/activate.sh
```

```bash
python revision/01_build_corrected_labels.py
```

[`revision/README.md`](revision/README.md) has the rest of the steps: building the dataset and folds, nnU-Net planning, PBS training jobs, scoring and analysis.

## Planned before submission

- Finish the original OmniMamba design and the remaining protocol-shift runs, then close the Holm family.
- Add a FLAIR experiment.
- Re-measure speed and memory on an idle GPU.
- Validate externally on an independent cohort.
- Make the figures and rewrite the manuscript.
