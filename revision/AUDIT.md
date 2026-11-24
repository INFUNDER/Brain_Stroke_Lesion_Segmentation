# Audit of the OmniMamba-UNet manuscript

Date: 2026-11-10. Everything below marked "verified" was checked against the raw data, code, checkpoints or job logs in this repository. The scripts and their outputs are in `revision/audit_scripts/` and `revision/audit_data/`.

## Bottom line

The reported results cannot be repaired by rewriting. The labels used for training and evaluation were corrupted for 47 of 250 subjects, the only baseline was trained on the test set, and several statements in the manuscript are contradicted by the project's own code and logs. The experiments have to be re-run on corrected data, and the manuscript has to be rewritten from the new results.

**Do not submit the current manuscript to any journal.** If any version is under review elsewhere, it should be withdrawn or corrected.

Acceptance cannot be promised by anyone. What can be done is to remove the reasons for rejection and build a paper whose value does not depend on one architecture winning.

## A. Problems that invalidate the reported numbers (all verified)

### A1. 47 of 250 labels are blank

- `ISLES2022_Formatted/labelsTr` is empty for 50 subjects. The original ISLES masks are empty for only 3 of them (`0150`, `0151`, `0170`). The other 47 contain real lesions.
- Likely cause: those 47 masks, and none of the other 200, are stored as `int16` with `scl_slope = 1.5259e-05`, so the lesion value reads as `0.9999999998`. An integer cast turns that into 0. The conversion script is not in the repository, so the exact line is inferred from this perfect separation.
- The damage follows the scanner protocol: 43 of 54 thick-slice (4 mm or more) lesion cases were blanked, against 4 of 193 thin-slice cases. Models were taught that real lesions in those scans are background.
- The image channels are fine. DWI and ADC match the raw files, and FLAIR matches a header-based resampling of the raw FLAIR (76 cases checked, r = 1.000).
- **Repaired** in `ISLES2022_Corrected/` by `revision/01_build_corrected_labels.py`: 200 unchanged (voxel-identical), 47 repaired, 3 empty in the source. The original folder was not modified.

### A2. The "lesion-free mimic" cohort does not exist

Of the 4 test "mimics", 3 have real lesions that were blanked. The existing ensemble segmented them well when scored against the true masks:

| Case | True lesion | Dice against true mask |
|---|---|---|
| `0175` | 2.50 mL | 0.869 |
| `0187` | 5.86 mL | 0.817 |
| `0226` | 7.07 mL | 0.764 |

The paper reports these correct detections as false positives on healthy scans. Only 1 test case (`0170`) and 3 cases in the whole cohort are truly empty.

Existing April ensemble against true masks at P = 0.5: **mean Dice 0.771, median 0.816** on 24 lesion-positive test cases. This is not a clean estimate. These models trained on corrupted labels, and the test set has been inspected repeatedly.

### A3. The UNETR baseline was trained on the test set

`experiments/train.py:46-47` splits the sorted file list without shuffling, so UNETR used subjects 1-200 for training and 201-250 for checkpoint selection. **17 of the 25 "strictly non-leaked" test subjects were in its training set and the other 8 in its validation set.** Every UNETR test number is contaminated.

### A4. The headline comparison rests on 2 voxels

The paper reports OmniMamba 0.6855 against UNETR 0.6664. On the 21 lesion-positive cases the order reverses: **OmniMamba 0.7685, UNETR 0.7933** (two-sided Wilcoxon p = 0.84). The apparent advantage comes entirely from `0170`, where UNETR predicted 2 voxels (Dice 0) and OmniMamba predicted none (Dice 1). On lesions under 5 mL, UNETR scored 0.751 and OmniMamba 0.682, the opposite of the paper's small-lesion argument, though UNETR's leakage makes neither figure trustworthy.

### A5. Post-processing was tuned on the test set

The manuscript says P = 0.45 and the size filter were "selected strictly" on the 225-subject validation folds and then "confirmed" on test data. The repository contains no code or log that tunes on those folds. `tune_cc_non_leaked.py` operates only on the 25 test cases, and the earlier `research_paper_draft.md:91` says tuning was done "on the 25 test cases". Unless evidence exists outside this repository, this statement is inaccurate and must not be repeated.

### A6. The central efficiency claim is wrong

The abstract's "12.3M vs 101.9M, nearly 90% fewer parameters" does not match the models used:

- OmniMamba-UNet: **35.01M** (training logs and checkpoint)
- UNETR: **127.4M** (checkpoint and rebuilt model)
- Actual reduction: 72.5%

12.3M and 101.9M appear only in `paper.tex` and `README.md`.

### A7. The "generalisation gap" is a metric inconsistency

Cross-validation Dice excluded blank-label cases, while the test Dice of 0.6418 counted them as zero. Scored the same way, the March test result is **0.7640 against 0.7461 in cross-validation, so there is no gap**. The paper's discussion section explaining the gap should be removed.

### A8. The "library scoring artefact" is not a library bug

MONAI gives `NaN` for an empty reference and excludes it from a dataset-wide mean. It returns 0 only when `aggregate()` is called on one empty case at a time, which is what this code does. `ignore_empty=False` already scores a correct empty prediction as 1.0 (all three behaviours tested directly). Contribution 3 and the "previously undocumented" claim should be removed.

### A9. Numbers from different runs are mixed

- The March run gives CV mean 0.7461 and test 0.6418.
- The April run, which overwrote the checkpoints, gives CV mean 0.7428, peak fold 0.8236, and the 0.6855 and 0.7685 test figures.
- The paper's 0.7461 ± 0.0740 matches neither log (0.0606 and 0.0661).

## B. Other statements the code does not support (verified)

- **"Identical" training for both models.** UNETR had 100 epochs against 200, flip probability 0.1 on one axis against rich augmentation, no EMA, a different loss, and a single pass against a 40-pass ensemble.
- **Mixed-precision training and early stopping.** Neither is in `experiments/train_mamba_13.py`, the script that produced the runs.
- **Boundary loss (Kervadec et al.).** The code instead applies BCE against a thin boundary shell (`train_mamba_13.py:205-211`), which penalises the interior of lesions.
- **"Monotonic gain per axis".** No controlled ablation exists. The single-Mamba scripts also differ in epochs (100-300) and other settings.
- **"Co-registered" FLAIR.** FLAIR was only resampled using header geometry.
- **Promised HD95 results.** No HD95 table appears in the paper.
- **"100% clinical concordance".** This rests on 2 positives, and 3 ground-truth volumes were wrong.
- **Adam at a fixed LR for UNETR.** The code uses AdamW.

## C. Why it would likely be rejected even with correct numbers

1. **Novelty.** Three-axis Mamba scanning is already published. SegMamba is not cited, and the paper's own Table 1 lists UlikeMamba and SegResMamba with the same idea.
2. **Baselines.** UNETR is a weak comparator. No job log shows nnU-Net was run, and nnU-Net is the expected reference; prior work found it beat other models on ISLES 2022. No SwinUNETR, U-Mamba or SegMamba.
3. **Statistics.** 25 test cases and p = 0.285, yet the abstract says "exceeds".
4. **Efficiency claims without measurements.** No FLOPs, memory or latency. 35M parameters is roughly the size of a default 3D nnU-Net (from memory), and 40 forward passes per scan contradicts "commodity hardware".
5. **Format and journal fit.** The manuscript uses a Springer template for an Elsevier journal, and code is only "on request" at a journal about programs.
6. **Details.** "The author" is used for three authors. At least two references look wrong (Campbell, Gonzalez; from memory, each needs checking). No generative-AI use statement is present.

## D. Recommended new paper

**Working question:** On correctly labelled data, does the backbone (CNN, Transformer or state-space) actually change infarct segmentation and volume estimation for DWI, and can the volume estimate be given a reliable uncertainty range across acquisition protocols?

| Part | Design |
|---|---|
| Comparison | ResEnc nnU-Net, SwinUNETR, U-Mamba or SegMamba, and your OmniMamba layer in one shared pipeline |
| Validation | Out-of-fold predictions for all 250 subjects (247 with lesions) instead of 25; all tuning inside training folds; pre-registered hypotheses and margins |
| Endpoints | ISLES metrics (Dice, lesion-wise F1, volume and count difference) in native space, plus measured runtime and memory |
| Protocol shift | Train on thin-slice, test on thick-slice, and the reverse. Acquisition geometry is available for all cases; vendor tags are missing for about 100 |
| Modalities | DWI+ADC against DWI+ADC+FLAIR, relevant because FLAIR needs registration |
| Method contribution | Cross-conformal volume intervals with coverage reported per protocol and lesion size, and abstention near treatment-decision thresholds |
| Release | Public code, splits, the label-repair script (other users of the dataset can hit the same bug), and TRIPOD+AI and Metrics Reloaded checklists |

This paper remains publishable if OmniMamba does not win. If it ties or loses to nnU-Net, that is the finding. Conformal volume intervals are already an active topic, so the novelty is in the stroke-specific, protocol-shift and decision-threshold evaluation, and it must be positioned against that literature.

**For a journal above CMPB, external validation is probably necessary.** The realistic source is ICPSR 38464 (2,888 acute stroke MRIs with manual DWI masks). It needs a restricted-data agreement and IRB approval or exemption, which only you or your institution can obtain, so apply early. SOOP has DWI and ADC, but I could not confirm that it has lesion masks.

**Compute estimate:** about 35 training runs and 300-500 GPU-hours, based on the earlier 9 hours per fold. The `gpu` queue allows 2 running jobs per user, and your ST_CVPR job shares that limit, so expect one to three weeks.

## E. Not verified

- The reviewers' comments are not in the repository, so the rejection reasons above are inferred.
- The official CMPB Guide for Authors refused automated access. Its structured-abstract, highlights and AI-declaration rules need checking directly.
- CMPB's impact factor: Elsevier's page lists 6.4, and third-party figures are lower.
- References flagged from memory and the SegMamba tri-orientation detail need checking against the originals during the rewrite.

## Sources

- [nnU-Net Revisited](https://arxiv.org/abs/2404.09556)
- [Deep models for stroke segmentation](https://arxiv.org/html/2403.17177v2)
- [DeepISLES, Nature Communications 2025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12335569/)
- [ISLES 2022 dataset paper](https://www.nature.com/articles/s41597-022-01875-5)
- [ICPSR 38464](https://www.icpsr.umich.edu/web/ICPSR/studies/38464)
- [SOOP data descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC11297183)
- [Robust Conformal Volume Estimation](https://arxiv.org/pdf/2407.19938)
- [LBMNet](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2026.1759114/full)
- [CMPB journal page](https://shop.elsevier.com/journals/computer-methods-and-programs-in-biomedicine/0169-2607)
