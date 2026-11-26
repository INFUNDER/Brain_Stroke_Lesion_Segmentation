# Study protocol and analysis plan

Version 1.0, fixed on 2026-11-15, before any full training run on the repaired labels. The only model code executed before this date was a two-epoch smoke test and a timing benchmark on synthetic inputs.
Changes after this date are appended to the "Amendments" section with a date and a reason; nothing above that section is edited.

## 1. Question

1. On correctly labelled data and under one shared, leakage-free pipeline, does the network backbone (convolutional, Transformer, state-space) change the accuracy of acute ischaemic lesion segmentation and lesion-volume measurement on diffusion MRI?
2. How much accuracy is lost when the acquisition protocol at deployment differs from training, and does any backbone reduce that loss?
3. Can the lesion-volume estimate be given an interval with a stated coverage that still holds across acquisition protocols and lesion sizes, and is it useful near treatment-decision thresholds?

The study is designed so that every outcome is reportable, including "no backbone differs from nnU-Net".

## 2. Data

- ISLES 2022 public training release, 250 subjects, DWI + ADC + FLAIR, expert masks.
- Labels are rebuilt from the original masks by thresholding at 0.5 (`revision/01_build_corrected_labels.py`). 247 subjects have a lesion; 3 have an empty mask (`0150`, `0151`, `0170`).
- Acquisition-protocol groups, defined from DWI geometry only (vendor tags are missing for about 100 subjects):
  A thin-slice (slice thickness < 3 mm), n = 196;
  B1 thick-slice, in-plane spacing >= 1.5 mm, n = 30;
  B2 thick-slice, in-plane spacing < 1.5 mm, n = 24.
- Primary input: DWI + ADC in native DWI space. FLAIR is not registered to DWI in the public release, so it is used only in the modality comparison (section 7).
- The per-subject table is `revision/data/cases.csv`; it is built without any model output.

## 3. Validation design

- Five-fold cross-validation over all 250 subjects, stratified by protocol group and lesion-volume tertile (`revision/data/splits_final.json`, seed 20261115). Each subject is predicted once, by a model that never saw it. The evaluation set is therefore all 250 subjects, not a 25-subject hold-out.
- All models use the same folds.
- The earlier 25-subject "test set" is retired: it was inspected repeatedly and used for threshold tuning.
- Nothing is tuned on validation subjects: no probability threshold, no connected-component filter, no checkpoint selection (the final checkpoint is used), no ensembling. Predictions are the arg-max of one model per fold with nnU-Net's default sliding-window and mirroring inference.

## 4. Models

All models are trained inside nnU-Net v2.8.1 with the same preprocessed data, spacing (2 mm isotropic), patch size (80 x 96 x 80), batch size (8), augmentation, loss (Dice + cross-entropy), training length and inference. Only the network, and where stated the optimiser, differ. Definitions are in `revision/strokebench/config.py`.

| Key | Network | Optimiser |
|---|---|---|
| `nnunet` (reference) | nnU-Net default U-Net, planner output unmodified | nnU-Net default (SGD) |
| `resenc_m` | nnU-Net ResEnc-M preset, planner output unmodified | SGD |
| `umamba_bot` | U-Mamba Bot, official network code | SGD, as in its official trainer |
| `omnimamba_nn` | nnU-Net U-Net + OmniMamba layers on all skips and the bottleneck | SGD |
| `omnimamba_orig` | network of the first manuscript, ported unchanged | AdamW, as in the first manuscript |
| `swinunetr` | MONAI SwinUNETR, feature size 48 | AdamW |
| `unetr` | MONAI UNETR, configuration of the first manuscript | AdamW |

Training length. Cost was measured with synthetic inputs before any training (`revision/results/benchmark.json`, `benchmark_omni.json`; shared GPU, so upper bounds):

| Model | Parameters (M) | Peak training memory (GB) | GPU-hours per fold at 1000 epochs |
|---|---|---|---|
| `nnunet` | 16.5 | 5.3 | 11.6 |
| `resenc_m` | 62.8 | 6.6 | 11.9 |
| `umamba_bot` | 22.6 | 11.1 | 22.8 |
| `unetr` | 124.2 | 5.7 | 12.5 |
| `swinunetr` | 62.2 | 33.7 | 70.5 |
| `omnimamba_nn` | 22.7 | 25.1 | 113.7 |
| `omnimamba_orig` | 35.0 | 26.0 | 121.7 |

- nnU-Net's default of 1000 epochs for all seven models and five folds would need about 1,800 GPU-hours for the main comparison alone, which the available cluster cannot provide. Therefore every model in the main comparison is trained for the same 250 epochs of 250 iterations (62,500 iterations, batch 8), with the learning-rate schedule annealed over those 250 epochs.
- To measure what the shorter schedule costs, the two convolutional references are also trained with the full 1000-epoch default (`nnunet_long`, `resenc_m_long`, all five folds). These are reported next to the main comparison and are the numbers to compare with other nnU-Net results in the literature.
- The OmniMamba layer uses gradient checkpointing so that nnU-Net's batch size fits in memory. This changes memory and time only; outputs and gradients are identical (unit-tested).
- SGD recipe: nnU-Net default (lr 0.01, Nesterov momentum 0.99, weight decay 3e-5, poly decay, deep supervision).
- AdamW recipe: lr 1e-4, weight decay 1e-5, 10-epoch linear warm-up, then the same poly decay. Networks with a single output are trained without deep supervision.
- The optimiser of each model is fixed here from its reference implementation. It is not selected on data.
- Optimiser sensitivity (supplementary, fold 0 only): every model is also trained with the other optimiser family. These runs are reported and are not used to choose anything.
- Mixed precision is used for all models; state-space layers run in float32 internally.
- `torch.compile` is off for all models.
- SegMamba will be added as an eighth model only if its official network can be run unmodified in this environment; its absence would be stated as a limitation.

## 5. Outcomes

All outcomes are computed per subject in native image space (`revision/strokebench/metrics.py`, unit-tested).

- Primary: Dice coefficient on the 247 lesion-positive subjects.
- Secondary: lesion-wise F1, absolute volume difference (mL), absolute lesion-count difference (the other three ISLES'22 ranking metrics), signed volume error, 95th-percentile Hausdorff distance, number of subjects whose lesion is missed entirely.
- Empty-reference subjects (n = 3) are reported descriptively and separately. They never enter a mean with lesion-positive subjects.
- Cost: parameters, training time per step, peak training memory, inference time and memory per volume, measured on an otherwise idle GPU.

## 6. Statistical analysis of the backbone comparison

- Unit of analysis: the subject. Comparisons are paired by subject.
- For each model against the `nnunet` reference: mean paired difference in Dice with a 95% bootstrap confidence interval (10,000 resamples of subjects, stratified by fold).
- Equivalence: a model is declared equivalent to the reference if the 90% confidence interval of the paired mean Dice difference lies within +/-0.02 (two one-sided tests at alpha = 0.05). The margin is fixed here; its justification against inter-rater variability will be documented from the literature in the manuscript.
- Difference: two-sided Wilcoxon signed-rank test on paired Dice, Holm-corrected over the six comparisons with the reference.
- Run-to-run noise floor: the reference model is trained a second time on all five folds. The distribution of per-subject differences between the two replicates is reported next to every between-model difference, so that a between-model difference smaller than replicate noise is not over-read.
- Pre-specified strata: protocol group (A, B1, B2) and lesion volume (< 1 mL, 1-10 mL, 10-70 mL, > 70 mL). Strata are descriptive, with confidence intervals and no significance claims.
- All other comparisons are labelled exploratory.

## 7. Protocol-shift and modality experiments

Protocol shift (models `nnunet`, `resenc_m`, `umamba_bot`, `omnimamba_nn`, `swinunetr`):
- fold 5: train on thin-slice subjects (n = 156), test on thick-slice subjects (B1 + B2, n = 54) and on 40 held-out thin-slice subjects. The held-out thin-slice subjects give same-protocol performance for the same model and serve as the source-protocol calibration set in section 8;
- fold 6: train on thick-slice (n = 54), test on thin-slice;
- fold 7: train on a random thin-slice subset of the same size (n = 54), test on all remaining subjects. This separates the effect of protocol from the effect of training-set size.
- Cost of shift = for the same thick-slice subjects, Dice from fold 5 minus Dice from cross-validation (where thick-slice subjects were in training). Reported per model with bootstrap intervals; the between-model contrast is a difference of paired differences.

Modality: `nnunet` trained on DWI + ADC + FLAIR (FLAIR resampled to the DWI grid by header geometry), same folds, paired comparison with DWI + ADC.

Layer ablation (same host network as `omnimamba_nn`): one scan axis instead of three; no local convolution branch; layers on the bottleneck only. Five folds each if compute allows, otherwise folds 0-2, decided by compute budget before any ablation result is seen.

## 8. Calibrated lesion-volume intervals

- Inputs: out-of-fold probability maps and reference volumes.
- Calibration: within each fold, the interval for a subject is calibrated on the other validation subjects of the same fold. All of them were unseen by that fold's model, so the standard split-conformal guarantee applies with the finite-sample quantile.
- Score functions compared: absolute volume residual; relative (log-ratio) residual; threshold-nested intervals, where the interval is the range of volumes obtained by sweeping the probability threshold symmetrically around 0.5 and the calibrated quantity is the sweep half-width.
- Nominal coverage levels: 80%, 90%, 95%.
- Primary outcome: empirical coverage at the 90% level, overall and within each protocol group and lesion-volume stratum, with exact binomial intervals. Secondary: median interval width relative to the reference volume.
- Under protocol shift (fold 5): intervals calibrated on the 40 held-out thin-slice subjects are evaluated on the 54 thick-slice subjects, all unseen by that model. Recovery is tested by calibrating instead on k thick-slice subjects (k = 10, 20, 30; repeated random draws, evaluated on the remaining thick-slice subjects).
- Group-wise (Mondrian) calibration by protocol group is evaluated on the cross-validation predictions by pooling out-of-fold scores across folds, because single folds hold too few thick-slice subjects. Pooled scores come from five models, so this analysis has the weaker cross-conformal guarantee and is labelled as such.
- Decision use: for volume thresholds used in thrombectomy trials (values to be taken from the trial publications, fixed before analysis), each subject is classified as below, above, or undetermined when the interval contains the threshold. Reported: error rate among determined subjects and the fraction undetermined, against the point-estimate error rate.
- The analysis code is written and tested on synthetic data before it is run on model outputs.

## 9. Run plan and priority

| Tier | Runs | Purpose | GPU-hours (upper bound) |
|---|---|---|---|
| 1 | 7 models x 5 folds = 35 at 250 epochs | main comparison | about 455 |
| 1 | 2 convolutional references x 5 folds at 1000 epochs | cost of the shorter schedule | about 120 |
| 2 | 5 replicate runs of the reference; 5 models x 3 shift folds = 15 | noise floor, protocol shift | about 190 |
| 3 | 5 (modality), 7 (optimiser sensitivity), up to 15 (layer ablation) | supporting analyses | about 310 |

Tier 3 may be reduced for compute reasons only, and the reduction is decided before its results exist.

## 10. Reporting and release

- Reporting follows TRIPOD+AI and CLAIM checklists; metric choice follows Metrics Reloaded.
- Released with the paper: all code in `revision/`, the environment lock file, the folds, the per-subject result tables, and the label-repair script.
- External validation on an independent cohort (ICPSR 38464) will be added if data access is granted; the frozen cross-validation models will be applied without any adaptation.

## Amendments

**2026-11-22. Section 8 settings fixed before any interval was computed on model outputs.** Segmentation results of five models were known at this point; no volume-interval result was.

- Primary model for section 8: the reference `nnunet`. The other models with complete cross-validation are reported as secondary.
- Volume thresholds for the decision analysis, taken from thrombectomy trial selection criteria for infarct core: 21, 31 and 51 mL (DAWN, Nogueira et al., N Engl J Med 2018; the three age/NIHSS-dependent core limits) and 70 mL (DEFUSE 3, Albers et al., N Engl J Med 2018). The DWI lesion in this dataset is used as a stand-in for infarct core; this is a limitation, and the analysis evaluates measurement agreement only, not treatment eligibility.
- The decision analysis uses the 90% intervals.
- All 250 subjects enter the coverage analysis, including the 3 with an empty reference (a volume of 0 mL is a valid target).
- Probability maps are nnU-Net's exported softmax in native image space (`--npz`); lesion probability is summarised per subject as a 1000-bin histogram, so threshold-swept volumes are exact to 0.001 in probability.

**2026-11-22 (later the same day). Baselines for the decision analysis, added after the interval decision results had been seen.** These comparisons are therefore exploratory, and the paper will say so.

- Flag-rate-matched baselines: for each threshold, flag the same number of subjects as the 90% log-ratio intervals did, choosing those whose point estimate is closest to the threshold, (a) by ratio |log(estimate / threshold)| and (b) by absolute difference in mL.
- Uncalibrated fixed bands: flag subjects whose estimate is within ±10%, ±20% or ±50% of the threshold.
- Reported for each rule: number flagged, point-estimate errors left among the subjects not flagged, and errors captured among the flagged.
- Note recorded before running: the log-ratio interval contains the threshold exactly when the estimate lies within a fold-specific ratio of the threshold, so baseline (a) is expected to match it closely. What the conformal step adds is a band width chosen by calibration with a stated coverage, not a better ranking of subjects.

**2026-11-22 (third entry). Adaptive (per-patient) volume intervals, specified before any adaptive result was computed.** Motivated by the baseline result above, so it is labelled as a secondary, post hoc method in the paper.

- Per-patient spread from the model's own probability map: s = log((V(p>=0.1) + e) / (V(p>=0.9) + e)), with V(p>=t) the volume of voxels with lesion probability at least t, and e = 0.1 mL as in the log-ratio score.
- Normalised score: |log((V_ref + e) / (V_hat + e))| / (s + d), with d = 0.1 fixed here. Interval: V_hat scaled by exp(-/+ q (s + d)), with q calibrated exactly as before (leave-one-out within each cross-validation fold).
- Sensitivity variant (reported, not used to judge): spread from the 0.25 and 0.75 thresholds.
- Diagnostic: Spearman correlation between s and the absolute log-ratio error, per model.
- Primary success criterion, fixed now: pooled over the five models and four thresholds, at the same number of flagged patients as the adaptive 90% intervals, the adaptive rule must catch more point-estimate errors than the better of the two matched baselines (ratio-ranked and mL-ranked). Anything else is reported as a failure of the adaptive method.
- Coverage within the protocol and lesion-volume strata and median relative width are reported alongside, using the same tables as before.

**Outcome of the adaptive-interval test, 2026-11-22 (recorded after running, nothing above was changed).** The pre-specified criterion was met narrowly: pooled adaptive 90% intervals caught 96 point-estimate errors against 92 for the better matched baseline (mL-ranked) and 85 for the ratio-ranked one, at 917 flagged. A patient-level bootstrap (2,000 resamples, added after the fact) gives a difference of +4 errors with a 95% interval of -3 to +13, so the advantage is not established. Adaptive intervals are narrower (nnU-Net median relative width 1.11 vs 1.29) but under-cover lesions below 1 mL (0.72-0.87 at the 90% level).

**2026-11-26. Rule for diverged training runs, added after the first divergence was seen.** `omnimamba_orig` fold 0 (AdamW, 250 epochs) reached a non-finite training loss at epoch 23 and never recovered; its final checkpoint predicts no lesion in any subject (Dice 0.000). It is the only run of the 60 trained so far with a non-finite loss. Section 3 forbids checkpoint selection, so the healthy checkpoints from before epoch 23 are not used.

- A run is "diverged" when its mean training loss for an epoch is not finite. From now on the trainer stops such a run at that epoch, before any checkpoint is written, and records `diverged.json` in the run folder; a diverged run is never resumed.
- A diverged run is restarted once, from scratch, with the same recipe, data and fold (a fresh, independent run: nnU-Net draws new random initialisation and data order). The diverged run is kept, renamed `fold_<k>_diverged_run1`, and is not scored.
- The restart counts as the result for that fold. If the restart also diverges, the fold is scored as it stands (no prediction) and the model's mean is reported both with and without that fold.
- The rule applies to every model and fold in the study. Every divergence, including this one, is reported in the paper as a training-stability result, with the model, fold and epoch.

**2026-11-26 (second entry). Mechanism, explainability and external-test experiments (E1-E6), specified before any of them was run.** Known at this point: the cross-validation, protocol-shift and volume-interval results of six models (sections 6-8 and the amendments above). Not known: any result below. Two design facts were checked first and are not results: (i) the raw ISLES 2022 ADC maps are stored in three unit scales (median brain ADC about 0.001, 1 or 1000; 106, 68 and 76 subjects), and (ii) nnU-Net's 5-stage U-Net has a theoretical receptive field of 143 voxels (286 mm) without normalisation layers, and its instance normalisation adds global statistics. The paper's aim changes to explaining *why* backbones perform alike; the backbone comparison stays as pre-specified.

- **ADC harmonisation.** Before any analysis that uses absolute ADC values, each subject's ADC is converted to 10^-6 mm^2/s from its median brain value m (m < 0.01: x 10^6; 0.01 <= m < 10: x 10^3; otherwise unchanged). Network training is unaffected (per-subject z-scoring).

- **E1. Context ladder.** Networks of increasing receptive field, all trained in the same nnU-Net recipe (SGD, 250 epochs, folds 0-4) with batch normalisation, so that no layer sees beyond its receptive field at inference: R0 voxel-wise (1x1x1 kernels, 1 voxel = 2 mm); U-Nets with 2, 3, 4 and 5 stages (15, 35, 71 and 143 voxels; 30, 70, 142 and 286 mm). Primary: mean Dice of each rung on the 247 lesion-positive subjects, paired against the 5-stage rung; the smallest rung that is equivalent (90% CI within +/-0.02) is reported as the context the task needs. Secondary: the same by lesion-volume stratum; the 5-stage batch-norm rung against the instance-norm reference (effect of normalisation). Hypothesis H1: a rung with a receptive field of 70 mm or less is equivalent to the full network.

- **Physics-rule baselines (no training).** Applied in native space against the corrected labels, thresholds fixed now and not tuned: R-ADC, ADC < 620 x 10^-6 mm^2/s inside the brain (ADC > 0); R-ADC+DWI, the same and DWI above the brain mean + 2 SD. Reported: Dice, lesion F1 and volume error, overall and by stratum.

- **E2. Physics counterfactuals (explainability).** Each lesion-positive subject is re-predicted by its own out-of-fold model of every backbone, with the cross-validation inference settings, after one edit of the input: CF-A, ADC inside the reference lesion set to the subject's median ADC of the brain outside the lesion (bright DWI, normal ADC, as in T2 shine-through or pseudo-normalisation); CF-B, DWI inside the lesion set to the median DWI of the brain outside the lesion (low ADC, normal DWI); CF-C, the whole ADC channel set to its brain median (DWI only); CF-D, the whole DWI channel set to its brain median (ADC only). Primary outcome: for CF-A, the retained fraction = predicted lesion volume inside the reference lesion after the edit divided by the same before it, per subject (subjects with a non-empty prediction inside the lesion before the edit), summarised by median and bootstrap 95% CI per backbone. A physics-consistent model should retain less than half (H2). Secondary: retained fraction for CF-B; Dice for CF-C and CF-D; differences between backbones (exploratory). Pixel attribution (integrated gradients, share of attribution on each channel inside the lesion) is exploratory and is reported only together with a model-randomisation sanity check.

- **E3. Effective receptive field.** For the fold-0 model of every backbone, the gradient magnitude of the lesion logit at the centre voxel with respect to the input, averaged over 20 validation patches centred on reference-lesion voxels (fixed seed); summarised as the radii (mm) containing 50% and 90% of the gradient mass. Descriptive; compared with E1.

- **E4. Data effects.** (a) nnU-Net retrained on folds 0-4 with the corrupted labels (integer cast of the original masks, reproducing the 47 blank labels), scored against the corrected labels; primary: paired Dice difference against the corrected-label nnU-Net on the lesion-positive subjects, overall and in the thick-slice groups. (b) Learning curve: nnU-Net trained on nested 25% and 50% subsets of each fold's training subjects (stratified by protocol, seed 20261126). (c) Variance decomposition (no training): two-way decomposition of per-subject Dice over subjects x backbones (all backbones with complete cross-validation), reporting the share of variance due to subject, backbone and residual, and the share of subject variance explained by protocol group and log reference volume.

- **E5. External tests, frozen models only.** (a) ISLES 2022 hidden test set (150 subjects, three centres, one unseen in training; one submission per month): first submission, nnU-Net 5-fold ensemble with nnU-Net's default inference on DWI+ADC; second, the OmniMamba (nnU-Net) 5-fold ensemble. Metrics are those returned by the challenge. Nothing is changed after a submission. (b) ICPSR 38464, if access is granted: 5-fold ensembles of every backbone; metrics against the provided masks; Spearman correlation of predicted and of reference lesion volume with admission NIHSS.

- **E6. Failure detection (no training).** On the out-of-fold predictions, two label-free scores per subject: mean pairwise Dice between the backbones' predictions, and the probability spread s of the adaptive-interval amendment. Primary: AUROC (bootstrap 95% CI) for detecting subjects with Dice < 0.5.

- **Deprioritised** (not run unless GPU time remains): SegMamba, optimiser-sensitivity runs, OmniMamba layer ablations, and FLAIR for models other than nnU-Net.

- **Order:** ADC harmonisation, E4c, physics rules and E6 first (no training); then E2 and E3 (prediction only); then E1 and E4a; then E4b. E5a starts as soon as the container is ready.
