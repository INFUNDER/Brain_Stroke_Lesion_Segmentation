# 🧠 OmniMamba-UNet: A Linear Selective State Space Architecture for Volumetric Stroke Lesion Delineation on MRI datasets

Official repository for **OmniMamba-UNet**, a parameter-efficient, linear-complexity volumetric deep learning architecture designed for automated 3D brain stroke lesion segmentation on multi-parametric MRI (stacked DWI, ADC, and FLAIR modalities).

---

## 🎯 High-Level Objective (Layman's Terms)
The primary objective of this project is to create a faster, lighter, and more accessible artificial intelligence (AI) system that can automatically outline regions of brain tissue damaged during an acute stroke. 
* **The Clinical Need**: When a patient has a stroke, "time is brain." Doctors must rapidly measure the volume of dead brain tissue (the stroke core) to decide if emergency therapies can save the surrounding tissue.
* **The Computational Barrier**: Existing high-accuracy AI models are computationally heavy, require expensive high-end GPU hardware, and struggle to detect small, punctate areas of damage.
* **Our Solution**: OmniMamba-UNet matches or exceeds the accuracy of state-of-the-art models while reducing the memory and parameter footprint by **nearly 90%** (12.3M vs. 101.9M parameters). This allows the tool to run efficiently on standard, commodity diagnostic workstations in local clinics, democratizing access to fast stroke delineation.

---

## 🚀 Key Features & Architectural Highlights

1. **Vision-Transformer (UNETR) Baseline**:
   - A standard MONAI-based UNETR baseline serving as a comparative landmark for 3D spatial modeling.
2. **Tri-Planar Selective State Space (OmniMamba) Layer**:
   - Replaces the traditional quadratic-complexity self-attention blocks with **Parallel Tri-Planar Selective Scanning** along the principal anatomical axes ($Z$, $Y$, and $X$).
   - Linear computational and memory complexity ($\mathcal{O}(N)$) enables high-resolution 3D medical volume processing at the native voxel level.
   - Combines a local depth-wise convolutional branch to preserve fine spatial details with global scan pathways.
3. **Mimic-Scoring Metric Correction**:
   - Documents and resolves a systematic metric artefact present in popular open-source evaluation libraries where perfect true-negative predictions on lesion-free (mimic) scans were erroneously penalized with a Dice score of `0.0` instead of receiving a correct score of `1.0`.
4. **Post-Processing Parameter Stability**:
   - Employs validation-fold optimization to select inference parameters ($P=0.45, \tau=0$ voxels), and provides a post-hoc sensitivity analysis demonstrating that this configuration generalizes robustly without overfitting or data leakage.

---

## 📂 Codespace Structure

```bash
├── paper.tex                   # Main LaTeX research manuscript
├── paper.pdf                   # Compiled, finalized PDF of the paper
├── sn-jnl.cls                  # Springer Nature journal LaTeX class file
├── new_work/                   # Core Python modules
│   ├── plot_modality_comparison.py # Plots side-by-side MRI modalities & prediction
│   ├── calculate_statistics.py # Computes bootstrap CIs, Wilcoxon p-value, and metrics
│   ├── evaluate_unetr_test.py  # UNETR baseline inference evaluator
│   ├── isles22_dataset.py      # Dataset loading, SimpleITK registration, & Z-score norm
│   ├── plot_charts.py          # Generates convergence & CC threshold tuning charts
│   └── smu_net.py              # Auxiliary SMU-Net block definitions
├── figures/                    # Publication-grade figures
│   ├── bland_altman.png        # Bland-Altman limit of agreement volume plot
│   ├── cc_threshold_tuning.png # Connected-component size and threshold sensitivity plot
│   ├── learning_curves.png     # Validation Dice & loss convergence across folds
│   ├── mamba_unet_architecture.png # Complete model block diagram
│   └── modality_prediction_comparison.png # Figure 2: Modalites vs OmniMamba prediction
├── hpc_scripts/                # PBS Cluster Job Submission Scripts
│   ├── submit_mamba.sh         # Job script for training OmniMamba-UNet
│   ├── submit_unetr_eval.sh    # Job script for UNETR baseline evaluation
│   └── install_mamba.pbs       # Helper to install Mamba CUDA packages
├── evaluate.py                 # Core segmentation metric calculation
├── run_final_test.py           # Runs 5-fold ensemble & 8-flip TTA on held-out test cohort
├── tune_cc_non_leaked.py       # Post-processing grid search on validation sets
└── README.md                   # Repository documentation (This file)
```

---

## 📊 Summary of Experimental Results

### 1. Cross-Validation Results (Internal Validation)
Both models were trained under identical preprocessing and patch resolutions ($96 \times 96 \times 96$ voxels).

| Architecture | Parameters | Mean Validation Dice | Peak Validation Dice |
| :--- | :---: | :---: | :---: |
| Vision-Transformer Baseline | 101.9M | 0.7108 | 0.7171 |
| **OmniMamba-UNet** (Ours) | **12.3M** | **0.7461 $\pm$ 0.0740** | **0.8236** |
| *Improvement* | *-88%* | *+0.0353 (+3.53%)* | *+0.1065 (+10.65%)* |

### 2. Held-Out Test Cohort Performance ($N=25$)
Evaluated on strictly non-leaked test subjects using validation-optimized post-processing ($P=0.45$, $\tau=0$).

* **UNETR Baseline**: $0.6664 \pm 0.3051$ (95% Bootstrap CI: $[0.5369, 0.7773]$)
* **OmniMamba-UNet**: $0.6855 \pm 0.3349$ (95% Bootstrap CI: $[0.5540, 0.7958]$)
* **Statistical Power**: Wilcoxon signed-rank test yields $p = 0.2850$. While the improvement (+1.91 Dice points) is not statistically significant due to the small sample size ($N=25$) constrained by the retrospective challenge, the consistent positive trend paired with the **90% parameter footprint reduction** demonstrates the practical clinical advantages of OmniMamba-UNet.

---

## ⚙️ How to Run & Verify

### 1. Pre-Processing & Training
Ensure you activate your CUDA environment containing PyTorch and MONAI (e.g., `GPU-env`):
```bash
# Submit package install job on PBS cluster
qsub hpc_scripts/install_mamba.pbs

# Run the training script
python new_work/train.py
```

### 2. Generate Figures
To regenerate the publication plots showing the modalities and segmentations:
```bash
# Generate Figure 2 (Modality vs. Prediction Comparison)
python new_work/plot_modality_comparison.py

# Generate Convergence and CC Tuning Charts
python new_work/plot_charts.py
```

---
*For the complete manuscript text, math formulation, and details on clinical thrombectomy eligibility concordance, please refer directly to the compiled [paper.pdf](paper.pdf) or the LaTeX source [paper.tex](paper.tex).*
