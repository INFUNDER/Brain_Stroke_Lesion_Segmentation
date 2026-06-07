# 🧠 Brain Stroke Lesion Segmentation (ISLES 2022)

This repository contains deep learning pipelines for automated 3D brain stroke lesion segmentation on the **ISLES 2022 dataset** (stacked DWI, ADC, and FLAIR MRI modalities). The project explores and compares standard **Vision Transformers (UNETR)** with **Selective State Space Models (Mamba)**, culminating in our custom **Stroke Mamba UNet (SMU-Net)** architecture.

---

## 🚀 Key Features & Architectural Highlights

1. **UNETR (Transformer) Baseline**:
   - Standard MONAI-based UNETR baseline serving as a comparative landmark for 3D spatial modeling.
2. **3D Mamba-UNet with OmniMamba**:
   - Replaces traditional transformer blocks with **Tri-Planar Selective Scanning** along the X, Y, and Z axes.
   - Linear computational complexity ($O(N)$) enables high-resolution 3D medical volume processing.
3. **Stroke Mamba UNet (SMU-Net)**:
   - **Physics-Guided Gate Channel**: Explicitly models tissue restriction using DWI × ADC correlation gating.
   - **Dual-Kernel Parallel Blocks**: Captures both large territorial infarcts and tiny punctiform embolic lesions.
   - **ASPP Multi-Scale Block**: Adapts to diverse lesion sizes via Atrous Spatial Pyramid Pooling.
   - **BiMamba3D Bottleneck**: Explores bidirectional scan paths.
4. **HPC Support**:
   - Ready-to-submit scripts for PBS job schedulers (targeting NVIDIA GPUs/H100).
5. **Post-Processing & Evaluation Tuning**:
   - Custom metric evaluation script correcting the MONAI foreground DiceMetric flaw (which penalizes correct negative predictions on healthy mimic scans).
   - Grid-search script for tuning probability threshold and Connected Component (CC) size filtering.

---

## 📂 Codespace Structure

```bash
├── new_work/                   # Active Development
│   ├── smu_net.py              # Stroke Mamba UNet (SMU-Net) architecture
│   ├── train.py                # Training runner for SMU-Net
│   └── isles22_dataset.py      # Preprocessing,SimpleITK registration, and dataset loader
├── experiments/                # Legacy Iterations & Experiments
│   ├── train.py                # Standard UNETR baseline training script
│   ├── train_mamba_1.py to 16  # Iterative Mamba experiments (OmniMamba variations)
│   ├── TRAIN_MAMBA_v5.py       # Stable intermediate training scripts
│   └── Train_NNUNET.py         # NNUNET comparative baseline script
├── hpc_scripts/                # HPC Cluster Job Submission Scripts
│   ├── submit_mamba.sh         # Shell script to submit active Mamba training
│   ├── submit_tune_cc.sh       # Shell script to submit threshold grid search
│   ├── run_mamba.pbs           # PBS queue script for Mamba U-Net
│   └── install_mamba.pbs       # PBS package installation helper
├── evaluate.py                 # Core evaluation script
├── run_final_test.py           # Runs 5-fold ensemble & 8-flip TTA on held-out test cases
├── tune_cc_non_leaked.py       # Post-processing grid-search (Probability and CC size)
├── isles22_splits.json         # 5-fold cross validation split metadata
├── progress_report.md          # Comprehensive progress report mapping job IDs to Dice scores
├── research_paper_draft.md     # Detailed LaTeX/Markdown manuscript draft of the study
└── README.md                   # Repository documentation (This file)
```

---

## ⚙️ How to Run

### 1. Requirements & Setup
You can install dependencies using the provided cluster installation script or via pip:
```bash
# Submit package install job on PBS HPC cluster
qsub hpc_scripts/install_mamba.pbs
```

### 2. Training the SMU-Net Model
To train the main Stroke Mamba UNet model on the HPC cluster:
```bash
# Submit training job
qsub hpc_scripts/submit_mamba.sh
```

### 3. Evaluating and Running Inference
To run the 5-fold ensemble with 8-flip Test-Time Augmentation (TTA) on held-out test cases:
```bash
python run_final_test.py
```

### 4. Post-Processing & Threshold Tuning
To perform grid search optimization over classification thresholds and connected component minimum voxel size:
```bash
python tune_cc_non_leaked.py
```

---

## 📊 Summary of Experimental Results

| Model / Configuration | Validation Dice (Mean) | Peak Validation Dice | Test Dice (25 Held-Out Cases) |
| :--- | :---: | :---: | :---: |
| **UNETR Baseline** | 0.7108 | 0.7171 | - |
| **3D Mamba-UNet** (Ours) | **0.7461** | **0.8236** | **0.6418** (Raw)<br>**0.6855** (Optimized P=0.45) |

*For full experimental timelines, hyperparameter settings, and architectural evolution, please refer to the detailed [progress_report.md](progress_report.md) and [research_paper_draft.md](research_paper_draft.md).*
