"""Study-wide settings. Everything here is fixed before any result is looked at
(see revision/PROTOCOL.md) and applies to every model alike."""

# Equal-budget schedule of the main comparison (epochs of 250 iterations). nnU-Net's default
# of 1000 epochs is not affordable for the Transformer and full-resolution state-space models
# on the available hardware (see revision/results/benchmark*.json), so every model in the main
# comparison gets the same shorter schedule ...
NUM_EPOCHS = 250
# ... and the convolutional references are additionally trained with nnU-Net's full default
# schedule, so the cost of the shorter schedule is measured instead of assumed.
NUM_EPOCHS_LONG = 1000

# Optimiser used when a backbone's reference implementation trains with AdamW.
ADAMW_LR = 1e-4
ADAMW_WEIGHT_DECAY = 1e-5
ADAMW_WARMUP_EPOCHS = 10

PLANS = "nnUNetPlans"
RESENC_PLANS = "nnUNetResEncUNetMPlans"        # written by nnU-Net's own ResEnc-M planner, used unmodified

# model key -> (plans, configuration, trainer). All share one preprocessed dataset,
# one patch size, one batch size, one augmentation pipeline and one set of folds.
MODELS = {
    # convolutional references
    "nnunet":         (PLANS,        "3d_fullres",                "StrokeTrainer"),
    "resenc_m":       (RESENC_PLANS, "3d_fullres",                "StrokeTrainer"),
    # state-space models
    "umamba_bot":     (PLANS,        "3d_fullres_umambabot",      "StrokeTrainer"),
    "omnimamba_nn":   (PLANS,        "3d_fullres_omnimamba",      "StrokeTrainer"),
    "omnimamba_orig": (PLANS,        "3d_fullres_omnimamba_orig", "StrokeTrainerAdamWNoDS"),
    # transformers
    "swinunetr":      (PLANS,        "3d_fullres_swinunetr",      "StrokeTrainerAdamWNoDS"),
    "unetr":          (PLANS,        "3d_fullres_unetr",          "StrokeTrainerAdamWNoDS"),
}

# The same convolutional references on nnU-Net's full default schedule.
LONG_SCHEDULE_MODELS = {
    "nnunet_long":    (PLANS,        "3d_fullres",                "StrokeTrainerLong"),
    "resenc_m_long":  (RESENC_PLANS, "3d_fullres",                "StrokeTrainerLong"),
}

# Layer ablations (same host network as omnimamba_nn).
ABLATION_MODELS = {
    "omnimamba_nn_zonly":      (PLANS, "3d_fullres_omnimamba_zonly",      "StrokeTrainer"),
    "omnimamba_nn_nolocal":    (PLANS, "3d_fullres_omnimamba_nolocal",    "StrokeTrainer"),
    "omnimamba_nn_bottleneck": (PLANS, "3d_fullres_omnimamba_bottleneck", "StrokeTrainer"),
}

ALL_MODELS = {**MODELS, **LONG_SCHEDULE_MODELS, **ABLATION_MODELS}

# Optimiser-sensitivity runs (fold 0 only, reported as a supplementary table, never used for selection):
# each model is also trained with the other optimiser family.
ALTERNATE_TRAINER = {
    "StrokeTrainer": "StrokeTrainerAdamW",
    "StrokeTrainerAdamWNoDS": "StrokeTrainerNoDS",
}

# Free GPU memory (GB) a training job waits for before it starts. The cluster's GPUs are
# shared between users, so this is a scheduling aid only; it does not affect any result.
# Values are measured peak training memory plus headroom (revision/results/benchmark*.json).
GPU_MEMORY_GB = {
    "nnunet": 10, "nnunet_long": 10, "resenc_m": 12, "resenc_m_long": 12, "umamba_bot": 16,
    "unetr": 10, "swinunetr": 40, "omnimamba_nn": 32, "omnimamba_orig": 32,
    "omnimamba_nn_zonly": 24, "omnimamba_nn_nolocal": 32, "omnimamba_nn_bottleneck": 12,
}
