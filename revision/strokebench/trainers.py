"""nnU-Net trainers used in the study.

StrokeTrainer            nnU-Net's default recipe, unchanged (SGD + Nesterov, lr 1e-2, poly decay,
                         Dice + cross-entropy with deep supervision, default augmentation).
StrokeTrainerNoDS        the same without deep supervision, for networks with a single output.
StrokeTrainerAdamW       the same with AdamW (lr 1e-4, weight decay 1e-5, 10-epoch linear warm-up,
                         then nnU-Net's poly decay), for backbones whose reference implementations
                         train with AdamW.
StrokeTrainerAdamWNoDS   AdamW without deep supervision.

The "...Debug" classes run 2 tiny epochs and write to their own result folders;
they exist only to smoke-test the pipeline.

This file is linked into nnU-Net's trainer folder by revision/env/setup_env.sh so
that the standard nnUNetv2_train / nnUNetv2_predict commands can find the classes.
"""
import torch
from batchgenerators.utilities.file_and_folder_operations import isfile, join, load_json, save_json

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from strokebench import config


class WarmupPolyLRScheduler(PolyLRScheduler):
    """nnU-Net's poly schedule multiplied by a linear ramp over the first `warmup_steps` epochs."""

    def __init__(self, optimizer, initial_lr: float, max_steps: int, warmup_steps: int, exponent: float = 0.9):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, initial_lr, max_steps, exponent)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        ramp = min(1.0, (current_step + 1) / self.warmup_steps) if self.warmup_steps > 0 else 1.0
        new_lr = self.initial_lr * ramp * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


class StrokeTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = config.NUM_EPOCHS
        # Checkpoint every 10 epochs instead of 50: the GPUs are shared and a job can be
        # killed by a neighbour's memory use. This changes restart cost only, not results.
        self.save_every = 10

    def on_train_start(self):
        """Record the schedule once, and refuse to resume a run under a different one."""
        super().on_train_start()
        marker = join(self.output_folder, "schedule.json")
        schedule = {"num_epochs": int(self.num_epochs), "iterations_per_epoch": int(self.num_iterations_per_epoch),
                    "optimizer": type(self.optimizer).__name__, "initial_lr": float(self.initial_lr),
                    "weight_decay": float(self.weight_decay), "deep_supervision": bool(self.enable_deep_supervision),
                    "batch_size": int(self.batch_size)}
        if isfile(marker):
            previous = load_json(marker)
            if previous != schedule:
                raise RuntimeError(f"schedule changed since this run started: {previous} -> {schedule}")
        else:
            save_json(schedule, marker)


class StrokeTrainerNoDS(StrokeTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False


class StrokeTrainerAdamW(StrokeTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = config.ADAMW_LR
        self.weight_decay = config.ADAMW_WEIGHT_DECAY
        self.warmup_epochs = config.ADAMW_WARMUP_EPOCHS

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)
        lr_scheduler = WarmupPolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, self.warmup_epochs)
        return optimizer, lr_scheduler


class StrokeTrainerAdamWNoDS(StrokeTrainerAdamW):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False


class StrokeTrainerLong(StrokeTrainer):
    """nnU-Net's full default schedule (1000 epochs). Used for the convolutional references
    only, to show what the equal-budget schedule of the main comparison costs in accuracy."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = config.NUM_EPOCHS_LONG


class StrokeTrainerReplicate2(StrokeTrainer):
    """Identical to StrokeTrainer. A second, independently seeded run of the same recipe;
    the separate class name only gives it a separate result folder (run-to-run noise floor)."""


def _shorten(trainer) -> None:
    trainer.num_epochs = 2
    trainer.num_iterations_per_epoch = 6
    trainer.num_val_iterations_per_epoch = 2
    if hasattr(trainer, "warmup_epochs"):
        trainer.warmup_epochs = 1


class StrokeTrainerDebug(StrokeTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        _shorten(self)


class StrokeTrainerNoDSDebug(StrokeTrainerNoDS):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        _shorten(self)


class StrokeTrainerAdamWDebug(StrokeTrainerAdamW):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        _shorten(self)


class StrokeTrainerAdamWNoDSDebug(StrokeTrainerAdamWNoDS):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        _shorten(self)
