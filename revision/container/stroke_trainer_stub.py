"""Stand-in for the study's StrokeTrainer inside the submission container.

The checkpoints record the trainer name "StrokeTrainer". nnU-Net looks that class up to
rebuild the network, and StrokeTrainer builds its network exactly as nnUNetTrainer does
(it only changes the training schedule), so the inherited method is the right one.
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class StrokeTrainer(nnUNetTrainer):
    pass
