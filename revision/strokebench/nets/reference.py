"""Published comparison networks, wrapped only as far as nnU-Net's interface requires.

UMambaBotNet   official U-Mamba (Bot) network, code vendored unmodified under Apache-2.0
               in strokebench/third_party/umamba.
SwinUNETRNet   MONAI SwinUNETR.
UNETRNet       MONAI UNETR with the configuration used in the first manuscript.

None of the wrappers changes the published architecture. The Swin wrapper pads the
input to a multiple of 32 voxels (a requirement of its patch merging) and crops the
output back.
"""
import copy
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from nnunetv2.utilities.network_initialization import InitWeights_He

from strokebench.nets.omnimamba import _SupervisionFlag
from strokebench.third_party.umamba.UMambaBot_3d import UMambaBot


class UMambaBotNet(UMambaBot):
    def __init__(self, *args, **kwargs):
        # The official constructor edits the per-stage block lists in place; give it copies
        # so the plans object held by the trainer is not altered.
        super().__init__(*copy.deepcopy(args), **copy.deepcopy(kwargs))

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)      # same initialisation as the official get_umamba_bot_3d_from_plans


class SwinUNETRNet(nn.Module):
    MULTIPLE = 32

    def __init__(self, input_channels: int, num_classes: int, feature_size: int = 48, use_v2: bool = False,
                 deep_supervision: bool = False):
        super().__init__()
        from monai.networks.nets import SwinUNETR
        if deep_supervision:
            raise ValueError("SwinUNETR has no auxiliary outputs; train it with a NoDS trainer")
        self.net = SwinUNETR(in_channels=input_channels, out_channels=num_classes, feature_size=feature_size,
                             spatial_dims=3, use_v2=use_v2)
        self.decoder = _SupervisionFlag()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        pad = [(-s) % self.MULTIPLE for s in size]
        if any(pad):
            x = F.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]))
        y = self.net(x)
        return y[..., :size[0], :size[1], :size[2]]


class UNETRNet(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, img_size: Sequence[int], feature_size: int = 16,
                 hidden_size: int = 768, mlp_dim: int = 3072, num_heads: int = 12, deep_supervision: bool = False):
        super().__init__()
        from monai.networks.nets import UNETR
        if deep_supervision:
            raise ValueError("UNETR has no auxiliary outputs; train it with a NoDS trainer")
        self.net = UNETR(in_channels=input_channels, out_channels=num_classes, img_size=tuple(img_size),
                         feature_size=feature_size, hidden_size=hidden_size, mlp_dim=mlp_dim, num_heads=num_heads,
                         proj_type="perceptron", norm_name="instance", res_block=True)
        self.decoder = _SupervisionFlag()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
