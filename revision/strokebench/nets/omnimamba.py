"""OmniMamba layer and the two networks that use it.

OmniMambaLayer        the tri-axis selective-scan layer of the first manuscript
                      (one shared LayerNorm, one independent Mamba per scan order,
                      a depth-wise 3x3x3 branch and an identity path, all summed).
OmniMambaUNetOriginal the first manuscript's network, ported without architectural change.
PlainConvUNetOmniMamba nnU-Net's own U-Net with OmniMamba layers on every skip and on the
                      bottleneck. Compared with plain nnU-Net it differs only by those
                      layers, which is the controlled test of what the layer adds.

The state-space operator is always run in float32. Its recurrence overflows in
half precision (the NaNs seen in the first study), so the layer leaves autocast.
"""
from typing import Callable, Optional, Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint_fn

from dynamic_network_architectures.architectures.unet import PlainConvUNet

# For each scan: the permutation applied to (B, C, d0, d1, d2) before row-major
# flattening, and the permutation that undoes it. In the flattened sequence the
# last permuted axis varies fastest.
SCAN_ORDERS = {
    "z": ((0, 1, 2, 3, 4), (0, 1, 2, 3, 4)),
    "y": ((0, 1, 3, 4, 2), (0, 1, 4, 2, 3)),
    "x": ((0, 1, 4, 2, 3), (0, 1, 3, 4, 2)),
}


def _mamba(dim: int, d_state: int, d_conv: int, expand: int) -> nn.Module:
    from mamba_ssm import Mamba  # imported lazily: the operator needs CUDA, the layer logic does not
    return Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)


class OmniMambaLayer(nn.Module):
    def __init__(self, dim: int, d_state: int = 32, d_conv: int = 4, expand: int = 2,
                 axes: Sequence[str] = ("z", "y", "x"), local_branch: bool = True,
                 sequence_module: Optional[Callable[[int], nn.Module]] = None, grad_checkpoint: bool = True):
        super().__init__()
        unknown = set(axes) - set(SCAN_ORDERS)
        if unknown:
            raise ValueError(f"unknown scan axes {unknown}")
        make = sequence_module or (lambda d: _mamba(d, d_state, d_conv, expand))
        self.axes = tuple(axes)
        self.norm = nn.LayerNorm(dim)
        self.scans = nn.ModuleDict({a: make(dim) for a in self.axes})
        self.local_conv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim) if local_branch else None
        # Gradient checkpointing recomputes each scan during the backward pass instead of
        # storing its activations. Outputs and gradients are unchanged; only memory and time differ.
        self.grad_checkpoint = grad_checkpoint

    def _scan(self, x: torch.Tensor, axis: str) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            perm, inverse = SCAN_ORDERS[axis]
            xp = x.permute(*perm)
            shape = xp.shape
            seq = xp.reshape(shape[0], shape[1], -1).transpose(1, 2)              # (B, N, C)
            y = self.scans[axis](self.norm(seq))
            return y.transpose(1, 2).reshape(shape).permute(*inverse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            out = x
            if self.local_conv is not None:
                out = out + self.local_conv(x)
            save_memory = self.grad_checkpoint and self.training and x.requires_grad
            for axis in self.axes:
                if save_memory:
                    out = out + grad_checkpoint_fn(self._scan, x, axis, use_reentrant=False)
                else:
                    out = out + self._scan(x, axis)
            return out


class _SupervisionFlag(nn.Module):
    """nnU-Net toggles `network.decoder.deep_supervision`; networks without
    auxiliary outputs expose this inert attribute so the same trainer code runs."""
    deep_supervision = False


class OmniMambaUNetOriginal(nn.Module):
    """Network of the first manuscript: 4-stage conv encoder (32-256), stride-2
    bottleneck (512) with two OmniMamba layers, OmniMamba on all four skips,
    transposed-conv decoder. No deep supervision, as in the original."""

    def __init__(self, input_channels: int, num_classes: int, features: Sequence[int] = (32, 64, 128, 256, 512),
                 d_state: int = 32, d_conv: int = 4, expand: int = 2, axes: Sequence[str] = ("z", "y", "x"),
                 deep_supervision: bool = False):
        super().__init__()
        from monai.networks.blocks import UnetBasicBlock, UnetUpBlock
        if deep_supervision:
            raise ValueError("OmniMambaUNetOriginal has no auxiliary outputs; train it with a NoDS trainer")
        f = list(features)
        layer = lambda dim: OmniMambaLayer(dim, d_state=d_state, d_conv=d_conv, expand=expand, axes=axes)
        self.enc1 = UnetBasicBlock(3, input_channels, f[0], kernel_size=3, stride=1, norm_name="instance")
        self.enc2 = UnetBasicBlock(3, f[0], f[1], kernel_size=3, stride=2, norm_name="instance")
        self.enc3 = UnetBasicBlock(3, f[1], f[2], kernel_size=3, stride=2, norm_name="instance")
        self.enc4 = UnetBasicBlock(3, f[2], f[3], kernel_size=3, stride=2, norm_name="instance")
        self.skip1_mamba, self.skip2_mamba = layer(f[0]), layer(f[1])
        self.skip3_mamba, self.skip4_mamba = layer(f[2]), layer(f[3])
        self.bottleneck_conv = UnetBasicBlock(3, f[3], f[4], kernel_size=3, stride=2, norm_name="instance")
        self.mamba_block1, self.mamba_block2 = layer(f[4]), layer(f[4])
        self.dec4 = UnetUpBlock(3, f[4], f[3], kernel_size=3, stride=1, upsample_kernel_size=2, norm_name="instance")
        self.dec3 = UnetUpBlock(3, f[3], f[2], kernel_size=3, stride=1, upsample_kernel_size=2, norm_name="instance")
        self.dec2 = UnetUpBlock(3, f[2], f[1], kernel_size=3, stride=1, upsample_kernel_size=2, norm_name="instance")
        self.dec1 = UnetUpBlock(3, f[1], f[0], kernel_size=3, stride=1, upsample_kernel_size=2, norm_name="instance")
        self.final = nn.Conv3d(f[0], num_classes, kernel_size=1)
        self.decoder = _SupervisionFlag()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.mamba_block2(self.mamba_block1(self.bottleneck_conv(e4)))
        d4 = self.dec4(b, self.skip4_mamba(e4))
        d3 = self.dec3(d4, self.skip3_mamba(e3))
        d2 = self.dec2(d3, self.skip2_mamba(e2))
        d1 = self.dec1(d2, self.skip1_mamba(e1))
        return self.final(d1)


class PlainConvUNetOmniMamba(PlainConvUNet):
    """nnU-Net's PlainConvUNet with an OmniMamba layer on each encoder output.

    The deepest output (the bottleneck) receives `bottleneck_depth` layers, as in
    the original design. `omni_stages` restricts the layers to some stages and
    `omni_axes` selects the scan orders; both exist for the ablation study.
    """

    def __init__(self, *args, omni_axes: Sequence[str] = ("z", "y", "x"), omni_stages: Optional[Sequence[int]] = None,
                 omni_local_branch: bool = True, omni_d_state: int = 32, bottleneck_depth: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        channels = list(self.encoder.output_channels)
        n = len(channels)
        stages = set(range(n)) if omni_stages is None else set(omni_stages)
        blocks = []
        for s, c in enumerate(channels):
            if s not in stages:
                blocks.append(nn.Identity())
                continue
            depth = bottleneck_depth if s == n - 1 else 1
            blocks.append(nn.Sequential(*[
                OmniMambaLayer(c, d_state=omni_d_state, axes=omni_axes, local_branch=omni_local_branch) for _ in range(depth)
            ]))
        self.context = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor):
        skips = self.encoder(x)
        skips = [block(s) for block, s in zip(self.context, skips)]
        return self.decoder(skips)
