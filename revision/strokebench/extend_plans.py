#!/usr/bin/env python3
"""Add the comparison backbones to nnU-Net's plans as extra configurations.

Each new configuration inherits everything from nnU-Net's own `3d_fullres`
(preprocessed data, spacing, patch size, batch size, normalisation) and replaces
only the `architecture` entry. nnU-Net's own plan is left untouched; the file as
written by the planner is kept next to it as nnUNetPlans.planner_original.json.

Also installs the study folds (revision/data/splits_final.json) for the dataset.
"""
import argparse
import copy
import json
import shutil

from strokebench import paths

BASE = "3d_fullres"


def extra_configurations(base: dict) -> dict:
    arch = base["architecture"]
    patch = list(base["patch_size"])

    def conv_like(class_name: str, **extra) -> dict:
        a = copy.deepcopy(arch)
        a["network_class_name"] = class_name
        a["arch_kwargs"].update(extra)
        return {"inherits_from": BASE, "architecture": a}

    def standalone(class_name: str, **kwargs) -> dict:
        return {"inherits_from": BASE,
                "architecture": {"network_class_name": class_name, "arch_kwargs": kwargs, "_kw_requires_import": []}}

    return {
        # official U-Mamba (Bot): takes the same arguments as nnU-Net's U-Net
        "3d_fullres_umambabot": conv_like("strokebench.nets.reference.UMambaBotNet"),
        # nnU-Net's U-Net + OmniMamba layers on every skip and the bottleneck
        "3d_fullres_omnimamba": conv_like("strokebench.nets.omnimamba.PlainConvUNetOmniMamba"),
        # ablations of the layer (same host network)
        "3d_fullres_omnimamba_zonly": conv_like("strokebench.nets.omnimamba.PlainConvUNetOmniMamba", omni_axes=["z"]),
        "3d_fullres_omnimamba_nolocal": conv_like("strokebench.nets.omnimamba.PlainConvUNetOmniMamba", omni_local_branch=False),
        "3d_fullres_omnimamba_bottleneck": conv_like("strokebench.nets.omnimamba.PlainConvUNetOmniMamba",
                                                     omni_stages=[arch["arch_kwargs"]["n_stages"] - 1]),
        # the first manuscript's network, unchanged
        "3d_fullres_omnimamba_orig": standalone("strokebench.nets.omnimamba.OmniMambaUNetOriginal",
                                                features=[32, 64, 128, 256, 512]),
        "3d_fullres_swinunetr": standalone("strokebench.nets.reference.SwinUNETRNet", feature_size=48),
        "3d_fullres_unetr": standalone("strokebench.nets.reference.UNETRNet", img_size=patch),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(paths.DATASETS), default="dwi_adc")
    args = ap.parse_args()
    _, name = paths.DATASETS[args.dataset]
    folder = paths.NNUNET_PREPROCESSED / name
    plans_path = folder / "nnUNetPlans.json"
    pristine = folder / "nnUNetPlans.planner_original.json"
    if not pristine.exists():
        shutil.copy(plans_path, pristine)

    plans = json.load(open(pristine))
    base = plans["configurations"][BASE]
    extras = extra_configurations(base)
    clash = set(extras) & set(plans["configurations"])
    assert not clash, f"planner already defines {clash}"
    plans["configurations"].update(extras)
    assert plans["configurations"][BASE] == json.load(open(pristine))["configurations"][BASE]
    with open(plans_path, "w") as f:
        json.dump(plans, f, indent=4)

    shutil.copy(paths.SPLITS_JSON, folder / "splits_final.json")
    print(f"{name}: configurations now {sorted(plans['configurations'])}")
    print(f"installed folds -> {folder / 'splits_final.json'}")


if __name__ == "__main__":
    main()
