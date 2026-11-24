"""CPU checks of everything that does not need the CUDA state-space operator."""
import json

import numpy as np
import pytest
import torch
from torch import nn

from strokebench import config, paths
from strokebench.nets.omnimamba import SCAN_ORDERS, OmniMambaLayer

paths.set_nnunet_env()


class CumSum(nn.Module):
    """Stand-in for Mamba: a causal sequence operator whose output depends on scan order."""

    def __init__(self, dim):
        super().__init__()

    def forward(self, seq):                      # (B, N, C)
        return torch.cumsum(seq, dim=1)


def reference_scan(x: np.ndarray, axis: str) -> np.ndarray:
    """Independent numpy implementation: running sum along the documented scan order."""
    b, c, d0, d1, d2 = x.shape
    out = np.zeros_like(x)
    # order in which the three spatial axes are nested, slowest first
    nest = {"z": (0, 1, 2), "y": (1, 2, 0), "x": (2, 0, 1)}[axis]
    sizes = (d0, d1, d2)
    for bi in range(b):
        for ci in range(c):
            total = 0.0
            for i in range(sizes[nest[0]]):
                for j in range(sizes[nest[1]]):
                    for k in range(sizes[nest[2]]):
                        idx = [0, 0, 0]
                        idx[nest[0]], idx[nest[1]], idx[nest[2]] = i, j, k
                        total += x[bi, ci, idx[0], idx[1], idx[2]]
                        out[bi, ci, idx[0], idx[1], idx[2]] = total
    return out


@pytest.mark.parametrize("axis", ["z", "y", "x"])
def test_scan_order_and_inverse_permutation(axis):
    torch.manual_seed(0)
    layer = OmniMambaLayer(dim=2, axes=(axis,), local_branch=False, sequence_module=CumSum)
    layer.norm = nn.Identity()
    x = torch.randn(1, 2, 3, 4, 5, dtype=torch.float64).float()
    got = layer(x).double().numpy() - x.double().numpy()            # remove the identity path
    want = reference_scan(x.double().numpy(), axis)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_permutations_are_inverses():
    x = torch.arange(2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6)
    for perm, inverse in SCAN_ORDERS.values():
        assert torch.equal(x.permute(*perm).permute(*inverse), x)


def test_layer_sums_all_branches_and_keeps_shape():
    layer = OmniMambaLayer(dim=4, sequence_module=CumSum)
    x = torch.randn(2, 4, 4, 6, 5)
    y = layer(x)
    assert y.shape == x.shape and y.dtype == torch.float32
    assert set(layer.scans.keys()) == {"z", "y", "x"}
    with pytest.raises(ValueError):
        OmniMambaLayer(dim=4, axes=("q",), sequence_module=CumSum)


class ScaledCumSum(nn.Module):
    """Causal stand-in with a trainable parameter, so parameter gradients can be compared."""

    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.5, 1.5, dim))

    def forward(self, seq):
        return torch.cumsum(seq * self.weight, dim=1)


def test_gradient_checkpointing_does_not_change_outputs_or_gradients():
    torch.manual_seed(0)
    x0 = torch.randn(2, 4, 3, 4, 5)
    results = []
    for use_checkpoint in (False, True):
        torch.manual_seed(1)
        layer = OmniMambaLayer(dim=4, sequence_module=ScaledCumSum, grad_checkpoint=use_checkpoint)
        layer.train()
        x = x0.clone().requires_grad_(True)
        y = layer(x)
        (y ** 2).sum().backward()
        grads = torch.cat([p.grad.flatten() for p in layer.parameters()])
        results.append((y.detach(), x.grad.detach(), grads))
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_layer_runs_in_float32_under_autocast():
    layer = OmniMambaLayer(dim=4, sequence_module=CumSum)
    x = torch.randn(1, 4, 4, 4, 4).half()
    assert layer(x).dtype == torch.float32


def _plans():
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    name = paths.DATASETS["dwi_adc"][1]
    return PlansManager(str(paths.NNUNET_PREPROCESSED / name / "nnUNetPlans.json"))


def _build(configuration: str, deep_supervision: bool):
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    cm = _plans().get_configuration(configuration)
    return cm, get_network_from_plans(cm.network_arch_class_name, cm.network_arch_init_kwargs,
                                      cm.network_arch_init_kwargs_req_import, 2, 2, allow_init=True,
                                      deep_supervision=deep_supervision)


def test_all_configurations_share_the_data_pipeline():
    pm = _plans()
    base = pm.get_configuration("3d_fullres")
    used = {cfg for plans, cfg, _ in config.ALL_MODELS.values() if plans == config.PLANS}
    for name in used:
        cm = pm.get_configuration(name)
        for key in ("data_identifier", "patch_size", "batch_size", "spacing", "normalization_schemes"):
            assert getattr(cm, key) == getattr(base, key), (name, key)


def test_resenc_plans_share_the_data_pipeline():
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    name = paths.DATASETS["dwi_adc"][1]
    res = PlansManager(str(paths.NNUNET_PREPROCESSED / name / f"{config.RESENC_PLANS}.json")).get_configuration("3d_fullres")
    base = _plans().get_configuration("3d_fullres")
    for key in ("data_identifier", "patch_size", "batch_size", "spacing"):
        assert getattr(res, key) == getattr(base, key), key


def test_planner_configuration_is_untouched():
    name = paths.DATASETS["dwi_adc"][1]
    folder = paths.NNUNET_PREPROCESSED / name
    now = json.load(open(folder / "nnUNetPlans.json"))["configurations"]["3d_fullres"]
    original = json.load(open(folder / "nnUNetPlans.planner_original.json"))["configurations"]["3d_fullres"]
    assert now == original


@pytest.mark.parametrize("configuration", ["3d_fullres_swinunetr", "3d_fullres_unetr"])
def test_transformer_wrappers_forward(configuration):
    cm, net = _build(configuration, deep_supervision=False)
    net.eval()
    x = torch.randn(1, 2, *cm.patch_size)
    with torch.no_grad():
        y = net(x)
    assert tuple(y.shape) == (1, 2, *cm.patch_size)
    net.decoder.deep_supervision = False          # the attribute nnU-Net toggles must exist


def test_plain_unet_forward_with_deep_supervision():
    cm, net = _build("3d_fullres", deep_supervision=True)
    y = net(torch.randn(1, 2, *cm.patch_size))
    assert isinstance(y, (list, tuple)) and len(y) == 4
    assert tuple(y[0].shape) == (1, 2, *cm.patch_size)


def test_trainers_are_discoverable_by_nnunet():
    import nnunetv2
    from batchgenerators.utilities.file_and_folder_operations import join
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    base = {t for _, _, t in config.MODELS.values()} | set(config.ALTERNATE_TRAINER.values())
    needed = base | {t for _, _, t in config.ALL_MODELS.values()} | {"StrokeTrainerReplicate2"}
    for trainer in needed | {t + "Debug" for t in base}:
        cls = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"), trainer,
                                          "nnunetv2.training.nnUNetTrainer")
        assert cls is not None, trainer


def test_schedules_and_memory_table_are_consistent():
    assert config.NUM_EPOCHS_LONG == 1000 and config.NUM_EPOCHS <= config.NUM_EPOCHS_LONG
    assert set(config.GPU_MEMORY_GB) == set(config.ALL_MODELS)
    main = {k: v for k, v in config.MODELS.items()}
    for key, (plans, configuration, trainer) in config.LONG_SCHEDULE_MODELS.items():
        base = key.replace("_long", "")
        assert main[base][:2] == (plans, configuration) and trainer == "StrokeTrainerLong"


def test_warmup_poly_schedule():
    from strokebench.trainers import WarmupPolyLRScheduler
    opt = torch.optim.AdamW([nn.Parameter(torch.zeros(1))], lr=1e-4)
    sched = WarmupPolyLRScheduler(opt, 1e-4, max_steps=1000, warmup_steps=10)
    lrs = []
    for epoch in range(1000):
        sched.step(epoch)
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[0] == pytest.approx(1e-5)
    assert lrs[9] == pytest.approx(1e-4 * (1 - 9 / 1000) ** 0.9)
    assert all(a >= b for a, b in zip(lrs[9:], lrs[10:]))
    assert lrs[-1] > 0
