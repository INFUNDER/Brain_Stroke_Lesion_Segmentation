"""Checks that need a GPU (the state-space operator is CUDA-only). Skipped on the login node."""
import pytest
import torch

from strokebench import config, paths

paths.set_nnunet_env()
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def test_cudnn_conv3d_works():
    assert torch.backends.cudnn.enabled and torch.backends.cudnn.is_available()
    x = torch.randn(2, 4, 32, 32, 32, device="cuda", requires_grad=True)
    y = torch.nn.Conv3d(4, 8, 3, padding=1).cuda()(x)
    y.mean().backward()
    assert torch.isfinite(x.grad).all()


def test_omnimamba_layer_forward_backward_under_autocast():
    from strokebench.nets.omnimamba import OmniMambaLayer
    layer = OmniMambaLayer(dim=32).cuda()
    x = torch.randn(2, 32, 40, 48, 40, device="cuda", requires_grad=True)
    with torch.autocast("cuda"):
        y = layer(x.half())
    assert y.shape == x.shape and y.dtype == torch.float32 and torch.isfinite(y).all()
    y.float().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())


def test_scan_directions_differ():
    """Each scan order must see the volume differently, otherwise the three branches are redundant."""
    from strokebench.nets.omnimamba import OmniMambaLayer
    torch.manual_seed(0)
    x = torch.randn(1, 16, 8, 10, 12, device="cuda")
    outs = {}
    for axis in ("z", "y", "x"):
        torch.manual_seed(1)                                       # identical weights for every axis
        layer = OmniMambaLayer(dim=16, axes=(axis,), local_branch=False).cuda().eval()
        with torch.no_grad():
            outs[axis] = layer(x)
    assert not torch.allclose(outs["z"], outs["y"], atol=1e-4)
    assert not torch.allclose(outs["z"], outs["x"], atol=1e-4)
    assert not torch.allclose(outs["y"], outs["x"], atol=1e-4)


@pytest.mark.parametrize("model", list(config.MODELS))
def test_every_model_builds_and_runs_on_the_study_patch(model):
    from strokebench.benchmark import build_trainer
    trainer = build_trainer(model, torch.device("cuda"))
    net = trainer.network.eval()
    trainer.set_deep_supervision_enabled(False)
    patch = trainer.configuration_manager.patch_size
    x = torch.randn(1, trainer.num_input_channels, *patch, device="cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        y = net(x)
    assert tuple(y.shape) == (1, 2, *patch)
    assert torch.isfinite(y.float()).all()
