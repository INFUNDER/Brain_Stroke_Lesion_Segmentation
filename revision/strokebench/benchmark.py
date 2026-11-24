#!/usr/bin/env python3
"""Measure what each model actually costs on the study's hardware.

For every model the real trainer object is built from the plans, so the numbers
describe exactly what is trained: parameter count, time and peak GPU memory of a
full training step (batch and patch size from the plans, mixed precision, the
real loss and optimiser), and time and memory of one inference forward pass.
Inputs are synthetic; no patient data and no result folders are touched.
"""
import argparse
import json
import os
import socket
import time

from strokebench import config, paths

paths.set_nnunet_env()
os.environ["nnUNet_results"] = str(paths.NNUNET / "results_benchmark")   # never write into the study's results

import numpy as np                                                          # noqa: E402
import torch                                                                # noqa: E402
import torch.nn.functional as F                                             # noqa: E402
from nnunetv2.run.run_training import get_trainer_from_args                 # noqa: E402


def build_trainer(model: str, device: torch.device):
    """Build the trainer through nnU-Net's own entry point, i.e. the code path of nnUNetv2_train."""
    plans_name, configuration, trainer_name = config.ALL_MODELS[model]
    trainer = get_trainer_from_args(paths.DATASETS["dwi_adc"][1], configuration, 0, trainer_name, plans_name,
                                    False, device=device)
    trainer.initialize()
    return trainer


def synthetic_batch(trainer, batch_size: int) -> dict:
    patch = list(trainer.configuration_manager.patch_size)
    g = torch.Generator().manual_seed(0)
    data = torch.randn(batch_size, trainer.num_input_channels, *patch, generator=g)
    seg = (torch.rand(batch_size, 1, *patch, generator=g) > 0.98).float()
    if trainer.enable_deep_supervision:
        scales = trainer._get_deep_supervision_scales()
        target = [seg if all(s == 1 for s in sc) else F.interpolate(seg, scale_factor=tuple(float(s) for s in sc), mode="nearest")
                  for sc in scales]
    else:
        target = seg
    return {"data": data, "target": target}


def measure(model: str, device: torch.device, n_train: int, n_infer: int, batch_size: int = 0) -> dict:
    """`batch_size` 0 means the plan's batch size. A smaller value is only used to find out
    how memory and time scale when the planned batch does not fit on a shared GPU."""
    out = {"model": model, "plans": config.ALL_MODELS[model][0], "configuration": config.ALL_MODELS[model][1],
           "trainer": config.ALL_MODELS[model][2]}
    trainer = build_trainer(model, device)
    net = trainer.network
    out["parameters_M"] = sum(p.numel() for p in net.parameters()) / 1e6
    out["planned_batch_size"] = int(trainer.configuration_manager.batch_size)
    out["batch_size"] = batch_size or out["planned_batch_size"]
    out["patch_size"] = list(trainer.configuration_manager.patch_size)
    cuda = device.type == "cuda"
    if cuda:
        free, total = torch.cuda.mem_get_info()
        out["gpu_free_GB_at_start"] = free / 2 ** 30

    def sync():
        if cuda:
            torch.cuda.synchronize()

    try:
        batch = synthetic_batch(trainer, out["batch_size"])
        net.train()
        trainer.set_deep_supervision_enabled(trainer.enable_deep_supervision)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        warmup, losses = 5, []
        for i in range(warmup + n_train):
            if i == warmup:
                sync()
                t0 = time.perf_counter()
            losses.append(float(trainer.train_step(batch)["loss"]))
        sync()
        out["train_s_per_iter"] = (time.perf_counter() - t0) / n_train
        out["train_peak_mem_GB"] = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else None
        out["loss_first"], out["loss_last"] = losses[0], losses[-1]
        out["loss_finite"] = bool(np.isfinite(losses).all())
        # GPU time for 1000 epochs of 250 iterations at the planned batch size (linear in batch size)
        out["hours_per_1000_epochs"] = (out["train_s_per_iter"] * out["planned_batch_size"] / out["batch_size"]) * 250 * 1000 / 3600

        net.eval()
        trainer.set_deep_supervision_enabled(False)
        x = batch["data"][:1].to(device)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.autocast(device.type, enabled=cuda):
            for i in range(3 + n_infer):
                if i == 3:
                    sync()
                    t0 = time.perf_counter()
                y = net(x)
            sync()
        out["infer_s_per_patch"] = (time.perf_counter() - t0) / n_infer
        out["infer_peak_mem_GB"] = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else None
        out["infer_output_finite"] = bool(torch.isfinite(y.float()).all())
        out["status"] = "ok"
    except torch.cuda.OutOfMemoryError:
        out["status"] = "out_of_memory"
    finally:
        del trainer, net
        if cuda:
            torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(config.MODELS))
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--n-infer", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fallback-batches", type=int, nargs="*", default=[],
                    help="smaller batch sizes to try, in order, if the planned batch runs out of memory")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default=str(paths.REVISION / "results" / "benchmark.json"))
    args = ap.parse_args()
    device = torch.device(args.device)
    rows = []
    for model in args.models:
        for batch in [0] + args.fallback_batches:          # stop at the first batch size that fits
            row = measure(model, device, args.n_train, args.n_infer, batch)
            rows.append(row)
            print(json.dumps(row), flush=True)
            if row["status"] == "ok":
                break
    meta = {"host": socket.gethostname(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "iterations_per_epoch": 250, "note": args.note}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "models": rows}, f, indent=1)
    print(f"\n{'model':16s} {'batch':>5s} {'params M':>9s} {'train s/it':>11s} {'train GB':>9s} {'h/1000ep':>9s} {'infer s':>8s} status")
    for r in rows:
        if r["status"] == "ok":
            print(f"{r['model']:16s} {r['batch_size']:5d} {r['parameters_M']:9.2f} {r['train_s_per_iter']:11.3f} "
                  f"{r['train_peak_mem_GB']:9.1f} {r['hours_per_1000_epochs']:9.1f} {r['infer_s_per_patch']:8.3f} "
                  f"ok (loss finite: {r['loss_finite']})")
        else:
            print(f"{r['model']:16s} {r['batch_size']:5d} {r['parameters_M']:9.2f} {'-':>11s} {'-':>9s} {'-':>9s} {'-':>8s} "
                  f"{r['status']} (GPU had {r.get('gpu_free_GB_at_start', float('nan')):.0f} GB free)")


if __name__ == "__main__":
    main()
