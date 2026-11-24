# Revision workspace

Everything for the re-done study lives here. Nothing in this folder uses the numbers, checkpoints or labels of the first manuscript.

Read first:
- `AUDIT.md`: what was wrong with the first study, with evidence.
- `PROTOCOL.md`: the analysis plan, fixed before any full training run.

## Layout

| Path | Purpose |
|---|---|
| `01_build_corrected_labels.py` | rebuilds the 250 labels from the original ISLES masks |
| `env/` | `setup_env.sh` builds the self-contained environment; `requirements.lock.txt` pins it; `activate.sh` activates it |
| `strokebench/` | study code: case table, folds, dataset builder, networks, trainers, metrics, scoring, conformal intervals, statistics |
| `strokebench/third_party/umamba/` | official U-Mamba network, vendored unmodified (Apache-2.0) |
| `data/` | `cases.csv` (per-subject table) and `splits_final.json` (the folds every model uses) |
| `tests/` | unit tests; `test_gpu.py` runs only on a GPU node |
| `hpc/` | PBS job scripts and task lists |
| `results/` | benchmark tables and per-subject scores |
| `logs/` | job and training logs (not version-controlled) |

## Reproduce from scratch

```bash
bash revision/env/setup_env.sh                      # once, on the login node (needs internet)
source revision/env/activate.sh
python revision/01_build_corrected_labels.py        # writes ISLES2022_Corrected/
python -m strokebench.cases                         # revision/data/cases.csv
python -m strokebench.make_splits                   # revision/data/splits_final.json
python -m strokebench.build_dataset --dataset dwi_adc
nnUNetv2_plan_and_preprocess -d 522 --verify_dataset_integrity -c 3d_fullres
nnUNetv2_plan_experiment -d 522 -pl nnUNetPlannerResEncM
python -m strokebench.extend_plans --dataset dwi_adc  # adds the comparison backbones, installs the folds
python -m pytest revision/tests -q
```

Train (from the repository root; jobs are resumable and skip finished folds):

```bash
qsub -N sb_x -l select=1:ncpus=10:ngpus=1:mem=40gb:host=gpu01 -o revision/logs/job_x.log \
     -v TASKFILE=revision/hpc/tasks/<list>.txt,nnUNet_n_proc_DA=8 revision/hpc/train.pbs
```

Check progress of every run, and read a running job's own log:

```bash
python -m strokebench.status
tail revision/logs/live_sb_*.log
```

Score a model's out-of-fold predictions once all five folds are done:

```bash
python -m strokebench.evaluate --model nnunet_long
```

Task lists in `hpc/tasks/`: `ref_nnunet_long.txt`, `main_a.txt`, `main_b.txt` (the three high-cost models) and `ref_resenc_long.txt` were all submitted on 2026-11-15; `tier2_cheap.txt` (protocol-shift folds and replicate runs for the low-cost models) on 2026-11-18.

## Cluster notes

- The `gpu` queue does not restrict a job to its own GPUs and the GPUs are shared between users. Job scripts therefore pick the GPU with the most free memory (`hpc/common.sh`) and retry after a crash.
- Timings measured on a shared GPU are upper bounds. The efficiency table in the paper must be re-measured on an idle GPU.
- Pin jobs to `host=gpu01` and request at most 40 GB of RAM (see the header of `hpc/train.pbs` for why).
- The old `GPU-env` conda environment mixes CUDA 11, 12 and 13 libraries. It is not used here.
