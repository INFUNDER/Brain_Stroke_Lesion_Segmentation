#!/bin/bash
# Build the self-contained environment used for every experiment in the revision.
# It does not read from or write to any existing conda environment's packages:
# the old GPU-env mixes CUDA 11/12/13 libraries, which is what caused the
# CUDNN_STATUS_NOT_INITIALIZED failures in earlier jobs.
#
# Usage (login node, needs internet):  bash revision/env/setup_env.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$ROOT/.venv"
BASE_PY="${BASE_PY:-/home/ronit.28010/.conda/envs/GPU-env/bin/python}"   # interpreter only (CPython 3.10)

# CUDA 12 toolkit: only needed if mamba-ssm has to be compiled from source.
if [ -d /home/soft/cuda-12.8 ]; then
  export CUDA_HOME=/home/soft/cuda-12.8
  export PATH="$CUDA_HOME/bin:$PATH"
fi

rm -rf "$VENV"
"$BASE_PY" -m venv "$VENV"
PIP="$VENV/bin/python -m pip --disable-pip-version-check"
$PIP install --quiet --upgrade pip wheel "setuptools<81" packaging ninja

# 1. One consistent GPU stack: PyTorch 2.4.1 built for CUDA 12.1, with its own bundled CUDA/cuDNN.
$PIP install --quiet torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# 2. Everything else, pinned so the resolver cannot replace the GPU stack.
cat > "$ROOT/revision/env/constraints.txt" <<'EOF'
torch==2.4.1
torchvision==0.19.1
numpy<2
transformers==4.44.2
EOF
$PIP install --quiet -c "$ROOT/revision/env/constraints.txt" \
  "nnunetv2==2.8.1" "monai==1.5.2" connected-components-3d statsmodels pytest ptflops

# 3. State-space layers (prebuilt wheel if the project publishes one for this stack, else compiled with nvcc).
$PIP install --quiet -c "$ROOT/revision/env/constraints.txt" --no-build-isolation causal-conv1d
$PIP install --quiet -c "$ROOT/revision/env/constraints.txt" --no-build-isolation mamba-ssm

# 4. Make the study code importable, and let nnU-Net's command-line tools find the study's trainers
#    (nnU-Net only searches its own package folder for trainer classes).
SITE="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$ROOT/revision" > "$SITE/strokebench.pth"
ln -sfn "$ROOT/revision/strokebench/trainers.py" "$SITE/nnunetv2/training/nnUNetTrainer/variants/strokebench_trainers.py"

$PIP freeze > "$ROOT/revision/env/requirements.lock.txt"
echo "SETUP_DONE"
