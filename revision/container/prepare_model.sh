#!/bin/bash
# Copy the frozen nnU-Net cross-validation models (final checkpoints only) into container/model/.
# Run from the repository root. model/ is not version-controlled (about 1 GB).
set -euo pipefail
src="nnunet/results/Dataset522_ISLES22DwiAdc/StrokeTrainer__nnUNetPlans__3d_fullres"
dst="revision/container/model"
mkdir -p "$dst"
cp "$src/plans.json" "$src/dataset.json" "$dst/"
for f in 0 1 2 3 4; do
  mkdir -p "$dst/fold_$f"
  cp "$src/fold_$f/checkpoint_final.pth" "$dst/fold_$f/"
done
du -sh "$dst"
