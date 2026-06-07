#!/bin/bash
#PBS -N mamba_unet
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=64g
#PBS -l walltime=72:00:00
#PBS -j oe

cd $PBS_O_WORKDIR

source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate cod-gpu
module load cuda

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python new_work/train.py \
  --data_root ./isless2022dataset/ISLES-2022_notformatted \
  --cache_dir ./cache_preprocessed \
  --output_dir ./runs/smunet_fold0 \
  --fold 0 \
  --epochs 300 \
  --batch_size 2 \
  --patch_size 96 128 128 \
  --compile   