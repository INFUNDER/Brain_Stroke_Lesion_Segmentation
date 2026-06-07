#!/bin/bash
#PBS -N mamba_unet
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=2:mem=64g
#PBS -j oe

cd $PBS_O_WORKDIR

source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate cod-gpu
module load cuda

python TRAIN_MAMBA_v5.py
